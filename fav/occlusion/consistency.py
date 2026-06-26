"""Forward-backward flow consistency / occlusion detection.

Vectorized port of ``consistencyChecker/consistencyChecker.cpp``
(``checkConsistency`` + ``computeCorners``). Produces a reliability map in
``[0, 255]`` where ``0`` marks occluded pixels (flow not trustworthy) and ``255``
marks reliable pixels (motion boundaries are kept reliable, matching the released
``MOTION_BOUNDARIE_VALUE = 255``).

All constants are preserved verbatim:

* occluded where ``|c - a|^2 >= 0.01*(|flow1|^2 + |flow2@b|^2) + structureTerm + 0.5``
* motion boundary where ``|grad flow1|^2 > 0.01*|flow1|^2 + 0.002``
* derivative kernel ``[-0.5, 0, 0.5]`` (CDerivative(3))
* structure term ``(4/avg) * max(0, avg/2 - structure(a))`` with the Harris
  smallest-eigenvalue map normalized to ``[0, 1]``

The Harris second-moment smoothing uses a separable Gaussian of std ``rho`` as a
close, modern stand-in for the legacy recursive (Deriche-style) ``recursiveSmooth``;
the residual difference is negligible after the downstream ``min_filter`` erosion.
Pass ``use_structure=False`` for the exact no-image path of the C++ tool, which
needs no smoothing at all.

Flow tensors are ``(2, H, W)`` in native ``.flo`` order: channel 0 = u (x), 1 = v (y).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

MOTION_BOUNDARY_VALUE = 255.0
_DERIV = (-0.5, 0.0, 0.5)


def _central_diff(x: torch.Tensor, axis: str) -> torch.Tensor:
    """Central difference [-0.5,0,0.5] along 'x' or 'y' with replicate borders.

    ``x`` is ``(N, C, H, W)``; returns same shape.
    """
    c = x.shape[1]
    k = torch.tensor(_DERIV, dtype=x.dtype, device=x.device)
    if axis == "x":
        kernel = k.view(1, 1, 1, 3).repeat(c, 1, 1, 1)
        xp = F.pad(x, (1, 1, 0, 0), mode="replicate")
        return F.conv2d(xp, kernel, groups=c)
    else:
        kernel = k.view(1, 1, 3, 1).repeat(c, 1, 1, 1)
        xp = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return F.conv2d(xp, kernel, groups=c)


def compute_corners(image: torch.Tensor, rho: float = 3.0) -> torch.Tensor:
    """Harris smallest-eigenvalue corner strength, normalized to ``[0, 1]``.

    Args:
        image: ``(C, H, W)`` content image (any channel count; summed over C).
        rho: Gaussian smoothing std for the second-moment matrix.
    """
    if image.dim() != 3:
        raise ValueError(f"expected (C,H,W) image, got {tuple(image.shape)}")
    x = image.unsqueeze(0).to(torch.float32)  # (1,C,H,W)
    dx = _central_diff(x, "x")
    dy = _central_diff(x, "y")
    # Second-moment matrix, summed over channels.
    dxx = (dx * dx).sum(dim=1, keepdim=True)
    dyy = (dy * dy).sum(dim=1, keepdim=True)
    dxy = (dx * dy).sum(dim=1, keepdim=True)
    dxx = _gaussian_blur(dxx, rho)
    dyy = _gaussian_blur(dyy, rho)
    dxy = _gaussian_blur(dxy, rho)
    a, b, c = dxx, dxy, dyy
    temp = 0.5 * (a + c)
    temp2 = (temp * temp + b * b - a * c).clamp_min(0.0)
    corners = temp - torch.sqrt(temp2)
    corners = corners.squeeze(0).squeeze(0)  # (H,W)
    cmin, cmax = corners.min(), corners.max()
    if (cmax - cmin) > 0:
        corners = (corners - cmin) / (cmax - cmin)
    else:
        corners = torch.zeros_like(corners)
    return corners


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    kx = g.view(1, 1, 1, -1)
    ky = g.view(1, 1, -1, 1)
    xp = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(xp, kx)
    xp = F.pad(x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(xp, ky)


def _bilinear_sample(field: torch.Tensor, bx: torch.Tensor, by: torch.Tensor):
    """Bilinear-sample a ``(H, W)`` field at float coords ``(bx, by)``.

    Returns ``(values, oob_mask)`` where ``oob_mask`` is True wherever the 2x2
    neighborhood leaves the grid (the C++ marks those pixels occluded).
    """
    h, w = field.shape
    x1 = torch.floor(bx)
    y1 = torch.floor(by)
    x1i = x1.long()
    y1i = y1.long()
    x2i = x1i + 1
    y2i = y1i + 1
    oob = (x1i < 0) | (x2i >= w) | (y1i < 0) | (y2i >= h)
    ax = (bx - x1).clamp(0, 1)
    ay = (by - y1).clamp(0, 1)
    xc1 = x1i.clamp(0, w - 1)
    xc2 = x2i.clamp(0, w - 1)
    yc1 = y1i.clamp(0, h - 1)
    yc2 = y2i.clamp(0, h - 1)
    flat = field.reshape(-1)

    def g(yy, xx):
        return flat[yy * w + xx]

    top = (1 - ax) * g(yc1, xc1) + ax * g(yc1, xc2)
    bot = (1 - ax) * g(yc2, xc1) + ax * g(yc2, xc2)
    return (1 - ay) * top + ay * bot, oob


def check_consistency(
    flow1: torch.Tensor,
    flow2: torch.Tensor,
    structure: torch.Tensor | None = None,
    motion_boundary_value: float = MOTION_BOUNDARY_VALUE,
) -> torch.Tensor:
    """Reliability map for ``flow1`` checked against ``flow2``.

    Args:
        flow1: ``(2, H, W)`` forward flow (u, v) from the reference frame.
        flow2: ``(2, H, W)`` backward flow (u, v) used for the round trip.
        structure: optional ``(H, W)`` Harris map (already normalized to [0,1]);
            if ``None`` the structure term is 0 (the no-image C++ path).
        motion_boundary_value: value written at motion boundaries (legacy 255).

    Returns:
        ``(H, W)`` float reliability map in ``[0, 255]``.
    """
    flow1 = flow1.to(torch.float32)
    flow2 = flow2.to(torch.float32)
    _, h, w = flow1.shape
    device = flow1.device

    ys = torch.arange(h, dtype=torch.float32, device=device)
    xs = torch.arange(w, dtype=torch.float32, device=device)
    ay, ax = torch.meshgrid(ys, xs, indexing="ij")

    u2 = flow1[0]
    v2 = flow1[1]
    bx = ax + u2
    by = ay + v2
    u, oob_u = _bilinear_sample(flow2[0], bx, by)
    v, _ = _bilinear_sample(flow2[1], bx, by)

    cx = bx + u
    cy = by + v

    if structure is not None:
        avg = float(structure.mean())
        if avg > 0:
            structure_term = (4.0 / avg) * torch.clamp_min(avg / 2.0 - structure, 0.0)
        else:
            structure_term = torch.zeros_like(u2)
    else:
        structure_term = torch.zeros_like(u2)

    round_trip = (cx - ax) ** 2 + (cy - ay) ** 2
    threshold = 0.01 * (u2 * u2 + v2 * v2 + u * u + v * v) + structure_term + 0.5
    occluded = (round_trip >= threshold) | oob_u

    # Motion edges from the gradient of flow1 (both components).
    f = flow1.unsqueeze(0)  # (1,2,H,W)
    fdx = _central_diff(f, "x")[0]
    fdy = _central_diff(f, "y")[0]
    motion_edge = fdx[0] ** 2 + fdx[1] ** 2 + fdy[0] ** 2 + fdy[1] ** 2
    boundary = motion_edge > (0.01 * (u2 * u2 + v2 * v2) + 0.002)

    reliable = torch.full((h, w), 255.0, dtype=torch.float32, device=device)
    reliable = torch.where(boundary, torch.full_like(reliable, motion_boundary_value), reliable)
    reliable = torch.where(occluded, torch.zeros_like(reliable), reliable)
    return reliable.clamp(0.0, 255.0)


def compute_reliability(
    flow1: torch.Tensor,
    flow2: torch.Tensor,
    content_image: torch.Tensor | None = None,
    rho: float = 3.0,
) -> torch.Tensor:
    """Convenience: optionally compute the Harris structure, then check.

    ``flow1``/``flow2`` are ``(2, H, W)``; ``content_image`` is ``(C, H, W)``.
    """
    structure = None
    if content_image is not None:
        # Keep the structure map on the flow's device (the content image may be
        # loaded on a different device than the estimator's flow output).
        structure = compute_corners(content_image.to(flow1.device), rho=rho)
    return check_consistency(flow1, flow2, structure=structure)
