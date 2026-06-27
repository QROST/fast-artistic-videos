"""Cross-face seam consistency for 360 video (ports ``fast_artistic_video_vr.lua``).

Within a single cube face, temporal consistency is the planar recipe (warp the
previous frame's same face by its optical flow). But adjacent faces share a
*spatial* seam: a strip of one face re-appears, perspective-distorted, on the
edge of its neighbour. When the 6 faces are stylized independently the seam
flickers because each face invents its own texture there.

The legacy code fixes this by processing faces in a fixed order
(:data:`fav.vr.cubemap.PROC_ORDER` = ``[6,1,2,5,3,4]``) and, for every face,
building a **border prior** -- the already-stylized neighbour faces warped
through the perspective maps in :mod:`fav.vr.perspective` into the current
face's overlap strip. That border prior is blended into the per-face temporal
prior over the overlap region using linear *gradient masks*, and the overlap is
marked certain (we have an exact correspondence there from the neighbour).

This module ports that geometry faithfully and operates in the same ``(1,C,H,W)``
tensor convention as the rest of ``fav``. Everything here is pure tensor logic
(device/dtype-agnostic, no model), so it is fully unit-testable; the seam-aware
stylization loop that consumes it lives in :mod:`fav.vr.stylize_vr`.

Index convention: faces are addressed by their **processing position** ``p`` in
``0..5`` (``p`` equals the legacy ``mode = (i-1) % 6``). ``segments[p]`` is the
already-stylized face that was processed ``p``-th this timestep, i.e. the face
``PROC_ORDER[p]``. Neighbour lookups below use the legacy ``last_segments``
1-based indices shifted to 0-based:

* ``segments[0]`` = face 6 (front), processed first
* ``segments[1]`` = face 1 (bottom)
* ``segments[2]`` = face 2 (top)
* ``segments[3]`` = face 5 (back)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fav.warp.grid_sample import warp
from fav.vr.perspective import (
    make_perspective_warp_map_left,
    make_perspective_warp_map_right,
    make_perspective_warp_map_top,
    make_perspective_warp_map_bottom,
)


# ---------------------------------------------------------------------------
# Gradient masks (port of utils.lua make_gradient_mask_*).
# Each is a 1-D linear ramp i/(size+1) expanded across the orthogonal axis.
# ---------------------------------------------------------------------------

def make_gradient_mask_w_inc(h: int, w: int) -> torch.Tensor:
    """Increasing ramp ``(1..w)/(w+1)`` along width, shape ``(1,1,h,w)``."""
    ramp = torch.arange(1, w + 1, dtype=torch.float32) / (w + 1)
    return ramp.view(1, 1, 1, w).expand(1, 1, h, w).contiguous()


def make_gradient_mask_w_dec(h: int, w: int) -> torch.Tensor:
    """Decreasing ramp ``(w..1)/(w+1)`` along width, shape ``(1,1,h,w)``."""
    ramp = torch.arange(w, 0, -1, dtype=torch.float32) / (w + 1)
    return ramp.view(1, 1, 1, w).expand(1, 1, h, w).contiguous()


def make_gradient_mask_h_inc(h: int, w: int) -> torch.Tensor:
    """Increasing ramp ``(1..h)/(h+1)`` along height, shape ``(1,1,h,w)``."""
    ramp = torch.arange(1, h + 1, dtype=torch.float32) / (h + 1)
    return ramp.view(1, 1, h, 1).expand(1, 1, h, w).contiguous()


def make_gradient_mask_h_dec(h: int, w: int) -> torch.Tensor:
    """Decreasing ramp ``(h..1)/(h+1)`` along height, shape ``(1,1,h,w)``."""
    ramp = torch.arange(h, 0, -1, dtype=torch.float32) / (h + 1)
    return ramp.view(1, 1, h, 1).expand(1, 1, h, w).contiguous()


# ---------------------------------------------------------------------------
# Rotations (port of the Lua reverse_tensor/transpose rotate helpers).
# Operate on ``(N,C,H,W)`` tensors; H is dim 2, W is dim 3.
# ---------------------------------------------------------------------------

def rotate90(t: torch.Tensor) -> torch.Tensor:
    """Lua ``reverse(t:transpose(2,3), 2)`` -> ``t.transpose(2,3).flip(2)``."""
    return t.transpose(2, 3).flip(2)


def rotate_minus90(t: torch.Tensor) -> torch.Tensor:
    """Lua ``reverse(t:transpose(2,3), 3)`` -> ``t.transpose(2,3).flip(3)``."""
    return t.transpose(2, 3).flip(3)


def rotate180(t: torch.Tensor) -> torch.Tensor:
    """Lua ``reverse(reverse(t,2),3)`` -> ``t.flip(2).flip(3)``."""
    return t.flip(2).flip(3)


# ---------------------------------------------------------------------------
# Precomputed seam geometry.
# ---------------------------------------------------------------------------

@dataclass
class SeamGeometry:
    """Warp maps, coverage masks and gradient blend masks for one cube size.

    Built once from the face size (``hplus``/``wplus`` include the overlap) and
    the overlap width/height. ``warp_map_*`` are ``(1,2,H,W)`` displacement maps
    (same ``(dy,dx)`` convention as the temporal flow); ``mask_*`` and
    ``grad_mask_*`` are ``(1,1,H,W)``.
    """

    hplus: int
    wplus: int
    overlap_w: int
    overlap_h: int
    warp_map_left: torch.Tensor
    warp_map_right: torch.Tensor
    warp_map_top: torch.Tensor
    warp_map_bottom: torch.Tensor
    mask_left: torch.Tensor
    mask_right: torch.Tensor
    mask_top: torch.Tensor
    mask_bottom: torch.Tensor
    mask_all: torch.Tensor
    mask_all_div: torch.Tensor
    grad_mask_left: torch.Tensor
    grad_mask_right: torch.Tensor
    grad_mask_top: torch.Tensor
    grad_mask_bottom: torch.Tensor
    grad_mask_all: torch.Tensor
    grad_mask_left_right: torch.Tensor

    @classmethod
    def build(cls, hplus: int, wplus: int, overlap_w: int = 20, overlap_h: int = 20,
              device=None, dtype=torch.float32) -> "SeamGeometry":
        def _map(fn, *a):
            return fn(*a).unsqueeze(0).to(device=device, dtype=dtype)

        warp_map_left = _map(make_perspective_warp_map_left, hplus, overlap_w, wplus)
        warp_map_right = _map(make_perspective_warp_map_right, hplus, overlap_w, wplus)
        warp_map_top = _map(make_perspective_warp_map_top, wplus, overlap_h, hplus)
        warp_map_bottom = _map(make_perspective_warp_map_bottom, wplus, overlap_h, hplus)

        ones = torch.ones(1, 1, hplus, wplus, device=device, dtype=dtype)
        mask_left = warp(ones, warp_map_left)
        mask_right = warp(ones, warp_map_right)
        mask_top = warp(ones, warp_map_top)
        mask_bottom = warp(ones, warp_map_bottom)

        mask_sum = mask_left + mask_right + mask_top + mask_bottom
        mask_all_div = mask_sum.clamp(min=1.0)
        mask_all = mask_sum.clamp(max=1.0)

        # grad_width = overlap - 10 (legacy). The ramp occupies the inner part of
        # the overlap strip; the outermost 10 px stay at the ramp's edge value.
        grad_width_w = max(1, overlap_w - 10)
        grad_width_h = max(1, overlap_h - 10)

        def _to(t):
            return t.to(device=device, dtype=dtype)

        z_w = torch.zeros(1, 1, hplus, wplus - grad_width_w, device=device, dtype=dtype)
        grad_mask_left = torch.cat([_to(make_gradient_mask_w_dec(hplus, grad_width_w)), z_w], dim=3)
        grad_mask_right = torch.cat([z_w, _to(make_gradient_mask_w_inc(hplus, grad_width_w))], dim=3)
        z_h = torch.zeros(1, 1, hplus - grad_width_h, wplus, device=device, dtype=dtype)
        grad_mask_top = torch.cat([_to(make_gradient_mask_h_dec(grad_width_h, wplus)), z_h], dim=2)
        grad_mask_bottom = torch.cat([z_h, _to(make_gradient_mask_h_inc(grad_width_h, wplus))], dim=2)
        grad_mask_all = torch.maximum(torch.maximum(grad_mask_left, grad_mask_right),
                                      torch.maximum(grad_mask_top, grad_mask_bottom))
        grad_mask_left_right = torch.maximum(grad_mask_left, grad_mask_right)

        return cls(
            hplus, wplus, overlap_w, overlap_h,
            warp_map_left, warp_map_right, warp_map_top, warp_map_bottom,
            mask_left, mask_right, mask_top, mask_bottom, mask_all, mask_all_div,
            grad_mask_left, grad_mask_right, grad_mask_top, grad_mask_bottom,
            grad_mask_all, grad_mask_left_right,
        )


# ---------------------------------------------------------------------------
# Border prior, border certainty, and the temporal/border blend.
# ---------------------------------------------------------------------------

def make_border_prior(geom: SeamGeometry, segments: list, p: int):
    """Warp already-stylized neighbour faces into face ``p``'s overlap strip.

    ``segments[k]`` is the ``k``-th processed face this timestep as a
    ``(1,C,H,W)`` tensor (any channel count; pre-space or RGB). Returns
    ``(border, grad_mask)`` where ``border`` is ``(1,C,H,W)`` and ``grad_mask``
    is the matching ``(1,1,H,W)`` blend ramp (``None`` for the first face, which
    has no processed neighbour).

    Faithful port of the ``func_make_last_frame_warped`` neighbour branch.
    """
    div = geom.mask_all_div
    if p == 0:
        # Front: first face, no neighbour processed yet -> empty border.
        c = segments[0].shape[1] if segments and segments[0] is not None else 3
        z = torch.zeros(1, c, geom.hplus, geom.wplus,
                        device=geom.mask_all.device, dtype=geom.mask_all.dtype)
        return z, None
    if p == 1:
        border = warp(segments[0], geom.warp_map_left)
        return border, geom.grad_mask_right
    if p == 2:
        border = warp(segments[0], geom.warp_map_right)
        return border, geom.grad_mask_left
    if p == 3:
        border = warp(segments[1], geom.warp_map_left) + warp(segments[2], geom.warp_map_right)
        return border, geom.grad_mask_left_right
    if p == 4:
        border = (
            warp(rotate90(segments[1]), geom.warp_map_left) / div
            + warp(rotate_minus90(segments[2]), geom.warp_map_right) / div
            + warp(segments[3], geom.warp_map_top) / div
            + warp(rotate180(segments[0]), geom.warp_map_bottom) / div
        )
        return border, geom.grad_mask_all
    if p == 5:
        border = (
            warp(rotate_minus90(segments[1]), geom.warp_map_left) / div
            + warp(rotate90(segments[2]), geom.warp_map_right) / div
            + warp(rotate180(segments[0]), geom.warp_map_top) / div
            + warp(segments[3], geom.warp_map_bottom) / div
        )
        return border, geom.grad_mask_all
    raise ValueError(f"processing position p must be in 0..5, got {p}")


def make_border_cert(geom: SeamGeometry, p: int) -> torch.Tensor:
    """Border certainty ``(1,1,H,W)``: 1 over the seam strips that face ``p``
    receives an exact correspondence for from already-processed neighbours.

    Faithful port of ``func_load_cert``'s ``cert_border`` assembly.
    """
    cert = torch.zeros(1, 1, geom.hplus, geom.wplus,
                       device=geom.mask_left.device, dtype=geom.mask_left.dtype)
    if p in (1, 3, 4, 5):
        cert = torch.maximum(cert, geom.mask_left)
    if p in (2, 3, 4, 5):
        cert = torch.maximum(cert, geom.mask_right)
    if p in (4, 5):
        cert = torch.maximum(cert, geom.mask_top)
        cert = torch.maximum(cert, geom.mask_bottom)
    return cert


def blend_border(geom: SeamGeometry, p: int, temporal_prior: torch.Tensor,
                 border: torch.Tensor, cert: torch.Tensor) -> torch.Tensor:
    """Blend the cross-face ``border`` prior into the ``temporal_prior``.

    Over the overlap strip the border prior is trusted more where the temporal
    flow is unreliable (low ``cert``): the blend weight is
    ``max(grad_mask, ceil(grad_mask) * (1-cert)) * coverage_mask``. Outside the
    overlap the weight is ``grad_mask == 0`` so the temporal prior is kept.

    Faithful port of the ``i >= 7`` blend in ``func_make_last_frame_warped``.
    ``p == 0`` (front) has no neighbour, so the temporal prior passes through.
    """
    if p == 0:
        return temporal_prior
    grad_masks = [geom.grad_mask_right, geom.grad_mask_left, geom.grad_mask_left_right,
                  geom.grad_mask_all, geom.grad_mask_all]
    masks = [geom.mask_left, geom.mask_right, geom.mask_left + geom.mask_right,
             geom.mask_all, geom.mask_all]
    grad_mask = grad_masks[p - 1]
    coverage = masks[p - 1]
    cert_inv = 1.0 - cert
    mask = torch.maximum(grad_mask, torch.ceil(grad_mask) * cert_inv) * coverage
    anti_mask = 1.0 - mask
    return temporal_prior * anti_mask + border * mask


# Neighbour table for the post-timestep re-blend (``blend_other_sides``): for each
# processing position, the four neighbours to warp into its overlap, as
# ``(segment_index, rotation, warp_map_name)``. ``rotation`` is one of
# ``None / 'r90' / 'rm90' / 'r180'``. Indices are 0-based processing positions.
_REBLEND_NEIGHBOURS = {
    0: [(1, None, "right"), (2, None, "left"), (4, "r180", "bottom"), (5, "r180", "top")],
    1: [(0, None, "left"), (3, None, "right"), (4, "rm90", "bottom"), (5, "r90", "top")],
    2: [(0, None, "right"), (3, None, "left"), (4, "r90", "bottom"), (5, "rm90", "top")],
    3: [(1, None, "left"), (2, None, "right"), (4, None, "bottom"), (5, None, "top")],
    4: [(0, "r180", "bottom"), (1, "r90", "left"), (2, "rm90", "right"), (3, None, "top")],
    5: [(0, "r180", "top"), (1, "rm90", "left"), (2, "r90", "right"), (3, None, "bottom")],
}

_ROTATIONS = {None: lambda t: t, "r90": rotate90, "rm90": rotate_minus90, "r180": rotate180}


def reblend_all_faces(geom: SeamGeometry, segments: list) -> list:
    """Re-blend every final face with its four warped neighbours (``blend_other_sides``).

    After a timestep's 6 faces are stylized, each face is blended with all four
    of its neighbours over the ``grad_mask_all`` overlap (neighbours combined via
    ``combineSides`` = sum of warps divided by ``mask_all_div``). The result feeds
    both the next timestep's temporal prior and the saved output, suppressing
    residual seams the single-pass border prior leaves behind.

    ``segments`` is the list of 6 stylized faces in processing order (``(1,C,H,W)``
    each). Returns a new list of 6 re-blended faces in the same order.
    """
    maps = {"left": geom.warp_map_left, "right": geom.warp_map_right,
            "top": geom.warp_map_top, "bottom": geom.warp_map_bottom}
    div = geom.mask_all_div
    grad = geom.grad_mask_all
    anti = 1.0 - grad
    out = []
    for p in range(6):
        border = None
        for seg_idx, rot, map_name in _REBLEND_NEIGHBOURS[p]:
            term = warp(_ROTATIONS[rot](segments[seg_idx]), maps[map_name]) / div
            border = term if border is None else border + term
        out.append(segments[p] * anti + border * grad)
    return out


def seam_prior_and_cert(geom: SeamGeometry, segments: list, p: int,
                        temporal_prior: torch.Tensor | None,
                        occlusion_cert: torch.Tensor | None,
                        blend: bool = True, occlusions_min_filter: int = 7):
    """High-level orchestrator mirroring ``func_make_last_frame_warped`` +
    ``func_load_cert`` for one face.

    The certainty is assembled exactly as the legacy core does: the occlusion map
    and the border strips are combined (``max``) *first*, then min-filtered once,
    and that single combined+filtered cert is used both as the blend weight's
    ``1-cert`` term and as the returned model-input mask. So on a covered seam
    strip ``cert == 1 -> cert_inv == 0`` and the blend collapses to the gradient
    ramp, matching the reference.

    Args:
        segments: already-stylized faces this timestep (processing order),
            ``(1,C,H,W)`` each (in RGB space, as in the legacy ``last_segments``).
        p: processing position of the current face (``0..5``).
        temporal_prior: the current face's previous output warped by its own
            optical flow (``(1,C,H,W)``), or ``None`` for the very first frame.
        occlusion_cert: the *raw* temporal occlusion certainty (``(1,1,H,W)``), or
            ``None`` (treated as all-zero) on the first frame. Not pre-filtered --
            the min-filter is applied here to the combined cert.
        blend: blend the border into the temporal prior (legacy ``i >= 7``); when
            ``False`` (first timestep) the prior is just the raw border.
        occlusions_min_filter: erosion radius applied to the combined cert.

    Returns:
        ``(prior, cert)``: the prior to feed as the warped-previous input and the
        combined+min-filtered certainty.
    """
    from fav.occlusion.filters import min_filter

    border, _ = make_border_prior(geom, segments, p)
    cert_border = make_border_cert(geom, p)

    if blend and temporal_prior is not None:
        occ = occlusion_cert if occlusion_cert is not None else torch.zeros_like(cert_border)
        cert = torch.maximum(occ, cert_border)
        if occlusions_min_filter and occlusions_min_filter > 1:
            cert = min_filter(cert, occlusions_min_filter)
        prior = blend_border(geom, p, temporal_prior, border, cert)
    else:
        prior = border
        cert = cert_border
        if occlusions_min_filter and occlusions_min_filter > 1:
            cert = min_filter(cert, occlusions_min_filter)
    return prior, cert
