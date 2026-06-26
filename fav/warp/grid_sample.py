"""Bilinear warping by a pixel-displacement flow field.

This replaces the legacy ``stnbdhw`` ``nn.BilinearSamplerBDHW`` CUDA module (used
via ``utils.warp_image``) with ``torch.nn.functional.grid_sample`` so it runs on
CPU, CUDA and Apple-Silicon MPS unchanged.

Conventions (must match the legacy sampler exactly)
---------------------------------------------------
* Flow is ``(dy, dx)`` **pixel displacement**: channel 0 = vertical (y), channel
  1 = horizontal (x). The sampled source position for output pixel ``(y, x)`` is
  ``(y + dy, x + dx)``. This is the canonical in-memory order produced by
  ``flow_io.uv_to_dydx`` and by the synthetic ``shift`` source.
* Out-of-bounds samples are **zero** (``padding_mode='zeros'``), matching the
  legacy ``image.warp(..., 'pad', 0)`` fallback and the CUDA kernel.
* ``align_corners=True``: normalized grid coordinate ``g`` maps to pixel index
  ``(g + 1) / 2 * (size - 1)``, so integer-pixel displacements are reproduced
  exactly (verified by the integer-shift test). The same ``warp`` serves the
  temporal warp, planar inference and the VR perspective / equirect maps, which
  use the identical displacement convention (with a large sentinel that simply
  lands out of bounds and samples zero).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Cache of base coordinate grids keyed by (H, W, device, dtype).
_BASE_CACHE: dict[tuple, torch.Tensor] = {}


def _base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a ``(1, H, W, 2)`` grid of integer pixel coords as ``(x, y)``."""
    key = (h, w, str(device), dtype)
    grid = _BASE_CACHE.get(key)
    if grid is None:
        ys = torch.arange(h, device=device, dtype=dtype)
        xs = torch.arange(w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W) each
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2) order (x,y)
        _BASE_CACHE[key] = grid
    return grid


def warp(img: torch.Tensor, flow_dydx: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """Warp ``img`` by a ``(dy, dx)`` pixel-displacement flow.

    Args:
        img: ``(N, C, H, W)`` (or ``(C, H, W)``) image to sample from.
        flow_dydx: ``(N, 2, H, W)`` (or ``(2, H, W)``) flow, channel 0 = dy,
            channel 1 = dx, in pixel units.
        align_corners: keep ``True`` for faithfulness (see module docstring).

    Returns:
        Warped image with the same shape as ``img``.
    """
    squeeze = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    if flow_dydx.dim() == 3:
        flow_dydx = flow_dydx.unsqueeze(0)
    if img.dim() != 4 or flow_dydx.dim() != 4 or flow_dydx.shape[1] != 2:
        raise ValueError(
            f"warp expects img (N,C,H,W) and flow (N,2,H,W); got {tuple(img.shape)} "
            f"and {tuple(flow_dydx.shape)}"
        )

    n, _, h, w = img.shape
    if flow_dydx.shape[0] != n or flow_dydx.shape[2:] != (h, w):
        raise ValueError(
            f"flow shape {tuple(flow_dydx.shape)} incompatible with img {tuple(img.shape)}"
        )

    flow_dydx = flow_dydx.to(device=img.device, dtype=img.dtype)
    base = _base_grid(h, w, img.device, img.dtype)  # (1,H,W,2) as (x,y)

    dy = flow_dydx[:, 0]  # (N,H,W)
    dx = flow_dydx[:, 1]  # (N,H,W)
    src_x = base[..., 0] + dx  # (N,H,W) via broadcast over batch
    src_y = base[..., 1] + dy

    if align_corners:
        denom_x = max(w - 1, 1)
        denom_y = max(h - 1, 1)
        norm_x = 2.0 * src_x / denom_x - 1.0
        norm_y = 2.0 * src_y / denom_y - 1.0
    else:
        norm_x = (2.0 * src_x + 1.0) / w - 1.0
        norm_y = (2.0 * src_y + 1.0) / h - 1.0

    grid = torch.stack([norm_x, norm_y], dim=-1)  # (N,H,W,2) order (x,y)
    out = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )
    return out.squeeze(0) if squeeze else out


def warp_masked(
    img: torch.Tensor, flow_dydx: torch.Tensor, cert: torch.Tensor
) -> torch.Tensor:
    """Warp then multiply by a certainty mask, broadcast over channels.

    Convenience for the inference/training path which always uses the warped
    previous frame masked by the occlusion certainty.
    """
    warped = warp(img, flow_dydx)
    if cert.dim() == warped.dim() - 1:
        cert = cert.unsqueeze(-3)
    return warped * cert.to(device=warped.device, dtype=warped.dtype)
