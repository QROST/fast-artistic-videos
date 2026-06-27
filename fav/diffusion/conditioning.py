"""Reuse-bridge conditioning for the Phase-3 diffusion path (step 3a).

Reuses the Phase-1/2 machinery — the ``(dy,dx)`` warp, optical flow, and
occlusion certainty — to produce ControlNet-style conditioning tensors that a
diffusion + per-style-LoRA stylizer can consume. The same per-frame recipe that
gives the feed-forward net its temporal consistency ("warp the previous output,
mask by occlusion") becomes the diffusion conditioning, so Phase-3 inherits
Phase-1's consistency for free.

Everything here is **device-agnostic** (operates on the input tensors' device)
and uses only core torch ops, so it runs unchanged on CPU, CUDA, and Apple-Silicon
**MPS** — no ``diffusers`` / numpy round-trips in the hot path. Inputs/outputs are
RGB ``[0,1]`` tensors shaped ``(N,3,H,W)`` (conditions are ``(N,C,H,W)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fav.occlusion.filters import min_filter
from fav.warp.grid_sample import warp


@dataclass
class ConditioningBundle:
    """ControlNet-style conditioning derived from Phase-1/2 signals.

    All tensors are ``(N,*,H,W)`` on the same device/dtype as the inputs.
    """

    content: torch.Tensor          # (N,3,H,W) current content frame, RGB [0,1]
    warped_prev: torch.Tensor      # (N,3,H,W) previous stylized output, warped to now
    warped_prev_masked: torch.Tensor  # warped_prev * cert (temporal prior)
    cert: torch.Tensor             # (N,1,H,W) occlusion certainty in [0,1]
    flow_image: torch.Tensor       # (N,3,H,W) flow color-wheel viz, RGB [0,1]
    structure: torch.Tensor        # (N,1,H,W) Sobel edge map of the content, [0,1]

    def to(self, device) -> "ConditioningBundle":
        return ConditioningBundle(
            **{k: v.to(device) for k, v in self.__dict__.items()}
        )

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorized HSV->RGB (all in ``[0,1]``), device-agnostic. Inputs ``(N,1,H,W)``."""
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).long()
    # Stack the 6 candidate (r,g,b) per sextant and select by i.
    r_opts = torch.cat([v, q, p, p, t, v], dim=1)
    g_opts = torch.cat([t, v, v, q, p, p], dim=1)
    b_opts = torch.cat([p, p, t, v, v, q], dim=1)
    idx = i.expand(-1, 1, -1, -1)
    r = torch.gather(r_opts, 1, idx)
    g = torch.gather(g_opts, 1, idx)
    b = torch.gather(b_opts, 1, idx)
    return torch.cat([r, g, b], dim=1)


def flow_to_rgb(flow_dydx: torch.Tensor, max_mag: float | None = None) -> torch.Tensor:
    """Color-wheel visualization of a ``(N,2,H,W)`` ``(dy,dx)`` flow as RGB ``[0,1]``.

    Hue encodes direction, value encodes magnitude (normalized per-batch unless
    ``max_mag`` is given). A standard ControlNet motion-conditioning image.
    """
    if flow_dydx.dim() == 3:
        flow_dydx = flow_dydx.unsqueeze(0)
    dy = flow_dydx[:, 0:1]
    dx = flow_dydx[:, 1:2]
    mag = torch.sqrt(dx * dx + dy * dy)
    if max_mag is None:
        m = mag.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    else:
        m = torch.as_tensor(max_mag, device=flow_dydx.device, dtype=flow_dydx.dtype).clamp_min(1e-6)
    value = (mag / m).clamp(0, 1)
    angle = torch.atan2(dy, dx)  # [-pi, pi]
    hue = (angle / (2 * torch.pi)) + 0.5  # [0,1]
    sat = torch.ones_like(value)
    return _hsv_to_rgb(hue, sat, value)


def sobel_edges(rgb: torch.Tensor) -> torch.Tensor:
    """Sobel edge-magnitude map (grayscale, ``[0,1]``) — a structure condition.

    Device-agnostic; replicate padding so it works at any size.
    """
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    gray = (0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3])
    kx = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=rgb.device, dtype=rgb.dtype)
    ky = kx.t().contiguous()
    g = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(g, kx.view(1, 1, 3, 3))
    gy = F.conv2d(g, ky.view(1, 1, 3, 3))
    mag = torch.sqrt(gx * gx + gy * gy)
    # Normalize by the theoretical max for [0,1] input (a hard edge gives |g|~=4),
    # NOT a per-image max — so a flat image maps to exactly 0 instead of amplified
    # float noise. A sharp edge saturates to ~1.
    return (mag / 4.0).clamp(0, 1)


def build_conditioning(
    content_rgb: torch.Tensor,
    prev_stylized_rgb: torch.Tensor,
    flow_dydx: torch.Tensor,
    cert: torch.Tensor,
    occlusions_min_filter: int = 7,
) -> ConditioningBundle:
    """Build the conditioning bundle for a subsequent frame.

    Args:
        content_rgb: ``(N,3,H,W)`` current content frame, RGB ``[0,1]``.
        prev_stylized_rgb: ``(N,3,H,W)`` previous stylized output, RGB ``[0,1]``.
        flow_dydx: ``(N,2,H,W)`` backward flow (current->previous), ``(dy,dx)``.
        cert: ``(N,1,H,W)`` occlusion certainty in ``[0,1]``.
    """
    cert = min_filter(cert.to(content_rgb.device), occlusions_min_filter)
    warped = warp(prev_stylized_rgb, flow_dydx)
    warped_masked = warped * cert
    return ConditioningBundle(
        content=content_rgb,
        warped_prev=warped,
        warped_prev_masked=warped_masked,
        cert=cert,
        flow_image=flow_to_rgb(flow_dydx),
        structure=sobel_edges(content_rgb),
    )


def first_frame_conditioning(content_rgb: torch.Tensor) -> ConditioningBundle:
    """Conditioning for the first frame: no temporal prior (all-occluded)."""
    n, _, h, w = content_rgb.shape
    z3 = torch.zeros(n, 3, h, w, device=content_rgb.device, dtype=content_rgb.dtype)
    z1 = torch.zeros(n, 1, h, w, device=content_rgb.device, dtype=content_rgb.dtype)
    return ConditioningBundle(
        content=content_rgb,
        warped_prev=z3,
        warped_prev_masked=z3,
        cert=z1,
        flow_image=z3,
        structure=sobel_edges(content_rgb),
    )


# Named conditions available for stacking into a ControlNet input tensor.
_CONTROL_CHANNELS = {
    "content": 3, "warped_prev": 3, "warped_prev_masked": 3,
    "cert": 1, "flow_image": 3, "structure": 1,
}


def stack_controls(bundle: ConditioningBundle, which=("warped_prev_masked", "cert", "structure")):
    """Concatenate selected conditions into one ``(N, sum(C), H, W)`` control tensor."""
    parts = []
    for name in which:
        if name not in _CONTROL_CHANNELS:
            raise ValueError(f"unknown condition {name!r}; choices: {sorted(_CONTROL_CHANNELS)}")
        parts.append(getattr(bundle, name))
    return torch.cat(parts, dim=1)
