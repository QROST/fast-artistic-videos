"""Optical-flow estimator interface and dataset flow/occlusion helpers.

The estimator is deliberately swappable (RAFT by default, SEA-RAFT/others via
ptlflow). Flow is returned in native ``(u, v)`` order — channel 0 = horizontal
(x), 1 = vertical (y) — so it feeds ``flow_io`` and the ``(dy, dx)`` warp swap
directly. Inputs are RGB ``[0, 1]`` tensors shaped ``(N, 3, H, W)``.

Filenames follow the legacy patterns so existing assets and the stylize
pipeline keep working: ``backward_{start}_{target}.flo`` and
``reliable_{start}_{target}.pgm``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F

from fav.occlusion.consistency import compute_reliability
from fav.warp.flow_io import write_flo, write_pgm


class FlowEstimator(ABC):
    """Estimate optical flow between two frames."""

    @abstractmethod
    def estimate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Return flow ``(N, 2, H, W)`` (u, v) carrying img1's pixels to img2."""

    def estimate_bidirectional(self, img1: torch.Tensor, img2: torch.Tensor):
        """Return ``(forward, backward)`` flow as ``img1->img2`` and ``img2->img1``."""
        return self.estimate(img1, img2), self.estimate(img2, img1)


class DummyFlowEstimator(FlowEstimator):
    """Zero-flow estimator — for tests and pipelines with analytic flow."""

    def estimate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        n, _, h, w = img1.shape
        return torch.zeros(n, 2, h, w, device=img1.device, dtype=img1.dtype)


def rescale_flow(flow: torch.Tensor, to_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a ``(N,2,H,W)`` flow field to ``to_hw`` and rescale its magnitude.

    The u (x) component scales by ``W_new/W`` and v (y) by ``H_new/H`` so the
    flow still measures pixel displacement at the new resolution.
    """
    n, c, h, w = flow.shape
    new_h, new_w = to_hw
    if (h, w) == (new_h, new_w):
        return flow
    resized = F.interpolate(flow, size=(new_h, new_w), mode="bilinear", align_corners=False)
    resized = resized.clone()
    resized[:, 0] *= new_w / w  # u (x)
    resized[:, 1] *= new_h / h  # v (y)
    return resized


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def estimate_at_friendly_size(
    estimator: FlowEstimator, img1: torch.Tensor, img2: torch.Tensor, multiple: int = 8
) -> torch.Tensor:
    """Run ``estimator`` at a size divisible by ``multiple``, return flow at the
    original resolution (magnitude rescaled). Many flow nets require /8 inputs.
    """
    _, _, h, w = img1.shape
    th, tw = _round_to_multiple(h, multiple), _round_to_multiple(w, multiple)
    if (th, tw) != (h, w):
        img1r = F.interpolate(img1, size=(th, tw), mode="bilinear", align_corners=False)
        img2r = F.interpolate(img2, size=(th, tw), mode="bilinear", align_corners=False)
    else:
        img1r, img2r = img1, img2
    flow = estimator.estimate(img1r, img2r)
    return rescale_flow(flow, (h, w))


def flow_filename(start: int, target: int) -> str:
    return f"backward_{start}_{target}.flo"


def occlusion_filename(start: int, target: int) -> str:
    return f"reliable_{start}_{target}.pgm"


def compute_pair(
    estimator: FlowEstimator,
    frame_prev: torch.Tensor,
    frame_cur: torch.Tensor,
    use_structure: bool = True,
):
    """Compute the backward flow and reliability for an adjacent frame pair.

    To stylize ``frame_cur`` we warp the previous stylized frame by the backward
    flow ``cur -> prev``; the reliability comes from a forward/backward
    consistency check.

    Args:
        frame_prev, frame_cur: RGB ``[0,1]`` tensors ``(1,3,H,W)``.
    Returns:
        ``(backward_flow_uv (2,H,W), reliable (H,W) in [0,255])``.
    """
    backward = estimate_at_friendly_size(estimator, frame_cur, frame_prev)[0]  # cur->prev
    forward = estimate_at_friendly_size(estimator, frame_prev, frame_cur)[0]  # prev->cur
    content = frame_cur[0] if use_structure else None
    reliable = compute_reliability(backward, forward, content_image=content)
    return backward, reliable


def write_pair(out_dir: str | Path, start: int, target: int, flow_uv: torch.Tensor, reliable: torch.Tensor):
    """Write ``backward_{start}_{target}.flo`` and ``reliable_{start}_{target}.pgm``."""
    out_dir = Path(out_dir)
    write_flo(out_dir / flow_filename(start, target), flow_uv)
    write_pgm(out_dir / occlusion_filename(start, target), reliable)
