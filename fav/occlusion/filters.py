"""Morphological min filter and median filter, porting ``utils.lua`` helpers.

* ``min_filter`` (erosion) shrinks the reliable (high) region so artifacts right
  at motion boundaries are removed. The legacy implementation computes it as
  ``1 - maxpool(1 - x)`` with a sliding window; default width 7.
* ``median_filter`` denoises the stylized output; default width 3. The median is
  computed on CPU (mirroring the legacy "median not defined for CudaTensors"
  cast) and the result returned to the input device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_n1hw(x: torch.Tensor):
    """Normalize input to (N, 1, H, W), remembering how to restore it."""
    if x.dim() == 2:  # (H,W)
        return x.view(1, 1, *x.shape), "hw"
    if x.dim() == 3:  # (1,H,W) or (C,H,W) -> treat leading as batch*channel=1
        if x.shape[0] == 1:
            return x.view(1, 1, *x.shape[1:]), "1hw"
        return x.unsqueeze(1), "n_hw"  # (N,H,W) -> (N,1,H,W)
    if x.dim() == 4:
        return x, "nchw"
    raise ValueError(f"unsupported shape {tuple(x.shape)}")


def _restore(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hw":
        return x.view(x.shape[-2], x.shape[-1])
    if mode == "1hw":
        return x.view(1, x.shape[-2], x.shape[-1])
    if mode == "n_hw":
        return x.squeeze(1)
    return x


def min_filter(x: torch.Tensor, radius: int = 7) -> torch.Tensor:
    """Erosion: per window minimum, width ``radius`` (kept on the input device).

    Implemented as ``1 - maxpool(1 - x)`` over a ``radius x radius`` window with
    stride 1 and ``radius // 2`` padding, matching ``utils.min_filter``. Assumes
    ``x`` is in ``[0, 1]`` (the certainty/reliability convention).
    """
    if radius <= 1:
        return x
    t, mode = _to_n1hw(x)
    pad = radius // 2
    eroded = 1.0 - F.max_pool2d(1.0 - t, kernel_size=radius, stride=1, padding=pad)
    # max_pool2d with even kernel can change size by 1; crop/realign to input.
    eroded = eroded[..., : t.shape[-2], : t.shape[-1]]
    return _restore(eroded, mode)


def median_filter(x: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Sliding ``radius x radius`` median, computed on CPU then moved back."""
    if radius <= 1:
        return x
    t, mode = _to_n1hw(x)
    device = t.device
    t_cpu = t.detach().to("cpu", torch.float32)
    pad = radius // 2
    padded = F.pad(t_cpu, (pad, pad, pad, pad), mode="replicate")
    # Unfold into windows and take the median over each window.
    patches = padded.unfold(2, radius, 1).unfold(3, radius, 1)  # N,C,H,W,r,r
    n, c, h, w, _, _ = patches.shape
    med = patches.contiguous().view(n, c, h, w, radius * radius).median(dim=-1).values
    return _restore(med.to(device), mode)
