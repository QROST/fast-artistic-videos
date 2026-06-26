"""Temporal pixel loss and total-variation penalty.

``temporal_pixel_loss`` is the consistency term: a pixel loss between the masked
current output and the masked warped previous output (consistency is only
enforced in reliable regions). ``tv_penalty`` reproduces the legacy
``TotalVariation`` gradient as a differentiable penalty (see plan §2a):
the legacy forward-difference finite-difference gradient is exactly the gradient
of ``0.5 * (sum(x_diff^2) + sum(y_diff^2))`` over the ``[:-1,:-1]`` interior.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_pixel_loss(
    out_masked: torch.Tensor, warped_masked: torch.Tensor, loss_type: str = "L2"
) -> torch.Tensor:
    """Pixel loss between masked current output and masked warped previous output.

    Uses the mean reduction (``nn.MSECriterion`` semantics). ``loss_type`` is one
    of ``L2`` (MSE), ``L1``, ``SmoothL1``.
    """
    if loss_type == "L2":
        return F.mse_loss(out_masked, warped_masked)
    if loss_type == "L1":
        return F.l1_loss(out_masked, warped_masked)
    if loss_type == "SmoothL1":
        return F.smooth_l1_loss(out_masked, warped_masked)
    raise ValueError(f"unknown pixel_loss_type {loss_type!r}")


def tv_penalty(x: torch.Tensor, strength: float) -> torch.Tensor:
    """Anisotropic squared total-variation penalty on forward differences.

    The legacy ``TotalVariation`` module injected a gradient equal to that of
    ``0.5 * strength * (sum(x_diff^2) + sum(y_diff^2))`` computed over the
    interior ``[:-1, :-1]`` of the (N,3,H,W) image, so this reproduces it exactly
    under autograd. Summed (not meaned), matching the per-element legacy gradient.
    """
    if strength == 0:
        return x.new_zeros(())
    x_diff = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
    y_diff = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
    tv = 0.5 * (x_diff.pow(2).sum() + y_diff.pow(2).sum())
    return strength * tv
