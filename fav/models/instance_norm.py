"""Normalization factory, a faithful port of ``InstanceNormalization.lua``.

The legacy module implemented instance norm via ``SpatialBatchNormalization``
over ``N*C`` features, always in training mode with no running statistics, eps
``1e-5`` and a learnable affine ``(C,)`` weight/bias. ``nn.InstanceNorm2d`` with
``track_running_stats=False`` reproduces this exactly. Batch norm is selected
when ``use_instance_norm`` is false (the original's ``-use_instance_norm 0``).
"""

from __future__ import annotations

import math

import torch.nn as nn

INSTANCE_NORM_EPS = 1e-5


def _pick_groups(num_features: int) -> int:
    """Largest divisor of ``num_features`` that is <= 32 (for GroupNorm)."""
    for g in (32, 16, 8, 4, 2):
        if num_features % g == 0:
            return g
    return 1


def make_norm(num_features: int, use_instance_norm: bool = True, norm: str | None = None) -> nn.Module:
    """Build a normalization layer.

    ``norm`` (Phase-2) selects ``instance`` (faithful default), ``batch`` or
    ``group``. When ``norm`` is ``None`` it derives from the legacy
    ``use_instance_norm`` flag, so existing callers are unchanged.
    """
    if norm is None:
        norm = "instance" if use_instance_norm else "batch"
    if norm == "instance":
        return nn.InstanceNorm2d(
            num_features, eps=INSTANCE_NORM_EPS, affine=True, track_running_stats=False
        )
    if norm == "batch":
        return nn.BatchNorm2d(num_features)
    if norm == "group":
        return nn.GroupNorm(_pick_groups(num_features), num_features)
    raise ValueError(f"unknown norm {norm!r}; expected instance|batch|group")
