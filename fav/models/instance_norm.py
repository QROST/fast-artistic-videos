"""Normalization factory, a faithful port of ``InstanceNormalization.lua``.

The legacy module implemented instance norm via ``SpatialBatchNormalization``
over ``N*C`` features, always in training mode with no running statistics, eps
``1e-5`` and a learnable affine ``(C,)`` weight/bias. ``nn.InstanceNorm2d`` with
``track_running_stats=False`` reproduces this exactly. Batch norm is selected
when ``use_instance_norm`` is false (the original's ``-use_instance_norm 0``).
"""

from __future__ import annotations

import torch.nn as nn

INSTANCE_NORM_EPS = 1e-5


def make_norm(num_features: int, use_instance_norm: bool = True) -> nn.Module:
    if use_instance_norm:
        return nn.InstanceNorm2d(
            num_features,
            eps=INSTANCE_NORM_EPS,
            affine=True,
            track_running_stats=False,
        )
    return nn.BatchNorm2d(num_features)
