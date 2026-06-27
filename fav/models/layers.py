"""Building blocks for the generator, ported from ``models_video.lua``.

Covers ``ShaveImage``, the (non-)residual conv blocks, and the ``Tanh * const``
output head. ``TotalVariation`` from the legacy tail is intentionally *not* a
layer here: it was a forward-identity module that only injected a gradient, so
it is re-expressed as a differentiable penalty in ``fav.losses.temporal`` and
applied to the generator output during training (see plan §2a). At inference it
was a no-op, so omitting it leaves inference output unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fav.models.instance_norm import make_norm


class ShaveImage(nn.Module):
    """Center-crop ``size`` pixels from all four borders.

    Used on the identity branch of a residual block when the block's convolutions
    are unpadded (``reflect-start`` / ``none``), so the two branches match in
    size before the additive skip. Autograd handles the backward (the legacy
    module zero-padded the gradient, which slicing reproduces).
    """

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.size
        if s == 0:
            return x
        return x[:, :, s:-s, s:-s]


class MulConstant(nn.Module):
    """Multiply by a fixed scalar (the ``tanh_constant`` output scaling)."""

    def __init__(self, constant: float):
        super().__init__()
        self.constant = float(constant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.constant


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation channel attention (Phase-2 generator primitive).

    Channel-wise gating: global-average-pool -> 1x1 reduce -> ReLU -> 1x1 expand
    -> sigmoid -> scale. Preserves spatial dims and channel count, so it slots
    into the arch string via the ``E`` token without changing shapes.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


def _pad_module(padding_type: str, p: int) -> nn.Module | None:
    if p == 0:
        return None
    if padding_type == "reflect":
        return nn.ReflectionPad2d(p)
    if padding_type == "replicate":
        return nn.ReplicationPad2d(p)
    return None


def build_conv_block(dim: int, padding_type: str, use_instance_norm: bool,
                     norm: str | None = None) -> nn.Sequential:
    """Two 3x3 convs with norm+ReLU between, matching ``build_conv_block``.

    For ``reflect`` / ``replicate`` a padding layer precedes each conv (conv pad
    0). For ``zero`` the convs use pad 1. For ``reflect-start`` / ``none`` the
    convs are unpadded (the block shrinks the map by 2px per conv), which is why
    the residual identity branch is shaved by 2.
    """
    layers: list[nn.Module] = []
    conv_pad = 1 if padding_type == "zero" else 0

    pad = _pad_module(padding_type, 1)
    if pad is not None:
        layers.append(pad)
    layers.append(nn.Conv2d(dim, dim, 3, stride=1, padding=conv_pad))
    layers.append(make_norm(dim, use_instance_norm, norm))
    layers.append(nn.ReLU(inplace=True))

    pad = _pad_module(padding_type, 1)
    if pad is not None:
        layers.append(pad)
    layers.append(nn.Conv2d(dim, dim, 3, stride=1, padding=conv_pad))
    layers.append(make_norm(dim, use_instance_norm, norm))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    """Residual block: ``conv_block(x) + shave_or_identity(x)``.

    Mirrors ``build_res_block``: no activation after the additive skip. The
    identity branch is shaved by 2px under ``reflect-start`` / ``none`` so it
    matches the unpadded conv block's output size.
    """

    def __init__(self, dim: int, padding_type: str, use_instance_norm: bool,
                 norm: str | None = None):
        super().__init__()
        self.conv_block = build_conv_block(dim, padding_type, use_instance_norm, norm)
        if padding_type in ("none", "reflect-start"):
            self.shortcut: nn.Module = ShaveImage(2)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x) + self.shortcut(x)
