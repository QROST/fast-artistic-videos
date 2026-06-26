"""Image-transformation network, a faithful port of ``models_video.lua``.

Parses the comma-separated architecture string into an ``nn.Sequential`` core,
then handles the ``reflect-start`` padding bookkeeping (the residual blocks use
unpadded convolutions and shave their skip branch, shrinking the map; a single
reflection pad at the input restores the output to the input size).

Layer codes (identical to the Lua parser):
    cXsY-Z  conv: kernel X, stride Y, Z filters (padding (X-1)/2)
    fXsY-Z  full (transposed) conv: kernel X, stride Y, Z filters
    dX      downsample: 3x3 stride-2 conv, X filters
    uX      learned upsample: 3x3 stride-1/2 transposed conv, X filters
    UX      nearest-neighbor upsample, factor X (avoids checkerboard)
    RX      residual block (two 3x3 convs), X filters
    CX      non-residual conv block, X filters

Every internal conv is followed by Norm then ReLU; the final layer gets neither.
The head is ``Tanh`` then ``* tanh_constant``. Input is 7 channels
(3 current frame + 3 warped-masked previous output + 1 certainty mask).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fav.models.instance_norm import make_norm
from fav.models.layers import MulConstant, ResidualBlock, build_conv_block

IN_CHANNELS = 7


def _build_core(
    arch: str, use_instance_norm: bool, padding_type: str, tanh_constant: float
) -> nn.Sequential:
    tokens = arch.split(",")
    layers: list[nn.Module] = []
    prev_dim = IN_CHANNELS

    for i, tok in enumerate(tokens):
        kind = tok[0]
        needs_norm = True
        needs_relu = True

        if kind == "c":
            f = int(tok[1])
            s = int(tok[3])
            next_dim = int(tok[5:])
            p = (f - 1) // 2
            if padding_type == "reflect":
                layers.append(nn.ReflectionPad2d(p))
                p = 0
            elif padding_type == "replicate":
                layers.append(nn.ReplicationPad2d(p))
                p = 0
            layers.append(nn.Conv2d(prev_dim, next_dim, f, stride=s, padding=p))
        elif kind == "f":
            f = int(tok[1])
            s = int(tok[3])
            next_dim = int(tok[5:])
            p = (f - 1) // 2
            layers.append(
                nn.ConvTranspose2d(prev_dim, next_dim, f, stride=s, padding=p, output_padding=s - 1)
            )
        elif kind == "d":
            next_dim = int(tok[1:])
            layers.append(nn.Conv2d(prev_dim, next_dim, 3, stride=2, padding=1))
        elif kind == "U":
            next_dim = prev_dim
            scale = int(tok[1:])
            layers.append(nn.Upsample(scale_factor=scale, mode="nearest"))
        elif kind == "u":
            next_dim = int(tok[1:])
            layers.append(
                nn.ConvTranspose2d(prev_dim, next_dim, 3, stride=2, padding=1, output_padding=1)
            )
        elif kind == "C":
            next_dim = int(tok[1:])
            layers.append(build_conv_block(next_dim, padding_type, use_instance_norm))
            needs_norm = False
            needs_relu = True
        elif kind == "R":
            next_dim = int(tok[1:])
            layers.append(ResidualBlock(next_dim, padding_type, use_instance_norm))
            needs_norm = False
            needs_relu = False
        else:
            raise ValueError(f"unknown architecture token: {tok!r}")

        if i == len(tokens) - 1:
            needs_norm = False
            needs_relu = False
        if needs_norm:
            layers.append(make_norm(next_dim, use_instance_norm))
        if needs_relu:
            layers.append(nn.ReLU(inplace=True))

        prev_dim = next_dim

    # Output head. TotalVariation (a forward-identity gradient injector in the
    # legacy tail) is handled as a loss penalty instead; see fav.losses.temporal.
    layers.append(nn.Tanh())
    layers.append(MulConstant(tanh_constant))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """The 7-channel image-transformation network."""

    def __init__(
        self,
        arch: str = "c9s1-32,d64,d128,R128,R128,R128,R128,R128,U2,c3s1-64,U2,c9s1-3",
        use_instance_norm: bool = True,
        padding_type: str = "reflect-start",
        tanh_constant: float = 150.0,
    ):
        super().__init__()
        self.padding_type = padding_type
        self.core = _build_core(arch, use_instance_norm, padding_type, tanh_constant)
        # For reflect-start, measure the constant spatial shrink once and prepend
        # a reflection pad so output size == input size (the "lazy size-fix" from
        # train_video.lua, computed deterministically at build time).
        self.pad_h = 0
        self.pad_w = 0
        if padding_type == "reflect-start":
            self.pad_h, self.pad_w = self._measure_pad()

    @torch.no_grad()
    def _measure_pad(self, ref: int = 128) -> tuple[int, int]:
        was_training = self.core.training
        self.core.eval()
        dummy = torch.zeros(1, IN_CHANNELS, ref, ref)
        out = self.core(dummy)
        sh = ref - out.shape[-2]
        sw = ref - out.shape[-1]
        if was_training:
            self.core.train()
        if sh < 0 or sw < 0 or sh % 2 or sw % 2:
            raise ValueError(
                f"unexpected reflect-start shrink (h={sh}, w={sw}); arch may be incompatible"
            )
        return sh // 2, sw // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != IN_CHANNELS:
            raise ValueError(f"generator expects {IN_CHANNELS} input channels, got {x.shape[1]}")
        if self.padding_type != "reflect-start":
            return self.core(x)

        out = self.core(self._reflect_pad(x))
        # One-time lazy correction for input sizes where the floor in the
        # downsampling layers makes the shrink differ from the reference (mirrors
        # the train_video.lua hack). Keeps output size == input size.
        if out.shape[-2:] != x.shape[-2:]:
            self.pad_h += (x.shape[-2] - out.shape[-2]) // 2
            self.pad_w += (x.shape[-1] - out.shape[-1]) // 2
            out = self.core(self._reflect_pad(x))
        return out

    def _reflect_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_h == 0 and self.pad_w == 0:
            return x
        return F.pad(x, (self.pad_w, self.pad_w, self.pad_h, self.pad_h), mode="reflect")


def build_model(config=None, **overrides) -> Generator:
    """Build a :class:`Generator` from a ModelConfig (or keyword overrides)."""
    kwargs = {}
    if config is not None:
        kwargs.update(
            arch=config.arch,
            use_instance_norm=bool(config.use_instance_norm),
            padding_type=config.padding_type,
            tanh_constant=config.tanh_constant,
        )
    kwargs.update(overrides)
    return Generator(**kwargs)
