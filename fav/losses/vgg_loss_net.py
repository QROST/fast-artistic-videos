"""VGG-16 loss network (caffe layout) with index-addressable feature taps.

The original FAV loss network is the ``vgg16.t7`` ``nn.Sequential`` from
jcjohnson's fast-neural-style (caffe ``VGG_ILSVRC_16_layers``). Style appearance
depends on these *exact* features, so torchvision's RGB-trained VGG-16 is NOT a
drop-in; weights come from converting that ``.t7`` (see ``fav.conversion``) or
the equivalent caffe weights. The loss net consumes inputs already in VGG-caffe
space (BGR, ×255, mean-subtracted) — the same space the generator output and the
preprocessed content targets live in — so it applies no preprocessing itself.

Layer indices follow the legacy 1-indexed ``nn.Sequential`` convention used by
the config (``content_layers='16'``, ``style_layers='4,9,16,23'``):

    4 = relu1_2,  9 = relu2_2,  16 = relu3_3,  23 = relu4_3
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# Caffe VGG-16 feature stack as an ordered spec (1-indexed positions match the
# legacy nn.Sequential). Each entry builds one module.
#   ('conv', in, out) | ('relu',) | ('pool',)
_VGG16_SPEC = [
    ("conv", 3, 64), ("relu",), ("conv", 64, 64), ("relu",), ("pool",),       # 1-5
    ("conv", 64, 128), ("relu",), ("conv", 128, 128), ("relu",), ("pool",),   # 6-10
    ("conv", 128, 256), ("relu",), ("conv", 256, 256), ("relu",),             # 11-14
    ("conv", 256, 256), ("relu",), ("pool",),                                 # 15-17
    ("conv", 256, 512), ("relu",), ("conv", 512, 512), ("relu",),             # 18-21
    ("conv", 512, 512), ("relu",), ("pool",),                                 # 22-24
    ("conv", 512, 512), ("relu",), ("conv", 512, 512), ("relu",),             # 25-28
    ("conv", 512, 512), ("relu",), ("pool",),                                 # 29-31
]

# Human-readable names for the tap indices (for tests / debugging).
LAYER_NAMES = {
    4: "relu1_2", 9: "relu2_2", 16: "relu3_3", 23: "relu4_3", 30: "relu5_3",
}


def _make_module(entry):
    kind = entry[0]
    if kind == "conv":
        return nn.Conv2d(entry[1], entry[2], kernel_size=3, stride=1, padding=1)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "pool":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    raise ValueError(entry)


class VGG16Features(nn.Module):
    """VGG-16 feature extractor that returns activations at given 1-based taps."""

    def __init__(self, max_index: int):
        super().__init__()
        if not 1 <= max_index <= len(_VGG16_SPEC):
            raise ValueError(f"max_index must be in [1, {len(_VGG16_SPEC)}]")
        self.max_index = max_index
        self.layers = nn.ModuleList(_make_module(e) for e in _VGG16_SPEC[:max_index])
        # Loss network is fixed: no gradients to its weights, always eval.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor, taps: set[int]) -> dict[int, torch.Tensor]:
        """Return ``{lua_index: activation}`` for each requested tap.

        Stops after the largest requested tap. ``taps`` are 1-indexed positions.
        """
        if taps and max(taps) > self.max_index:
            raise ValueError(f"tap {max(taps)} exceeds built depth {self.max_index}")
        out: dict[int, torch.Tensor] = {}
        stop = max(taps) if taps else 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            idx = i + 1  # 1-indexed
            if idx in taps:
                out[idx] = x
            if idx >= stop:
                break
        return out

    def conv_modules(self):
        """Yield the conv layers in order (used by the .t7 converter)."""
        return [m for m in self.layers if isinstance(m, nn.Conv2d)]


def build_vgg16_loss_net(
    content_layers, style_layers, weights_path: str | Path | None = None
) -> VGG16Features:
    """Build a VGG-16 loss net deep enough for the requested taps.

    Args:
        content_layers, style_layers: iterables of 1-indexed tap positions.
        weights_path: optional ``.pt`` state_dict produced by the converter. If
            ``None`` the net keeps random init (valid for pipeline/smoke tests;
            real style fidelity needs the converted caffe weights).
    """
    max_index = max([*content_layers, *style_layers])
    net = VGG16Features(max_index=max_index)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = net.load_state_dict(state, strict=False)
        # Only the depth we built must be satisfied; deeper keys are ignored.
        relevant_missing = [k for k in missing if k.split(".")[1].isdigit()]
        if relevant_missing:
            raise RuntimeError(f"missing VGG weights for built layers: {relevant_missing}")
    return net
