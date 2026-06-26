"""Perceptual criterion: content + style losses through the VGG loss network.

Faithful port of ``PerceptualCriterion.lua`` but using a forward feature
extractor + autograd instead of the legacy inject-modules + manual
``updateGradInput`` machinery (the result is identical; see plan §2).

Usage::

    crit = PerceptualCriterion(loss_net, content_layers, content_weights,
                               style_layers, style_weights, agg_type='gram')
    crit.set_style_target(style_img_vggspace)   # once
    loss = crit(output, content_target)         # per batch
    # crit.total_content_loss / crit.total_style_loss available for logging
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fav.losses.content_style import content_distance, style_aggregate, style_distance


def _broadcast_weights(layers, weights):
    """Pair each layer with a weight, broadcasting a single weight to all."""
    layers = list(layers)
    weights = list(weights)
    if len(weights) == 1 and len(layers) > 1:
        weights = weights * len(layers)
    if len(weights) != len(layers):
        raise ValueError(f"got {len(weights)} weights for {len(layers)} layers")
    return list(zip(layers, weights))


class PerceptualCriterion(nn.Module):
    def __init__(
        self,
        loss_net: nn.Module,
        content_layers,
        content_weights,
        style_layers,
        style_weights,
        agg_type: str = "gram",
    ):
        super().__init__()
        self.loss_net = loss_net
        self.content = _broadcast_weights(content_layers, content_weights)
        self.style = _broadcast_weights(style_layers, style_weights)
        self.agg_type = agg_type
        self._content_taps = {l for l, _ in self.content}
        self._style_taps = {l for l, _ in self.style}
        self._all_taps = self._content_taps | self._style_taps
        self._style_targets: dict[int, torch.Tensor] = {}
        # Per-call diagnostics (for loss history logging).
        self.content_losses: list[float] = []
        self.style_losses: list[float] = []
        self.total_content_loss = 0.0
        self.total_style_loss = 0.0

    @torch.no_grad()
    def set_style_target(self, style_img: torch.Tensor) -> None:
        """Capture style-layer Gram (or mean) targets from a VGG-space image."""
        feats = self.loss_net(style_img, self._style_taps)
        self._style_targets = {
            layer: style_aggregate(feats[layer], self.agg_type) for layer in self._style_taps
        }

    def forward(self, output: torch.Tensor, content_target: torch.Tensor) -> torch.Tensor:
        if not self._style_targets and self.style:
            raise RuntimeError("call set_style_target() before computing the loss")

        # Content-layer targets from the (frozen) content image.
        with torch.no_grad():
            target_feats = self.loss_net(content_target, self._content_taps)

        out_feats = self.loss_net(output, self._all_taps)

        total_content = output.new_zeros(())
        self.content_losses = []
        for layer, weight in self.content:
            d = content_distance(out_feats[layer], target_feats[layer])
            self.content_losses.append(float(d.detach()))
            total_content = total_content + weight * d

        total_style = output.new_zeros(())
        self.style_losses = []
        for layer, weight in self.style:
            agg = style_aggregate(out_feats[layer], self.agg_type)
            d = style_distance(agg, self._style_targets[layer])
            self.style_losses.append(float(d.detach()))
            total_style = total_style + weight * d

        self.total_content_loss = float(total_content.detach())
        self.total_style_loss = float(total_style.detach())
        return total_content + total_style
