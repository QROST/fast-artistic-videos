"""Gram matrix, a faithful port of ``GramMatrix.lua``.

``G = X X^T`` with ``X`` reshaped to ``(N, C, H*W)``, **normalized by ``C*H*W``**
(the legacy default ``normalize = true``). Implemented with autograd so the
manual ``updateGradInput`` is unnecessary.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gram_matrix(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Compute Gram matrices for a feature batch.

    Args:
        x: ``(N, C, H, W)`` feature maps.
        normalize: divide by ``C*H*W`` (default, matching the legacy module).

    Returns:
        ``(N, C, C)`` Gram matrices.
    """
    if x.dim() != 4:
        raise ValueError(f"expected (N,C,H,W), got {tuple(x.shape)}")
    n, c, h, w = x.shape
    feats = x.view(n, c, h * w)
    gram = torch.bmm(feats, feats.transpose(1, 2))
    if normalize:
        gram = gram / (c * h * w)
    return gram


class GramMatrix(nn.Module):
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gram_matrix(x, self.normalize)
