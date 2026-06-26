"""Content and style aggregation helpers (ports of ContentLoss/StyleLoss.lua).

Content loss is an MSE on raw activations. Style loss compares an aggregation of
the activations: a Gram matrix (``agg_type='gram'``) or the spatial mean
(``agg_type='mean'``). Kept as small pure functions so ``PerceptualCriterion``
stays the single orchestrator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fav.losses.gram import gram_matrix


def style_aggregate(feat: torch.Tensor, agg_type: str = "gram") -> torch.Tensor:
    """Aggregate ``(N,C,H,W)`` activations into the style statistic to match."""
    if agg_type == "gram":
        return gram_matrix(feat, normalize=True)
    if agg_type == "mean":
        return feat.mean(dim=(2, 3))  # (N, C)
    raise ValueError(f"unknown style agg_type {agg_type!r}")


def _match_batch(target: torch.Tensor, n: int) -> torch.Tensor:
    """Broadcast a style target with batch 1 up to batch ``n``."""
    if target.shape[0] == n:
        return target
    if target.shape[0] == 1:
        return target.expand(n, *target.shape[1:])
    raise ValueError(f"style target batch {target.shape[0]} incompatible with {n}")


def content_distance(out_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(out_feat, target_feat)


def style_distance(out_agg: torch.Tensor, target_agg: torch.Tensor) -> torch.Tensor:
    target_agg = _match_batch(target_agg, out_agg.shape[0])
    return F.mse_loss(out_agg, target_agg)
