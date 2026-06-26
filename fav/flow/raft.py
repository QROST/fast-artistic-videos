"""RAFT optical-flow backend (torchvision), the Phase-1 default.

Lazily imports torchvision so the package works without it installed. RAFT
returns flow as ``(N, 2, H, W)`` in ``(u, v)`` order — already the native
convention used throughout this package. On MPS, RAFT's iterative/correlation
ops are mostly supported; ``PYTORCH_ENABLE_MPS_FALLBACK`` (set in
``fav.device``) covers any gap, and flow is a precompute step (not the training
inner loop), so a CPU fallback is acceptable for correctness.
"""

from __future__ import annotations

import torch

from fav.flow.estimator import FlowEstimator


class RaftFlowEstimator(FlowEstimator):
    def __init__(self, model: str = "raft_large", device: str | torch.device | None = None):
        try:
            from torchvision.models.optical_flow import (  # type: ignore
                raft_large,
                raft_small,
                Raft_Large_Weights,
                Raft_Small_Weights,
            )
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "the RAFT backend requires torchvision (pip install torchvision)"
            ) from e

        from fav.device import select_device

        self.device = torch.device(device) if device is not None else select_device()
        if model == "raft_small":
            self._weights = Raft_Small_Weights.DEFAULT
            net = raft_small(weights=self._weights)
        else:
            self._weights = Raft_Large_Weights.DEFAULT
            net = raft_large(weights=self._weights)
        self.net = net.eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def estimate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # torchvision RAFT expects images normalized to roughly [-1, 1].
        img1 = (img1 * 2 - 1).to(self.device)
        img2 = (img2 * 2 - 1).to(self.device)
        flows = self.net(img1, img2)  # list of refinements; last is best
        return flows[-1].to(img1.device)
