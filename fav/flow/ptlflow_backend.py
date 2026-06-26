"""Optional ptlflow backend exposing SEA-RAFT, GMA, FlowFormer, etc.

Lazily imports ``ptlflow`` so it is a no-cost optional dependency. Same
``FlowEstimator`` interface; pick the model via ``flow.model`` in config.
"""

from __future__ import annotations

import torch

from fav.flow.estimator import FlowEstimator


class PtlflowEstimator(FlowEstimator):
    def __init__(self, model: str = "sea_raft", ckpt: str = "things", device=None):
        try:
            import ptlflow  # type: ignore
            from ptlflow.utils.io_adapter import IOAdapter  # noqa: F401
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError("the ptlflow backend requires ptlflow (pip install ptlflow)") from e

        from fav.device import select_device

        self.device = torch.device(device) if device is not None else select_device()
        self._ptlflow = ptlflow
        self.net = ptlflow.get_model(model, ckpt).eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def estimate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        images = torch.stack([img1, img2], dim=1).to(self.device)  # (N,2,3,H,W)
        out = self.net({"images": images})
        return out["flows"][:, 0].to(img1.device)  # (N,2,H,W)
