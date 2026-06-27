"""Phase-3 diffusion video pipeline: interface, temporal loop, and backends.

The video loop (``stylize_video_diffusion``) mirrors the feed-forward
``infer.stylize_sequence`` but for the diffusion path: it builds the Phase-3a
conditioning each frame (reusing the validated flow/occlusion/warp) and hands it
to a pluggable :class:`DiffusionStylizer`. This keeps the temporal-consistency
recipe identical to Phase 1, independent of the stylizer backend.

``DummyDiffusionStylizer`` makes the whole loop runnable + testable here without
``diffusers``. ``SDXLControlNetStylizer`` is the real backend to be fleshed out
on the M5 Max (lazy ``diffusers`` import); the seam between the loop and the
backend is fully defined so only the backend changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from fav.diffusion.conditioning import ConditioningBundle, build_conditioning


class DiffusionStylizer(ABC):
    """Backend that stylizes one frame, given Phase-3a conditioning."""

    @abstractmethod
    def stylize_first(self, content_rgb: torch.Tensor, style_ref=None) -> torch.Tensor:
        """Stylize the first frame (no temporal prior). Returns RGB ``[0,1]``."""

    @abstractmethod
    def stylize_next(self, content_rgb: torch.Tensor, conditioning: ConditioningBundle,
                     style_ref=None) -> torch.Tensor:
        """Stylize a subsequent frame using the conditioning bundle."""


class DummyDiffusionStylizer(DiffusionStylizer):
    """Deterministic, dependency-free backend for wiring/tests.

    Demonstrates the temporal mechanism: keep the warped previous output where
    the flow is reliable, fall back to the content where occluded. Real backends
    replace this with a diffusion denoise conditioned on the same signals.
    """

    def stylize_first(self, content_rgb, style_ref=None):
        return content_rgb.clamp(0, 1)

    def stylize_next(self, content_rgb, conditioning, style_ref=None):
        cert = conditioning.cert
        out = conditioning.warped_prev_masked + content_rgb * (1.0 - cert)
        return out.clamp(0, 1)


def stylize_video_diffusion(
    stylizer: DiffusionStylizer,
    frames_rgb: list[torch.Tensor],
    flows: list[torch.Tensor],
    certs: list[torch.Tensor],
    style_ref=None,
    occlusions_min_filter: int = 7,
) -> list[torch.Tensor]:
    """Stylize a clip with a diffusion backend, frame by frame.

    Args:
        frames_rgb: list of ``(1,3,H,W)`` RGB frames.
        flows: list (len N-1) of ``(1,2,H,W)`` ``(dy,dx)`` backward flows.
        certs: list (len N-1) of ``(1,1,H,W)`` certainties in [0,1].
    Returns:
        list of stylized ``(1,3,H,W)`` RGB frames in [0,1].
    """
    outputs = [stylizer.stylize_first(frames_rgb[0], style_ref)]
    for i in range(1, len(frames_rgb)):
        cond = build_conditioning(
            frames_rgb[i], outputs[-1], flows[i - 1], certs[i - 1], occlusions_min_filter
        )
        outputs.append(stylizer.stylize_next(frames_rgb[i], cond, style_ref))
    return outputs


class SDXLControlNetStylizer(DiffusionStylizer):
    """Real diffusion backend (SDXL + ControlNet + per-style LoRA) — M5 Max.

    Lazily imports ``diffusers``; the concrete denoise is implemented on the
    target (MPS) hardware where the base model + ControlNet + LoRA are available.
    Defined here so the pipeline seam is explicit and stable.
    """

    def __init__(self, cfg):
        try:
            import diffusers  # type: ignore  # noqa: F401
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "SDXLControlNetStylizer requires `diffusers` (install on the M5 Max / "
                "GPU host: pip install 'diffusers[torch]' transformers accelerate)"
            ) from e
        self.cfg = cfg
        raise NotImplementedError(
            "SDXL+ControlNet denoise is the Phase-3b step to implement on MPS; the "
            "conditioning bridge (fav.diffusion.conditioning) and this pipeline seam "
            "are ready — wire stack_controls(...) into a ControlNet img2img call."
        )

    def stylize_first(self, content_rgb, style_ref=None):  # pragma: no cover
        raise NotImplementedError

    def stylize_next(self, content_rgb, conditioning, style_ref=None):  # pragma: no cover
        raise NotImplementedError


def build_stylizer(cfg=None, backend: str = "dummy") -> DiffusionStylizer:
    """Construct a diffusion stylizer by backend name ('dummy' | 'sdxl')."""
    if backend == "dummy":
        return DummyDiffusionStylizer()
    if backend == "sdxl":
        return SDXLControlNetStylizer(cfg)
    raise ValueError(f"unknown diffusion backend {backend!r}; expected dummy|sdxl")
