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
import torch.nn.functional as F


def _round8(x: int) -> int:
    """Round a dimension to the nearest positive multiple of 8 (SDXL/VAE needs /8)."""
    return max(8, int(round(x / 8.0)) * 8)

from fav.diffusion.conditioning import (
    ConditioningBundle,
    build_conditioning,
    first_frame_conditioning,
)


def make_init_image(content_rgb: torch.Tensor, cond: ConditioningBundle) -> torch.Tensor:
    """img2img init = temporal anchor: warped previous output where the flow is
    reliable, content where occluded. ``(N,3,H,W)`` RGB ``[0,1]``. (Pure torch.)
    """
    return (cond.warped_prev_masked + content_rgb * (1.0 - cond.cert)).clamp(0, 1)


def make_control_image(cond: ConditioningBundle, control: str = "structure") -> torch.Tensor:
    """ControlNet conditioning image as 3-channel RGB ``[0,1]``. (Pure torch.)"""
    if control == "structure":
        return cond.structure.repeat(1, 3, 1, 1)
    if control == "flow":
        return cond.flow_image
    if control == "content":
        return cond.content
    raise ValueError(f"unknown control {control!r}; expected structure|flow|content")


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
        # Same temporal-anchor init the real backend denoises from.
        return make_init_image(content_rgb, conditioning)


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
    """SDXL + ControlNet + per-style LoRA backend (runs on the M5 Max / MPS).

    Per frame: img2img-denoise from the temporal-anchor init image
    (:func:`make_init_image`) under a structure ControlNet
    (:func:`make_control_image`) with a per-style LoRA + style prompt. The
    occlusion certainty decides where the previous output is trusted, so the
    Phase-1 temporal recipe carries over. ``diffusers`` is imported lazily; the
    denoise itself is validated on the target hardware (not in CPU CI), but the
    tensor<->pipeline plumbing and the conditioning helpers are pure-torch and
    unit-tested.

    Note: this is an *init-only* temporal scheme — the init image anchors each
    frame but the denoise can still drift, so higher ``strength`` trades more
    stylization for less stability. Cross-frame attention (Phase 3c) is the next
    lever if residual flicker matters.
    """

    def __init__(self, cfg):
        try:
            import torch as _torch  # noqa: F401
            from diffusers import (  # type: ignore
                ControlNetModel,
                StableDiffusionXLControlNetImg2ImgPipeline,
            )
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "SDXLControlNetStylizer requires `diffusers` (install on the M5 Max / "
                "GPU host: pip install 'diffusers[torch]' transformers accelerate)"
            ) from e

        from fav.device import select_device, _PRECISION_DTYPE

        self.cfg = cfg
        self.device = select_device(cfg.device)
        # Honor bf16 vs fp16 vs fp32 (a plain `in (...)` check silently loaded fp16
        # for bf16). bf16 is the safer default on MPS.
        self.dtype = _PRECISION_DTYPE.get(cfg.precision, torch.float32)
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet, torch_dtype=self.dtype)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            cfg.base_model, controlnet=controlnet, torch_dtype=self.dtype
        )
        if cfg.lora_path:
            pipe.load_lora_weights(cfg.lora_path)
        pipe = pipe.to(self.device)
        # SDXL's fp16/bf16 VAE produces black/NaN frames; upcast it to fp32. Enable
        # tiling/slicing for unified-memory headroom on the M5 Max.
        for fn, args in (("upcast_vae", ()), ("enable_vae_tiling", ()), ("enable_attention_slicing", ())):
            if hasattr(pipe, fn):
                try:
                    getattr(pipe, fn)(*args)
                except Exception:  # best-effort; not all builds expose all of these
                    pass
        self.pipe = pipe
        self.prompt = cfg.prompt or cfg.style_ref or "an artwork"

    def _to_pil(self, t: torch.Tensor):
        from PIL import Image
        import numpy as np

        arr = (t[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
        return Image.fromarray(arr)

    def _from_pil(self, im) -> torch.Tensor:
        import numpy as np

        arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _run(self, init_rgb, control_rgb, strength):  # pragma: no cover - needs diffusers+models
        # SDXL/VAE require dims divisible by 8; round, run, then restore size.
        h, w = init_rgb.shape[-2], init_rgb.shape[-1]
        h8, w8 = _round8(h), _round8(w)
        if (h8, w8) != (h, w):
            init_rgb = F.interpolate(init_rgb, (h8, w8), mode="bilinear", align_corners=False)
            control_rgb = F.interpolate(control_rgb, (h8, w8), mode="bilinear", align_corners=False)
        out = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.cfg.negative_prompt or None,
            image=self._to_pil(init_rgb),
            control_image=self._to_pil(control_rgb),
            strength=strength,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            controlnet_conditioning_scale=self.cfg.controlnet_scale,
        ).images[0]
        out_t = self._from_pil(out)
        if out_t.shape[-2] != h or out_t.shape[-1] != w:
            out_t = F.interpolate(out_t, (h, w), mode="bilinear", align_corners=False)
        return out_t

    def stylize_first(self, content_rgb, style_ref=None):  # pragma: no cover
        cond = first_frame_conditioning(content_rgb)
        return self._run(make_init_image(content_rgb, cond),
                         make_control_image(cond, self.cfg.control), self.cfg.first_strength)

    def stylize_next(self, content_rgb, conditioning, style_ref=None):  # pragma: no cover
        return self._run(make_init_image(content_rgb, conditioning),
                         make_control_image(conditioning, self.cfg.control), self.cfg.strength)


def build_stylizer(cfg=None, backend: str = "dummy") -> DiffusionStylizer:
    """Construct a diffusion stylizer by backend name ('dummy' | 'sdxl')."""
    if backend == "dummy":
        return DummyDiffusionStylizer()
    if backend == "sdxl":
        return SDXLControlNetStylizer(cfg)
    raise ValueError(f"unknown diffusion backend {backend!r}; expected dummy|sdxl")
