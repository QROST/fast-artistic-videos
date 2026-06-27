"""Optional auxiliary perceptual losses (Phase-2): LPIPS and DINOv2.

These are *additive* content-fidelity terms layered on top of the faithful VGG
perceptual + Gram style loss, gated by ``loss.lpips_weight`` / ``loss.dino_weight``
(both 0 by default, so the faithful path is unchanged). They operate on RGB
``[0,1]`` images. Dependencies are imported lazily and only when enabled, so the
base package stays dependency-light.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _make_lpips(device):
    try:
        import lpips  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError("loss.lpips_weight > 0 requires `pip install lpips`") from e
    net = lpips.LPIPS(net="vgg").to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    def fn(out_rgb, content_rgb):
        # LPIPS expects inputs in [-1, 1].
        return net(out_rgb * 2 - 1, content_rgb * 2 - 1).mean()

    return fn


def _make_dino(device):
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception as e:  # pragma: no cover - optional dep / network
        raise RuntimeError(
            "loss.dino_weight > 0 requires torch.hub access to facebookresearch/dinov2"
        ) from e
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _feat(rgb):
        # DINOv2 needs spatial dims divisible by 14 and ImageNet normalization.
        h = (rgb.shape[-2] // 14) * 14
        w = (rgb.shape[-1] // 14) * 14
        x = F.interpolate(rgb, size=(max(h, 14), max(w, 14)), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return model(x)

    def fn(out_rgb, content_rgb):
        return F.mse_loss(_feat(out_rgb), _feat(content_rgb))

    return fn


def build_aux_terms(loss_cfg, device) -> list[tuple[float, object]]:
    """Return ``[(weight, fn(out_rgb, content_rgb) -> scalar), ...]`` for enabled terms."""
    terms = []
    if getattr(loss_cfg, "lpips_weight", 0.0) > 0:
        terms.append((float(loss_cfg.lpips_weight), _make_lpips(device)))
    if getattr(loss_cfg, "dino_weight", 0.0) > 0:
        terms.append((float(loss_cfg.dino_weight), _make_dino(device)))
    return terms
