"""Planar video stylization with the legacy file-pattern interface.

Ports the CLI behavior of ``fast_artistic_video.lua``: frames are read by
``input_pattern`` (a printf like ``frame_%05d.ppm``), flow and occlusion files by
the bracketed templates ``backward_[%d]_{%d}.flo`` / ``reliable_[%d]_{%d}.pgm``
where ``[...]`` is the start frame index and ``{...}`` the target index. Outputs
are written as ``{output_prefix}-%05d.png``.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from fav.infer.core import stylize_sequence
from fav.warp.flow_io import read_flo, read_pgm, uv_to_dydx

_BRACKET = re.compile(r"\[([^\]]*)\]")
_BRACE = re.compile(r"\{([^}]*)\}")


def resolve_pattern(pattern: str, start: int, target: int) -> str:
    """Resolve a bracketed flow/occlusion template to a concrete filename.

    ``[spec]`` is formatted with ``start`` and ``{spec}`` with ``target`` using
    the printf spec inside, e.g. ``backward_[%02d]_{%02d}.flo`` -> ``backward_03_02.flo``.
    """
    out = _BRACKET.sub(lambda m: (m.group(1) or "%d") % start, pattern)
    out = _BRACE.sub(lambda m: (m.group(1) or "%d") % target, out)
    return out


def _load_rgb(path) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _save_rgb(path, img: torch.Tensor) -> None:
    from PIL import Image
    import numpy as np

    arr = (img[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _count_frames(input_pattern: str, start: int, limit: int) -> int:
    n = 0
    i = start
    while n < limit and Path(input_pattern % i).exists():
        n += 1
        i += 1
    return n


def stylize_video(cfg, model_vid, model_img="self", device="cpu") -> list[Path]:
    """Stylize a planar video described by ``cfg`` (a StylizeConfig). Returns paths."""
    start = cfg.continue_with
    num = _count_frames(cfg.input_pattern, start, cfg.num_frames)
    if num == 0:
        raise FileNotFoundError(f"no frames match {cfg.input_pattern} from {start}")

    frames, flows, certs = [], [], []
    for k in range(num):
        idx = start + k
        frames.append(_load_rgb(cfg.input_pattern % idx).to(device))
        if k > 0:
            flo = read_flo(resolve_pattern(cfg.flow_pattern, idx, idx - 1)).to(device)
            flows.append(uv_to_dydx(flo).unsqueeze(0))
            cert_px = read_pgm(resolve_pattern(cfg.occlusions_pattern, idx, idx - 1)).float()
            certs.append((cert_px / 255.0).view(1, 1, *cert_px.shape).to(device))

    outputs = stylize_sequence(
        model_vid, frames, flows, certs, model_img=model_img,
        preprocessing="vgg", occlusions_min_filter=cfg.occlusions_min_filter,
        median_filter_size=cfg.median_filter, fill_occlusions=cfg.fill_occlusions,
    )

    paths = []
    for k, out in enumerate(outputs):
        p = Path(f"{cfg.output_prefix}-{start + k:05d}.png")
        _save_rgb(p, out)
        paths.append(p)
    return paths
