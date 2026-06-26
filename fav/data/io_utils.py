"""Small shared image I/O helpers (lazy Pillow import)."""

from __future__ import annotations

from pathlib import Path

import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def load_rgb(path: str | Path, batched: bool = True) -> torch.Tensor:
    """Load an image file as an RGB ``[0,1]`` tensor.

    Returns ``(1,3,H,W)`` when ``batched`` else ``(3,H,W)``.
    """
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype="float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.unsqueeze(0) if batched else t


def save_rgb(path: str | Path, img: torch.Tensor) -> None:
    """Save a ``(1,3,H,W)`` or ``(3,H,W)`` RGB ``[0,1]`` tensor to ``path``."""
    from PIL import Image
    import numpy as np

    if img.dim() == 4:
        img = img[0]
    arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def list_frames(directory: str | Path) -> list[Path]:
    """Sorted image files directly inside ``directory``."""
    d = Path(directory)
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
