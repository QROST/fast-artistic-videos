"""Streaming single-image source (COCO-style), replacing the HDF5 image dataset.

Yields RGB ``[0,1]`` tensors. Images are loaded at ``source_size`` (default 384)
and a random ``crop`` (default 256) is taken — matching the legacy "multiple
smaller crops resized to 256" behavior. Preprocessing (VGG/resnet) is applied by
the caller so the source stays format-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}


def _load_image(path: Path) -> torch.Tensor:
    """Load an image file as an RGB ``[0,1]`` tensor ``(3, H, W)``."""
    try:
        from PIL import Image  # lazy: only needed for real I/O
    except Exception as e:  # pragma: no cover
        raise RuntimeError("loading images requires Pillow (pip install pillow)") from e
    import numpy as np

    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _random_crop(img: torch.Tensor, size: int, generator=None) -> torch.Tensor:
    _, h, w = img.shape
    if h < size or w < size:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(max(h, size), max(w, size)), mode="bilinear", align_corners=False
        )[0]
        _, h, w = img.shape
    top = int(torch.randint(0, h - size + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, w - size + 1, (1,), generator=generator).item())
    return img[:, top : top + size, left : left + size].contiguous()


class ImageFolderSource(Dataset):
    def __init__(self, root: str | Path, source_size: int = 384, crop: int = 256, generator=None):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in _IMAGE_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"no images under {self.root}")
        self.source_size = source_size
        self.crop = crop
        self.generator = generator

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = _load_image(self.paths[idx])
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(self.source_size, self.source_size),
            mode="bilinear", align_corners=False,
        )[0]
        return _random_crop(img, self.crop, self.generator)
