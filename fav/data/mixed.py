"""Weighted data-source mixing, ported from the train_video.lua mixing logic.

Parses a ``data_mix`` string like ``"shift:1,zoom_out:1,video:3,single_image:5"``
into an integer-weighted "wheel" and samples a source per step with probability
proportional to its weight. ``single_image_until`` (forcing the single-image
source early in training) is applied by the training loop, not here.
"""

from __future__ import annotations

import torch


class DataMix:
    def __init__(self, spec: str, generator=None):
        self.weights: dict[str, int] = {}
        self.wheel: list[str] = []
        self.unique: list[str] = []
        self.generator = generator
        for part in spec.split(","):
            source, count = part.split(":")
            count = int(count)
            self.weights[source] = count
            self.unique.append(source)
            self.wheel.extend([source] * count)
        self.total = len(self.wheel)
        if self.total == 0:
            raise ValueError(f"empty data_mix: {spec!r}")

    def sample(self) -> str:
        idx = int(torch.randint(0, self.total, (1,), generator=self.generator).item())
        return self.wheel[idx]

    def uses(self, source: str) -> bool:
        return source in self.weights

    @property
    def needs_real_video(self) -> bool:
        return "video" in self.weights

    @property
    def needs_synthetic(self) -> bool:
        return any(s != "video" for s in self.weights)
