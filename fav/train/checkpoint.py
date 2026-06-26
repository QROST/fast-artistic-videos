"""Checkpoint save/load (.pt) with a JSON history sidecar.

Mirrors the legacy ``train_video.lua`` checkpointing: a torch checkpoint holding
the model + optimizer state + iteration + config, plus a ``.json`` with the loss
history. ``resume`` restores the iteration so training continues seamlessly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, iteration, config_dict, history) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "iter": iteration,
            "config": config_dict,
            "history": history,
        },
        path,
    )
    # JSON sidecar (config + history) for quick inspection.
    sidecar = path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({"iter": iteration, "config": config_dict, "history": history}, f, indent=2)
    return path


def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
