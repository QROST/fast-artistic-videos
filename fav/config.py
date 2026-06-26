"""Typed configuration for training, stylization and VR.

Every default below equals the verified default from the original Lua code
(``train_video.lua`` / ``fast_artistic_video*.lua``) so that an unconfigured run
reproduces the paper settings. Configs can be loaded from YAML and overridden by
CLI flags (see ``fav/cli.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    # The improved architecture from the FAV paper (nearest-neighbor upsampling
    # avoids checkerboard artifacts). Input is always 7 channels.
    arch: str = "c9s1-32,d64,d128,R128,R128,R128,R128,R128,U2,c3s1-64,U2,c9s1-3"
    use_instance_norm: bool = True
    padding_type: str = "reflect-start"  # zero|reflect|replicate|reflect-start
    tanh_constant: float = 150.0
    tv_strength: float = 1e-6
    preprocessing: str = "vgg"  # vgg|resnet


@dataclass
class LossConfig:
    loss_network: str = "models/vgg16.t7"  # source for the caffe VGG-16 weights
    content_layers: tuple[int, ...] = (16,)            # relu3_3
    content_weights: tuple[float, ...] = (1.0,)
    style_layers: tuple[int, ...] = (4, 9, 16, 23)     # relu1_2,2_2,3_3,4_3
    style_weights: tuple[float, ...] = (10.0,)
    style_image_size: int = 384
    style_target_type: str = "gram"  # gram|mean
    pixel_loss_type: str = "L2"      # L2|L1|SmoothL1
    pixel_loss_weight: float = 50.0  # raise to 100 for mixed/multi-frame training
    percep_loss_weight: float = 1.0


@dataclass
class DataConfig:
    h5_file: str = ""               # single-image source (COCO-style), optional
    h5_file_video: str = ""         # legacy packed video dataset, optional
    image_dir: str = ""             # streaming single-image source (preferred)
    video_dir: str = ""             # streaming real-video clips source (preferred)
    data_mix: str = "shift:1,zoom_out:1,video:3"
    num_frame_steps: str = "0:1"
    single_image_until: int = 0
    reliable_map_min_filter: int = 7
    fill_occlusions: str = "vgg-mean"  # vgg-mean|uniform-random
    train_img_size: str = "256:256"    # H:W training crop
    source_img_size: int = 384         # images are stored/loaded at this size
    flow_backend: str = "raft"         # estimator for the real-video source


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    style_image: str = ""
    image_model: str = "self"  # 'self' or path to a single-image first-frame model
    num_iterations: int = 60000
    batch_size: int = 4
    learning_rate: str = "1e-3"  # supports schedule 'r0,iter:r1,...'
    lr_decay_every: int = -1
    lr_decay_factor: float = 0.5
    weight_decay: float = 0.0
    checkpoint_name: str = "checkpoint"
    checkpoint_every: int = 1000
    history_every: int = 100
    num_val_batches: int = 100
    images_every: int = 100
    print_every: int = 10
    device: str | None = None  # None=auto (mps>cuda>cpu)
    seed: int = 0


@dataclass
class StylizeConfig:
    model_vid: str = ""
    model_img: str = "self"
    input_pattern: str = ""
    flow_pattern: str = ""
    occlusions_pattern: str = ""
    output_prefix: str = "out"
    num_frames: int = 9999
    continue_with: int = 1
    backward: bool = False
    occlusions_min_filter: int = 7
    median_filter: int = 3
    fill_occlusions: str = "vgg-mean"
    scale_factor: float = 1.0
    create_inconsistent: bool = False
    device: str | None = None


@dataclass
class VRConfig(StylizeConfig):
    overlap_pixel_w: int = 20
    overlap_pixel_h: int = 20
    out_cubemap: bool = False
    out_equi: bool = False
    out_equi_w: int = 2560
    out_equi_h: int = 1440
    create_inconsistent_border: bool = False
    vr_backend: str = "py360convert"  # py360convert|custom


def to_dict(cfg: Any) -> dict:
    return asdict(cfg)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a plain dict (requires pyyaml)."""
    import yaml  # lazy import; only needed when YAML configs are used

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_overrides(cfg: Any, overrides: dict) -> Any:
    """Return a deep copy of dataclass ``cfg`` with ``overrides`` applied.

    Walks the live instance (rather than its annotations) so nested dataclasses
    are recognized even under ``from __future__ import annotations``. Override
    values that are dicts recurse into nested dataclasses; unknown keys raise
    ``KeyError`` so typos surface early.
    """
    import copy
    import dataclasses

    cfg = copy.deepcopy(cfg)
    _apply(cfg, overrides)
    return cfg


def _apply(obj, updates: dict) -> None:
    import dataclasses

    for key, value in updates.items():
        if not hasattr(obj, key):
            raise KeyError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if isinstance(value, dict) and dataclasses.is_dataclass(current):
            _apply(current, value)
        else:
            setattr(obj, key, value)
