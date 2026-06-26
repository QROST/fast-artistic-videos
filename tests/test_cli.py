"""Tests for the CLI helpers: config merge, compute-flow, data_fn, package import."""

import pytest
import torch

from fav import config as C
from fav.cli import build_data_fn, run_compute_flow, _apply_overrides, _coerce


def test_config_override_dotted():
    cfg = C.TrainConfig()
    cfg2 = _apply_overrides(cfg, ["num_iterations=5", "loss.pixel_loss_weight=100", "model.tanh_constant=120"])
    assert cfg2.num_iterations == 5
    assert cfg2.loss.pixel_loss_weight == 100.0
    assert cfg2.model.tanh_constant == 120.0
    # Original unchanged.
    assert cfg.num_iterations == 60000


def test_coerce_types():
    assert _coerce("5") == 5
    assert _coerce("1e-3") == 0.001
    assert _coerce("true") is True
    assert _coerce("self") == "self"


def test_config_yaml_merge(tmp_path):
    pytest.importorskip("yaml")
    p = tmp_path / "c.yaml"
    p.write_text("num_iterations: 7\nloss:\n  pixel_loss_weight: 99\n")
    cfg = C.merge_overrides(C.TrainConfig(), C.load_yaml(p))
    assert cfg.num_iterations == 7 and cfg.loss.pixel_loss_weight == 99


def _write_frames(d, n, size=32):
    pytest.importorskip("PIL")
    from PIL import Image
    import numpy as np

    for i in range(1, n + 1):
        arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(d / f"frame_{i:05d}.png")


def test_compute_flow_dummy_writes_assets(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    out = tmp_path / "flow"
    _write_frames(frames, 3)
    n = run_compute_flow(str(frames), str(out), backend="dummy", start=1)
    assert n == 2
    assert (out / "backward_2_1.flo").exists()
    assert (out / "reliable_2_1.pgm").exists()
    assert (out / "backward_3_2.flo").exists()


def test_build_data_fn_synthetic(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    _write_frames(imgs, 4, size=300)
    cfg = C.TrainConfig()
    cfg.data.image_dir = str(imgs)
    cfg.data.data_mix = "shift:1,single_image:1"
    cfg.batch_size = 2
    data_fn = build_data_fn(cfg, device="cpu")
    source, imgs_list, flows, certs = data_fn(1)
    assert source in ("shift", "single_image")
    assert imgs_list[0].shape[0] == 2  # batch
    assert imgs_list[0].shape[2:] == (256, 256)  # train crop


def test_package_imports():
    # Smoke import of the whole package surface.
    import fav
    import fav.models, fav.losses, fav.warp, fav.occlusion
    import fav.flow, fav.data, fav.train, fav.infer, fav.vr, fav.cli
    assert fav.__version__
