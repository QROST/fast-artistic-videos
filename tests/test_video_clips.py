"""Tests for the real-video training source and build-dataset / stylize-vr CLI."""

import numpy as np
import pytest
import torch

from fav.data.video_clips import VideoClipsSource, build_video_tuple_batched
from fav.flow.estimator import DummyFlowEstimator


def _write_clip(d, n, size=48):
    pytest.importorskip("PIL")
    from PIL import Image

    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(d / f"frame_{i:05d}.png")


def test_build_video_tuple_batched_shapes():
    frames = [torch.rand(2, 3, 40, 40) for _ in range(3)]  # b=2, num=2
    imgs, flows, certs = build_video_tuple_batched(frames, DummyFlowEstimator(), use_structure=False)
    assert len(imgs) == 3 and len(flows) == 2 and len(certs) == 2
    assert imgs[0].shape == (2, 3, 40, 40)
    assert flows[0].shape == (2, 2, 40, 40)
    assert certs[0].shape == (2, 1, 40, 40)
    # zero flow -> interior reliable
    assert certs[0][0, 0, :39, :39].eq(1.0).all()


def test_video_clips_source_sample(tmp_path):
    vdir = tmp_path / "clips"
    _write_clip(vdir / "clipA", 5)
    _write_clip(vdir / "clipB", 4)
    src = VideoClipsSource(vdir, DummyFlowEstimator(), crop=32)
    assert len(src.clips) == 2
    imgs, flows, certs = src.sample(num=2, batch=3)
    assert len(imgs) == 3 and len(flows) == 2 and len(certs) == 2
    assert imgs[0].shape == (3, 3, 32, 32)
    assert flows[0].shape == (3, 2, 32, 32)


def test_video_clips_source_short_clip(tmp_path):
    vdir = tmp_path / "clips"
    _write_clip(vdir / "short", 2)  # only 2 frames; num=3 must degrade gracefully
    src = VideoClipsSource(vdir, DummyFlowEstimator(), crop=24)
    imgs, flows, certs = src.sample(num=3, batch=1)
    assert len(imgs) == 4 and len(flows) == 3  # last frame repeated to fill
    # Padded transitions (frames 2 and 3 are repeats of frame 1) are fully occluded.
    assert certs[0].sum() > 0          # real transition 0->1
    assert certs[1].sum() == 0         # padded
    assert certs[2].sum() == 0         # padded


def test_build_video_tuple_output_on_frames_device():
    # Contract: output tensors live on the frames' device regardless of estimator.
    frames = [torch.rand(1, 3, 24, 24) for _ in range(2)]
    imgs, flows, certs = build_video_tuple_batched(frames, DummyFlowEstimator(), use_structure=False)
    assert flows[0].device == frames[0].device
    assert certs[0].device == frames[0].device


def test_build_data_fn_video_fallback_label(tmp_path):
    # With no video_dir, a 'video' draw must report the actual data it produced.
    from fav import config as C
    from fav.cli import build_data_fn

    imgs_dir = tmp_path / "imgs"
    _write_clip(imgs_dir, 3, size=300)
    cfg = C.TrainConfig()
    cfg.data.image_dir = str(imgs_dir)
    cfg.data.data_mix = "video:1"  # only video, but no video_dir -> shift fallback
    cfg.batch_size = 1
    data_fn = build_data_fn(cfg, device="cpu")
    source, imgs, flows, certs = data_fn(1)
    assert source == "shift"  # not "video"


def test_build_dataset_writes_per_clip_assets(tmp_path):
    from fav.cli import run_build_dataset

    vdir = tmp_path / "clips"
    _write_clip(vdir / "c1", 3)
    _write_clip(vdir / "c2", 3)
    out = tmp_path / "ds"
    n = run_build_dataset(str(vdir), str(out), backend="dummy")
    assert n == 4  # 2 pairs per clip
    assert (out / "c1" / "backward_2_1.flo").exists()
    assert (out / "c2" / "reliable_3_2.pgm").exists()


def test_build_data_fn_uses_video_source(tmp_path):
    from fav import config as C
    from fav.cli import build_data_fn

    vdir = tmp_path / "clips"
    _write_clip(vdir / "clipA", 4)
    cfg = C.TrainConfig()
    cfg.data.video_dir = str(vdir)
    cfg.data.data_mix = "video:1"
    cfg.data.flow_backend = "dummy"
    cfg.batch_size = 2
    data_fn = build_data_fn(cfg, device="cpu")
    source, imgs, flows, certs = data_fn(1)
    assert source == "video"
    assert imgs[0].shape[0] == 2  # batch
    assert flows[0].shape[1] == 2


def test_stylize_vr_cli_end_to_end(tmp_path):
    pytest.importorskip("PIL")
    from fav import config as C
    from fav.cli import run_stylize_vr
    from fav.models.generator import Generator
    from fav.train.checkpoint import save_checkpoint
    from fav.flow.estimator import DummyFlowEstimator, compute_pair
    from fav.warp.flow_io import write_flo, write_pgm
    from fav.data.io_utils import save_rgb

    arch = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"
    model = Generator(arch)
    ckpt = tmp_path / "vr.pt"
    save_checkpoint(ckpt, model, None, 1, C.to_dict(C.TrainConfig(model=C.ModelConfig(arch=arch))), [])

    frames_dir = tmp_path / "frames"
    flow_dir = tmp_path / "flow"
    frames_dir.mkdir()
    est = DummyFlowEstimator()
    T = 2
    for t in range(1, T + 1):
        for f in range(1, 7):
            img = torch.rand(1, 3, 32, 32)
            save_rgb(frames_dir / f"frame_{t:05d}-{f}.png", img)
    # flow/occ assets for t=2 vs t=1, per face
    for f in range(1, 7):
        prev = torch.rand(1, 3, 32, 32)
        cur = torch.rand(1, 3, 32, 32)
        flow_uv, reliable = compute_pair(est, prev, cur, use_structure=False)
        write_flo(flow_dir / f"backward_2_1-{f}.flo", flow_uv)
        write_pgm(flow_dir / f"reliable_2_1-{f}.pgm", reliable)

    cfg = C.VRConfig()
    cfg.model_vid = str(ckpt)
    cfg.input_pattern = str(frames_dir / "frame_%05d-%d.png")
    cfg.flow_pattern = str(flow_dir / "backward_[%d]_{%d}-%d.flo")
    cfg.occlusions_pattern = str(flow_dir / "reliable_[%d]_{%d}-%d.pgm")
    cfg.output_prefix = str(tmp_path / "out")
    cfg.median_filter = 0
    paths = run_stylize_vr(cfg, device="cpu")
    assert len(paths) == T * 6
    assert all(p.exists() for p in paths)
