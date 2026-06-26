"""Tests for the data pipeline: synthetic sources, schedules, mix, real video."""

import torch

from fav.data.synthetic import make_shift, make_zoom_out, make_single_image, SyntheticSource
from fav.data.mixed import DataMix
from fav.data.video_real import video_tuple_from_frames
from fav.flow.estimator import DummyFlowEstimator
from fav.train.schedules import parse_step_schedule, parse_lr_schedule, value_at
from fav.warp.grid_sample import warp


def test_shift_shapes_and_lists():
    img = torch.randn(2, 3, 64, 64)
    imgs, flows, certs = make_shift(img, num=3, displ=(5, -4))
    assert len(imgs) == 4 and len(flows) == 3 and len(certs) == 3
    assert imgs[0].shape == (2, 3, 64, 64)
    assert flows[0].shape == (2, 2, 64, 64)
    assert certs[0].shape == (2, 1, 64, 64)


def test_shift_warp_consistency_cross_check():
    # THE key cross-check: warp(frame_i, flow_i) must match frame_{i+1} inside the
    # reliable region — this links the warp port to the synthetic flow convention.
    torch.manual_seed(0)
    img = torch.randn(1, 3, 80, 80)
    dx, dy = 6, -5
    imgs, flows, certs = make_shift(img, num=2, displ=(dx, dy))
    for i in range(2):
        warped = warp(imgs[i], flows[i])
        cert = certs[i]
        # Compare only well-inside-the-reliable-region pixels (avoid bilinear edge
        # effects at the occlusion boundary).
        m = torch.zeros_like(cert)
        margin = max(abs(dx), abs(dy)) + 2
        m[:, :, margin:-margin, margin:-margin] = cert[:, :, margin:-margin, margin:-margin]
        diff = ((warped - imgs[i + 1]) * m).abs().sum() / m.sum().clamp_min(1)
        assert diff < 1e-3, f"step {i} warp/flow mismatch: {diff}"


def test_shift_cert_zeros_occluded_border():
    img = torch.randn(1, 3, 40, 40)
    _, _, certs = make_shift(img, num=1, displ=(8, 0))  # pan right -> right border occluded
    cert = certs[0][0, 0]
    assert cert[:, -1].sum() == 0  # rightmost column occluded
    assert cert[:, 0].sum() > 0    # left column reliable


def test_zoom_out_shapes():
    img = torch.randn(2, 3, 64, 64)
    imgs, flows, certs = make_zoom_out(img, num=2, displ=(4, 3))
    assert len(imgs) == 3 and len(flows) == 2 and len(certs) == 2
    assert imgs[-1].shape == (2, 3, 64, 64)
    assert flows[0].shape == (2, 2, 64, 64)


def test_single_image_empty_prior():
    img = torch.randn(2, 3, 32, 32)
    imgs, flows, certs = make_single_image(img)
    assert len(imgs) == 2 and len(flows) == 1 and len(certs) == 1
    assert torch.count_nonzero(imgs[0]) == 0       # empty prior
    assert torch.count_nonzero(certs[0]) == 0      # all occluded
    assert torch.equal(imgs[1], img)


def test_synthetic_source_dispatch():
    src = SyntheticSource()
    img = torch.randn(1, 3, 48, 48)
    for mode in ("shift", "zoom_out", "single_image"):
        imgs, flows, certs = src.sample(mode, img, num=1)
        assert len(imgs) == 2 and len(flows) == 1 and len(certs) == 1


def test_schedules():
    sched = parse_step_schedule("0:1,50000:2,60000:4")
    assert sched == [(0, 1), (50000, 2), (60000, 4)]
    assert value_at(sched, 1) == 1
    assert value_at(sched, 50001) == 2
    assert value_at(sched, 70000) == 4

    lr = parse_lr_schedule("1e-3,50000:1e-4")
    assert lr[0] == (0, 1e-3) and lr[1] == (50000, 1e-4)
    assert value_at(lr, 10) == 1e-3
    assert value_at(lr, 60000) == 1e-4


def test_data_mix_weights():
    mix = DataMix("shift:1,zoom_out:1,video:3")
    assert mix.total == 5
    assert mix.needs_real_video and mix.needs_synthetic
    counts = {"shift": 0, "zoom_out": 0, "video": 0}
    torch.manual_seed(0)
    for _ in range(2000):
        counts[mix.sample()] += 1
    # video has weight 3/5 -> roughly 60% of samples.
    assert 0.5 < counts["video"] / 2000 < 0.7


def test_video_tuple_from_frames_dummy_estimator():
    frames = [torch.rand(1, 3, 32, 32) for _ in range(3)]
    imgs, flows, certs = video_tuple_from_frames(frames, DummyFlowEstimator(), use_structure=False)
    assert len(imgs) == 3 and len(flows) == 2 and len(certs) == 2
    assert flows[0].shape == (1, 2, 32, 32)
    assert certs[0].shape == (1, 1, 32, 32)
    # Zero flow -> interior reliable (cert == 1).
    assert certs[0][0, 0, :31, :31].eq(1.0).all()
