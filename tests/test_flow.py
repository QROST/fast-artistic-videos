"""Tests for the flow estimator interface, rescale, and dataset writing."""

import torch

from fav.flow import build_estimator, DummyFlowEstimator
from fav.flow.estimator import (
    rescale_flow,
    flow_filename,
    occlusion_filename,
    compute_pair,
    write_pair,
)
from fav.warp.flow_io import read_flo, read_pgm


def test_dummy_estimator_zero_and_shape():
    est = DummyFlowEstimator()
    a = torch.rand(2, 3, 9, 11)
    b = torch.rand(2, 3, 9, 11)
    flow = est.estimate(a, b)
    assert flow.shape == (2, 2, 9, 11)
    assert torch.count_nonzero(flow) == 0
    fwd, bwd = est.estimate_bidirectional(a, b)
    assert fwd.shape == bwd.shape == (2, 2, 9, 11)


def test_build_estimator_dummy():
    assert isinstance(build_estimator("dummy"), DummyFlowEstimator)


def test_rescale_flow_magnitude():
    # A constant flow of (u=2, v=4) at 8x8 -> at 16x16 it should double in u and v
    # because a pixel displacement scales with resolution.
    flow = torch.zeros(1, 2, 8, 8)
    flow[:, 0] = 2.0
    flow[:, 1] = 4.0
    up = rescale_flow(flow, (16, 16))
    assert up.shape == (1, 2, 16, 16)
    assert torch.allclose(up[:, 0], torch.full((1, 16, 16), 4.0), atol=1e-4)
    assert torch.allclose(up[:, 1], torch.full((1, 16, 16), 8.0), atol=1e-4)
    # Identity when size unchanged.
    assert torch.equal(rescale_flow(flow, (8, 8)), flow)


def test_filenames_match_legacy_pattern():
    assert flow_filename(5, 4) == "backward_5_4.flo"
    assert occlusion_filename(5, 4) == "reliable_5_4.pgm"


def test_compute_pair_and_write(tmp_path):
    est = DummyFlowEstimator()
    prev = torch.rand(1, 3, 16, 16)
    cur = torch.rand(1, 3, 16, 16)
    flow_uv, reliable = compute_pair(est, prev, cur, use_structure=False)
    assert flow_uv.shape == (2, 16, 16)
    assert reliable.shape == (16, 16)
    # Zero flow -> interior reliable (255).
    assert reliable[:15, :15].eq(255).all()

    write_pair(tmp_path, 5, 4, flow_uv, reliable)
    assert (tmp_path / "backward_5_4.flo").exists()
    assert (tmp_path / "reliable_5_4.pgm").exists()
    back = read_flo(tmp_path / "backward_5_4.flo")
    assert torch.allclose(back, flow_uv, atol=1e-5)
    pgm = read_pgm(tmp_path / "reliable_5_4.pgm")
    assert pgm.shape == (16, 16)
