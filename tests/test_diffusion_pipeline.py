"""Tests for the Phase-3 diffusion video pipeline skeleton."""

import importlib.util

import pytest
import torch

from fav.config import DiffusionConfig
from fav.diffusion import (
    DummyDiffusionStylizer,
    SDXLControlNetStylizer,
    stylize_video_diffusion,
    build_stylizer,
)
from fav.data.synthetic import make_shift
from fav.warp.grid_sample import warp


def test_dummy_stylizer_first_and_next():
    s = DummyDiffusionStylizer()
    content = torch.rand(1, 3, 32, 32)
    assert torch.allclose(s.stylize_first(content), content.clamp(0, 1))


def test_video_loop_shapes_and_range():
    s = DummyDiffusionStylizer()
    torch.manual_seed(0)
    base = torch.rand(1, 3, 48, 48)
    imgs, flows, certs = make_shift(base, num=3, displ=(4, -3))
    outs = stylize_video_diffusion(s, imgs, flows, certs)
    assert len(outs) == 4
    for o in outs:
        assert o.shape == (1, 3, 48, 48)
        assert float(o.min()) >= 0 and float(o.max()) <= 1


def test_video_loop_temporal_consistency():
    # The dummy backend keeps the warped previous output where reliable, so the
    # loop must be temporally consistent: warp(out[i-1]) == out[i] in the certain
    # region. This validates the conditioning -> backend -> loop wiring.
    s = DummyDiffusionStylizer()
    torch.manual_seed(1)
    base = torch.rand(1, 3, 64, 64)
    dx, dy = 5, -4
    imgs, flows, certs = make_shift(base, num=2, displ=(dx, dy))
    outs = stylize_video_diffusion(s, imgs, flows, certs, occlusions_min_filter=1)
    for i in range(1, len(outs)):
        warped_prev = warp(outs[i - 1], flows[i - 1])
        margin = max(abs(dx), abs(dy)) + 2
        m = torch.zeros_like(certs[i - 1])
        m[:, :, margin:-margin, margin:-margin] = certs[i - 1][:, :, margin:-margin, margin:-margin]
        diff = ((outs[i] - warped_prev) * m).abs().sum() / m.sum().clamp_min(1)
        assert diff < 1e-3


def test_build_stylizer_dummy_and_unknown():
    assert isinstance(build_stylizer(backend="dummy"), DummyDiffusionStylizer)
    with pytest.raises(ValueError):
        build_stylizer(backend="nope")


def test_sdxl_stub_requires_diffusers_or_not_implemented():
    cfg = DiffusionConfig()
    if importlib.util.find_spec("diffusers") is None:
        with pytest.raises(RuntimeError) as e:
            SDXLControlNetStylizer(cfg)
        assert "diffusers" in str(e.value).lower()
    else:
        # diffusers present: construction reaches the not-yet-implemented denoise.
        with pytest.raises(NotImplementedError):
            SDXLControlNetStylizer(cfg)


def test_diffusion_config_defaults():
    c = DiffusionConfig()
    assert "warped_prev_masked" in c.controls
    assert c.num_inference_steps > 0 and 0 < c.strength <= 1
