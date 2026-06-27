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
    make_init_image,
    make_control_image,
    first_frame_conditioning,
    build_conditioning,
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


class _RecordingStylizer:
    """Returns a distinct constant per frame and records the conditioning it gets,
    so we can verify the loop feeds the PREVIOUS output (not a tautology)."""

    def __init__(self):
        self.conds = []
        self.n = 0

    def stylize_first(self, content_rgb, style_ref=None):
        self.n += 1
        return torch.full_like(content_rgb, 0.123)

    def stylize_next(self, content_rgb, conditioning, style_ref=None):
        self.conds.append(conditioning)
        self.n += 1
        return torch.full_like(content_rgb, 0.05 * self.n)  # distinct, != init image


def test_loop_feeds_previous_output_into_conditioning():
    s = _RecordingStylizer()
    torch.manual_seed(2)
    base = torch.rand(1, 3, 32, 32)
    imgs, flows, certs = make_shift(base, num=2, displ=(3, 2))
    outs = stylize_video_diffusion(s, imgs, flows, certs, occlusions_min_filter=1)
    # Each frame's conditioning must warp the actual PREVIOUS output by that flow —
    # verifies real loop wiring (the recorded outputs are constants unrelated to
    # the init-image formula, so this can't pass by construction).
    for i in range(1, len(outs)):
        cond = s.conds[i - 1]
        assert torch.allclose(cond.warped_prev, warp(outs[i - 1], flows[i - 1]), atol=1e-5)


def test_build_stylizer_dummy_and_unknown():
    assert isinstance(build_stylizer(backend="dummy"), DummyDiffusionStylizer)
    with pytest.raises(ValueError):
        build_stylizer(backend="nope")


def test_sdxl_requires_diffusers():
    # Without diffusers the backend must fail at construction with a clear message.
    if importlib.util.find_spec("diffusers") is not None:
        pytest.skip("diffusers installed; missing-dep path not exercised")
    with pytest.raises(RuntimeError) as e:
        SDXLControlNetStylizer(DiffusionConfig())
    assert "diffusers" in str(e.value).lower()


def test_make_init_image_temporal_anchor():
    # Reliable region -> warped previous output; occluded region -> content.
    content = torch.rand(1, 3, 16, 16)
    cond = first_frame_conditioning(content)  # cert all zero -> init == content
    assert torch.allclose(make_init_image(content, cond), content.clamp(0, 1))
    # A fully-certain bundle -> init == warped_prev_masked.
    prev = torch.rand(1, 3, 16, 16)
    flow = torch.zeros(1, 2, 16, 16)
    cert = torch.ones(1, 1, 16, 16)
    c2 = build_conditioning(content, prev, flow, cert, occlusions_min_filter=1)
    assert torch.allclose(make_init_image(content, c2), c2.warped_prev_masked.clamp(0, 1))


def test_make_control_image_channels():
    content = torch.rand(1, 3, 16, 16)
    cond = first_frame_conditioning(content)
    assert make_control_image(cond, "structure").shape == (1, 3, 16, 16)
    assert make_control_image(cond, "flow").shape == (1, 3, 16, 16)
    with pytest.raises(ValueError):
        make_control_image(cond, "nope")


def test_diffusion_config_defaults():
    c = DiffusionConfig()
    assert "warped_prev_masked" in c.controls
    assert c.num_inference_steps > 0 and 0 < c.strength <= 1
    assert 0 < c.first_strength <= 1 and c.controlnet

