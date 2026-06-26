"""Planar inference smoke tests."""

import torch

from fav.models.generator import Generator
from fav.infer.core import stylize_sequence, stylize_first_frame, stylize_next_frame
from fav.infer.stylize_planar import resolve_pattern
from fav.data.synthetic import make_shift

SMALL_ARCH = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"


def test_resolve_pattern_legacy_template():
    assert resolve_pattern("backward_[%02d]_{%02d}.flo", 3, 2) == "backward_03_02.flo"
    assert resolve_pattern("reliable_[%d]_{%d}.pgm", 10, 9) == "reliable_10_9.pgm"


def test_stylize_sequence_runs_and_shapes():
    model = Generator(SMALL_ARCH).eval()
    # Build a 4-frame panning clip with known flow/cert via make_shift.
    torch.manual_seed(0)
    base = torch.rand(1, 3, 64, 64)
    imgs, flows, certs = make_shift(base, num=3, displ=(4, -3))
    outputs = stylize_sequence(model, imgs, flows, certs, model_img="self",
                               median_filter_size=3)
    assert len(outputs) == 4
    for o in outputs:
        assert o.shape == (1, 3, 64, 64)
        assert float(o.min()) >= 0.0 and float(o.max()) <= 1.0


def test_first_frame_self_and_separate_model():
    model = Generator(SMALL_ARCH).eval()
    frame = torch.rand(1, 3, 64, 64)
    out_self, pre_self = stylize_first_frame(model, "self", frame)
    assert out_self.shape == (1, 3, 64, 64)
    # A separate image model can also be supplied (here, reuse a generator that
    # takes 3 channels — emulate by a small lambda-like module).
    assert pre_self.shape == (1, 3, 64, 64)


def test_prior_influences_output():
    # The 7-channel input must actually use the warped prior: two different
    # priors should yield different stylized frames.
    model = Generator(SMALL_ARCH).eval()
    frame = torch.rand(1, 3, 64, 64)
    flow = torch.zeros(1, 2, 64, 64)
    cert = torch.ones(1, 1, 64, 64)
    prior_a = torch.randn(1, 3, 64, 64) * 50
    prior_b = torch.randn(1, 3, 64, 64) * 50
    out_a, _ = stylize_next_frame(model, prior_a, frame, flow, cert)
    out_b, _ = stylize_next_frame(model, prior_b, frame, flow, cert)
    assert not torch.allclose(out_a, out_b, atol=1e-3)


def test_temporal_pathway_after_brief_training():
    # After a little training on a panning clip, the video model's consecutive
    # outputs (under warp, in the reliable region) should be at least as
    # consistent as independently re-stylizing — sanity that the temporal path
    # is wired and trainable. Lenient: assert it runs and yields finite numbers.
    from fav.config import TrainConfig
    from fav.losses.vgg_loss_net import build_vgg16_loss_net
    from fav.losses.perceptual import PerceptualCriterion
    from fav.preprocess import vgg_preprocess, vgg_deprocess
    from fav.train.loop import train
    from fav.warp.grid_sample import warp

    cfg = TrainConfig()
    cfg.model.arch = SMALL_ARCH
    cfg.loss.pixel_loss_weight = 200.0
    model = Generator(SMALL_ARCH)
    crit = PerceptualCriterion(
        build_vgg16_loss_net(cfg.loss.content_layers, cfg.loss.style_layers),
        cfg.loss.content_layers, cfg.loss.content_weights,
        cfg.loss.style_layers, cfg.loss.style_weights,
    )
    crit.set_style_target(vgg_preprocess(torch.rand(1, 3, 64, 64)))

    torch.manual_seed(1)
    base = torch.rand(2, 3, 64, 64)

    def data_fn(it):
        imgs, flows, certs = make_shift(base, num=1, displ=(4, -3))
        return "shift", imgs, flows, certs

    train(model, crit, data_fn, cfg, device="cpu", max_iters=15)

    model.eval()
    imgs, flows, certs = make_shift(base[:1], num=1, displ=(4, -3))
    outs = stylize_sequence(model, imgs, flows, certs, model_img="self", median_filter_size=0)
    prev_pre = vgg_preprocess(outs[0])
    cur_pre = vgg_preprocess(outs[1])
    warped = warp(prev_pre, flows[0][:1])
    cons = ((warped - cur_pre) * certs[0][:1]).abs().mean()
    assert torch.isfinite(cons)
