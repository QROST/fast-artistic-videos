"""Tests for optional auxiliary perceptual losses (LPIPS/DINO), PR-E."""

import importlib.util

import pytest
import torch

from fav.config import TrainConfig, ModelConfig, LossConfig
from fav.losses.aux import build_aux_terms
from fav.models.generator import Generator
from fav.losses.vgg_loss_net import build_vgg16_loss_net
from fav.losses.perceptual import PerceptualCriterion
from fav.preprocess import vgg_preprocess
from fav.data.synthetic import make_shift, make_single_image
from fav.train.loop import train_step, train

SMALL_ARCH = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"


def _setup(**loss_over):
    cfg = TrainConfig(model=ModelConfig(arch=SMALL_ARCH))
    for k, v in loss_over.items():
        setattr(cfg.loss, k, v)
    model = Generator(SMALL_ARCH)
    crit = PerceptualCriterion(
        build_vgg16_loss_net(cfg.loss.content_layers, cfg.loss.style_layers),
        cfg.loss.content_layers, cfg.loss.content_weights,
        cfg.loss.style_layers, cfg.loss.style_weights,
    )
    crit.set_style_target(vgg_preprocess(torch.rand(1, 3, 64, 64)))
    return cfg, model, crit


def test_defaults_have_no_aux_terms():
    assert LossConfig().lpips_weight == 0.0
    assert LossConfig().dino_weight == 0.0
    assert build_aux_terms(LossConfig(), "cpu") == []


def test_lpips_missing_raises_informative():
    if importlib.util.find_spec("lpips") is not None:
        pytest.skip("lpips installed; missing-dep path not exercised")
    with pytest.raises(RuntimeError) as e:
        build_aux_terms(LossConfig(lpips_weight=1.0), "cpu")
    assert "lpips" in str(e.value).lower()


def test_aux_term_integrates_into_train_step():
    # Inject a trivial aux term to verify wiring without external deps.
    cfg, model, crit = _setup()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    imgs, flows, certs = make_shift(torch.rand(2, 3, 64, 64), num=1, displ=(4, -3))
    aux = [(2.0, lambda out_rgb, content_rgb: (out_rgb - content_rgb).abs().mean())]
    m = train_step(model, crit, opt, ("shift", imgs, flows, certs), cfg, device="cpu",
                   aux_terms=aux)
    assert m["aux"] > 0
    assert torch.isfinite(torch.tensor(m["loss"]))
    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(gnorm) and gnorm > 0


def test_faithful_default_unchanged_no_aux():
    cfg, model, crit = _setup()
    torch.manual_seed(7)
    fixed = torch.rand(2, 3, 64, 64)

    def data_fn(it):
        if it % 2 == 0:
            return ("shift", *make_shift(fixed, num=1, displ=(4, -3)))
        return ("single_image", *make_single_image(fixed))

    hist = train(model, crit, data_fn, cfg, device="cpu", max_iters=14)
    assert all(h["aux"] == 0.0 for h in hist)  # no aux at the faithful default
    assert sum(h["loss"] for h in hist[-3:]) / 3 < sum(h["loss"] for h in hist[:3]) / 3
