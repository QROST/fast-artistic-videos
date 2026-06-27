"""Tests for the Phase-2 MPS-throughput knobs (PR-C).

All knobs are config-gated; the fp32 default must reproduce the faithful path.
"""

import contextlib

import torch

from fav.config import TrainConfig, ModelConfig
from fav.device import autocast_context, needs_grad_scaler
from fav.models.generator import Generator
from fav.losses.vgg_loss_net import build_vgg16_loss_net
from fav.losses.perceptual import PerceptualCriterion
from fav.preprocess import vgg_preprocess
from fav.data.synthetic import make_shift, make_single_image
from fav.train.loop import train_step, train

SMALL_ARCH = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"


def _setup(**cfg_over):
    cfg = TrainConfig(model=ModelConfig(arch=SMALL_ARCH))
    for k, v in cfg_over.items():
        setattr(cfg, k, v)
    model = Generator(SMALL_ARCH)
    crit = PerceptualCriterion(
        build_vgg16_loss_net(cfg.loss.content_layers, cfg.loss.style_layers),
        cfg.loss.content_layers, cfg.loss.content_weights,
        cfg.loss.style_layers, cfg.loss.style_weights,
    )
    crit.set_style_target(vgg_preprocess(torch.rand(1, 3, 64, 64)))
    return cfg, model, crit


def test_defaults_are_faithful_fp32():
    cfg = TrainConfig()
    assert cfg.precision == "fp32"
    assert cfg.compile_model is False
    assert cfg.grad_checkpoint is False


def test_autocast_fp32_is_noop():
    ctx = autocast_context("cpu", "fp32")
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_bf16_is_context():
    ctx = autocast_context("cpu", "bf16")
    # Either a real autocast or a nullcontext fallback — must be usable.
    with ctx:
        y = torch.randn(2, 2) @ torch.randn(2, 2)
    assert torch.isfinite(y).all()


def test_needs_grad_scaler():
    assert needs_grad_scaler("cuda", "fp16") is True
    assert needs_grad_scaler("cpu", "fp16") is False
    assert needs_grad_scaler("mps", "bf16") is False


def test_train_step_bf16_runs():
    cfg, model, crit = _setup(precision="bf16")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    imgs, flows, certs = make_shift(torch.rand(2, 3, 64, 64), num=1, displ=(4, -3))
    m = train_step(model, crit, opt, ("shift", imgs, flows, certs), cfg, device="cpu")
    assert torch.isfinite(torch.tensor(m["loss"]))


def test_train_step_grad_checkpoint_runs():
    cfg, model, crit = _setup(grad_checkpoint=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    imgs, flows, certs = make_shift(torch.rand(2, 3, 64, 64), num=2, displ=(3, 2))
    m = train_step(model, crit, opt, ("shift", imgs, flows, certs), cfg, device="cpu")
    assert torch.isfinite(torch.tensor(m["loss"]))
    gnorm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(gnorm) and gnorm > 0


def test_fp32_path_unchanged_loss_decreases():
    # Regression: the default fp32 path still trains (loss decreases), unaffected
    # by the new knobs being present.
    cfg, model, crit = _setup()
    torch.manual_seed(7)
    fixed = torch.rand(2, 3, 64, 64)

    def data_fn(it):
        if it % 2 == 0:
            return ("shift", *make_shift(fixed, num=1, displ=(4, -3)))
        return ("single_image", *make_single_image(fixed))

    hist = train(model, crit, data_fn, cfg, device="cpu", max_iters=16)
    assert sum(h["loss"] for h in hist[-3:]) / 3 < sum(h["loss"] for h in hist[:3]) / 3


def test_bench_runs():
    from fav.bench import benchmark

    cfg = TrainConfig(model=ModelConfig(arch=SMALL_ARCH))
    cfg.batch_size = 1
    r = benchmark(cfg, device="cpu", size=64, iters=2, warmup=1)
    assert r["iter_per_s"] > 0 and r["ms_per_iter"] > 0
    assert r["precision"] == "fp32"
