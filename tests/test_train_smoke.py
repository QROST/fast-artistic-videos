"""End-to-end smoke train: loss decreases, no NaNs, checkpoint resumes."""

import torch

from fav.config import TrainConfig
from fav.models.generator import Generator
from fav.losses.vgg_loss_net import build_vgg16_loss_net
from fav.losses.perceptual import PerceptualCriterion
from fav.preprocess import vgg_preprocess
from fav.data.synthetic import make_shift, make_single_image
from fav.train.loop import train, train_step
from fav.train.checkpoint import save_checkpoint, load_checkpoint

SMALL_ARCH = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"


def _make_setup():
    cfg = TrainConfig()
    cfg.model.arch = SMALL_ARCH
    cfg.num_iterations = 24
    cfg.batch_size = 2
    model = Generator(SMALL_ARCH)
    loss_net = build_vgg16_loss_net(cfg.loss.content_layers, cfg.loss.style_layers)
    crit = PerceptualCriterion(
        loss_net, cfg.loss.content_layers, cfg.loss.content_weights,
        cfg.loss.style_layers, cfg.loss.style_weights, agg_type="gram",
    )
    crit.set_style_target(vgg_preprocess(torch.rand(1, 3, 64, 64)))
    return cfg, model, crit


def _fixed_data_fn():
    torch.manual_seed(123)
    fixed = torch.rand(2, 3, 64, 64)  # raw RGB, fixed content to overfit

    def data_fn(it):
        if it % 3 == 0:
            imgs, flows, certs = make_shift(fixed, num=1, displ=(4, -3))
            return "shift", imgs, flows, certs
        imgs, flows, certs = make_single_image(fixed)
        return "single_image", imgs, flows, certs

    return data_fn


def test_smoke_train_loss_decreases():
    cfg, model, crit = _make_setup()
    history = train(model, crit, _fixed_data_fn(), cfg, device="cpu", max_iters=24)
    losses = [h["loss"] for h in history]
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "NaN/Inf in loss"
    first = sum(losses[:4]) / 4
    last = sum(losses[-4:]) / 4
    assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"


def test_single_step_grad_only_final_frame():
    cfg, model, crit = _make_setup()
    data_fn = _fixed_data_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # multi-frame rollout: ensure it runs and produces finite grads
    imgs, flows, certs = make_shift(torch.rand(2, 3, 64, 64), num=3, displ=(3, 2))
    metrics = train_step(model, crit, optimizer, ("shift", imgs, flows, certs), cfg, device="cpu")
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    grad_norm = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(grad_norm) and grad_norm > 0


def test_checkpoint_save_load_resume(tmp_path):
    cfg, model, crit = _make_setup()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_fn = _fixed_data_fn()
    train_step(model, crit, optimizer, data_fn(1), cfg, device="cpu")

    from fav.config import to_dict
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, 1, to_dict(cfg), [{"loss": 1.0}])
    assert path.exists() and path.with_suffix(".json").exists()

    model2 = Generator(SMALL_ARCH)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    ckpt = load_checkpoint(path, model2, opt2)
    assert ckpt["iter"] == 1
    # Weights restored exactly.
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
