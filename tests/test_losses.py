"""Tests for the perceptual loss stack: gram, VGG taps, perceptual, TV."""

import torch

from fav.losses.gram import gram_matrix
from fav.losses.temporal import tv_penalty, temporal_pixel_loss
from fav.losses.vgg_loss_net import VGG16Features, build_vgg16_loss_net, LAYER_NAMES, _VGG16_SPEC
from fav.losses.perceptual import PerceptualCriterion


def test_gram_hand_computed_and_normalized():
    # x has C=2, H=1, W=2; feats = [[1,2],[3,4]].
    x = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])  # (1,2,1,2)
    g = gram_matrix(x, normalize=True)
    # Raw gram = [[1+4, 3+8],[3+8, 9+16]] = [[5,11],[11,25]], / (C*H*W=4).
    expected = torch.tensor([[[5.0, 11.0], [11.0, 25.0]]]) / 4.0
    assert torch.allclose(g, expected, atol=1e-6)


def test_gram_batch_shapes():
    x = torch.randn(3, 8, 5, 7)
    assert gram_matrix(x).shape == (3, 8, 8)


def test_vgg_index_to_relu_mapping():
    # The built modules at the documented 1-indexed positions must be ReLUs that
    # immediately follow the expected conv depth.
    net = VGG16Features(max_index=len(_VGG16_SPEC))
    for idx, name in LAYER_NAMES.items():
        mod = net.layers[idx - 1]
        assert isinstance(mod, torch.nn.ReLU), f"index {idx} ({name}) is {type(mod)}"
    # Sanity on the conv just before relu1_2 (index 3 -> conv 64->64).
    conv = net.layers[2]
    assert isinstance(conv, torch.nn.Conv2d) and conv.in_channels == 64 and conv.out_channels == 64


def test_vgg_taps_return_requested_layers():
    net = VGG16Features(max_index=23)
    x = torch.randn(1, 3, 32, 32)
    feats = net(x, {4, 9, 16, 23})
    assert set(feats) == {4, 9, 16, 23}
    # relu1_2 keeps full resolution; relu4_3 is downsampled by 3 pools (/8).
    assert feats[4].shape[-2:] == (32, 32)
    assert feats[23].shape[-2:] == (4, 4)


def test_vgg_deterministic():
    net = build_vgg16_loss_net((16,), (4, 9, 16, 23))
    x = torch.randn(1, 3, 32, 32)
    a = net(x, {16})[16]
    b = net(x, {16})[16]
    assert torch.equal(a, b)


def test_perceptual_runs_and_backprops():
    net = build_vgg16_loss_net((16,), (4, 9, 16, 23))
    crit = PerceptualCriterion(net, (16,), (1.0,), (4, 9, 16, 23), (10.0,), agg_type="gram")
    style = torch.randn(1, 3, 48, 48)
    crit.set_style_target(style)

    out = torch.randn(2, 3, 48, 48, requires_grad=True)
    target = torch.randn(2, 3, 48, 48)
    loss = crit(out, target)
    assert loss.dim() == 0 and loss.item() >= 0
    loss.backward()
    assert out.grad is not None and torch.isfinite(out.grad).all()
    # Diagnostics populated.
    assert len(crit.style_losses) == 4 and len(crit.content_losses) == 1
    assert crit.total_style_loss >= 0 and crit.total_content_loss >= 0


def test_perceptual_zero_when_output_equals_target():
    net = build_vgg16_loss_net((16,), (16,))
    crit = PerceptualCriterion(net, (16,), (1.0,), (16,), (1.0,))
    img = torch.randn(1, 3, 32, 32)
    crit.set_style_target(img)
    loss = crit(img.clone(), img.clone())
    # Output identical to both content and style target -> ~0 loss.
    assert loss.item() < 1e-6


def test_tv_penalty_matches_manual_gradient():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 6, 6, requires_grad=True)
    strength = 1e-3
    tv_penalty(x, strength).backward()
    auto_grad = x.grad.clone()

    # Manual scatter of the legacy finite-difference gradient.
    with torch.no_grad():
        xd = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
        yd = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
        g = torch.zeros_like(x)
        g[:, :, :-1, :-1] += xd + yd
        g[:, :, :-1, 1:] -= xd
        g[:, :, 1:, :-1] -= yd
        g *= strength
    assert torch.allclose(auto_grad, g, atol=1e-6)


def test_temporal_pixel_loss_zero_for_equal():
    a = torch.randn(2, 3, 8, 8)
    assert temporal_pixel_loss(a, a.clone()).item() < 1e-8
