"""Faithfulness tests for fav.preprocess (ports preprocess.lua)."""

import torch

from fav import preprocess as P


def test_vgg_roundtrip_identity():
    img = torch.rand(2, 3, 8, 8)
    out = P.vgg_deprocess(P.vgg_preprocess(img))
    assert torch.allclose(img, out, atol=1e-5)


def test_vgg_hand_pixel():
    # Pure red RGB (1, 0, 0). After RGB->BGR it becomes (B=0, G=0, R=1), then
    # *255 and minus the Caffe BGR mean {103.939, 116.779, 123.68}.
    img = torch.zeros(1, 3, 1, 1)
    img[0, 0, 0, 0] = 1.0  # R channel
    out = P.vgg_preprocess(img)[0, :, 0, 0]
    expected = torch.tensor([0.0 - 103.939, 0.0 - 116.779, 255.0 - 123.68])
    assert torch.allclose(out, expected, atol=1e-4)


def test_vgg_channel_reorder_is_bgr():
    # A green pixel (0,1,0) is unchanged by the R<->B swap, so only the middle
    # (G) channel carries 255 before mean subtraction.
    img = torch.zeros(1, 3, 2, 2)
    img[0, 1] = 1.0
    out = P.vgg_preprocess(img)
    mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
    restored = out + mean
    assert torch.allclose(restored[0, 1], torch.full((2, 2), 255.0), atol=1e-3)
    assert torch.allclose(restored[0, 0], torch.zeros(2, 2), atol=1e-3)
    assert torch.allclose(restored[0, 2], torch.zeros(2, 2), atol=1e-3)


def test_resnet_roundtrip_and_stats():
    img = torch.rand(3, 3, 4, 4)
    out = P.resnet_deprocess(P.resnet_preprocess(img))
    assert torch.allclose(img, out, atol=1e-5)
    # Hand check one channel: (x - mean)/std.
    x = P.resnet_preprocess(img)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(x, (img - mean) / std, atol=1e-5)


def test_registry_dispatch():
    pre, de = P.get_methods("vgg")
    assert pre is P.vgg_preprocess and de is P.vgg_deprocess
    img = torch.rand(1, 3, 5, 5)
    assert torch.allclose(P.deprocess(P.preprocess(img, "vgg"), "vgg"), img, atol=1e-5)
