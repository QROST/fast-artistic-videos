"""Shape/structure tests for the generator port (models_video.lua)."""

import torch

from fav.models.generator import Generator, build_model, IN_CHANNELS
from fav.models.layers import ResidualBlock, ShaveImage


DEFAULT_ARCH = "c9s1-32,d64,d128,R128,R128,R128,R128,R128,U2,c3s1-64,U2,c9s1-3"


def test_seven_to_three_same_size_default():
    net = Generator(DEFAULT_ARCH).eval()
    for size in (128, 256, 132):  # multiples of 4 incl. the training crop
        x = torch.randn(1, IN_CHANNELS, size, size)
        out = net(x)
        assert out.shape == (1, 3, size, size), f"size {size} -> {tuple(out.shape)}"


def test_reflect_start_pad_measured():
    net = Generator(DEFAULT_ARCH)
    # 5 residual blocks * 4px shrink at 1/4 resolution, upsampled x4 -> 80px
    # full-res shrink -> 40px reflection pad each side.
    assert (net.pad_h, net.pad_w) == (40, 40)


def test_output_range_bounded_by_tanh_constant():
    net = Generator(DEFAULT_ARCH).eval()
    x = torch.randn(1, IN_CHANNELS, 128, 128) * 5
    out = net(x)
    assert out.abs().max() <= 150.0 + 1e-3


def test_rejects_wrong_channel_count():
    net = Generator(DEFAULT_ARCH).eval()
    try:
        net(torch.randn(1, 3, 128, 128))
    except ValueError as e:
        assert "channels" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for wrong channel count")


def test_residual_block_shave_under_reflect_start():
    blk = ResidualBlock(8, "reflect-start", use_instance_norm=True).eval()
    assert isinstance(blk.shortcut, ShaveImage)
    x = torch.randn(1, 8, 20, 20)
    out = blk(x)
    # two unpadded 3x3 convs shrink by 4px; identity is shaved to match.
    assert out.shape == (1, 8, 16, 16)


def test_residual_block_identity_under_zero_padding():
    blk = ResidualBlock(8, "zero", use_instance_norm=True).eval()
    assert blk.shortcut.__class__.__name__ == "Identity"
    x = torch.randn(1, 8, 20, 20)
    assert blk(x).shape == (1, 8, 20, 20)


def test_learned_upsample_arch_runs():
    # The original (pre-improvement) arch uses learned upsampling (u64,u32).
    arch = "c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3"
    net = Generator(arch).eval()
    out = net(torch.randn(1, IN_CHANNELS, 128, 128))
    assert out.shape == (1, 3, 128, 128)


def test_build_model_from_config():
    from fav.config import ModelConfig

    net = build_model(ModelConfig())
    assert isinstance(net, Generator)
    out = net.eval()(torch.randn(1, IN_CHANNELS, 128, 128))
    assert out.shape == (1, 3, 128, 128)
