"""Tests for the Phase-2 generator-backbone primitives (PR-F): GroupNorm + SE."""

import torch

from fav.config import ModelConfig
from fav.models.generator import Generator, build_model, IN_CHANNELS
from fav.models.instance_norm import make_norm
from fav.models.layers import SqueezeExcite

DEFAULT_ARCH = "c9s1-32,d64,d128,R128,R128,R128,R128,R128,U2,c3s1-64,U2,c9s1-3"
SMALL_ARCH = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"


def test_default_norm_is_instance():
    assert ModelConfig().norm == "instance"
    n = make_norm(16)  # legacy default
    assert isinstance(n, torch.nn.InstanceNorm2d)
    assert make_norm(16, norm="instance").__class__ is torch.nn.InstanceNorm2d


def test_make_norm_variants():
    assert isinstance(make_norm(16, norm="batch"), torch.nn.BatchNorm2d)
    g = make_norm(64, norm="group")
    assert isinstance(g, torch.nn.GroupNorm)
    assert 64 % g.num_groups == 0


def test_groupnorm_generator_runs():
    net = Generator(SMALL_ARCH, norm="group").eval()
    out = net(torch.randn(1, IN_CHANNELS, 64, 64))
    assert out.shape == (1, 3, 64, 64)
    # The core actually contains GroupNorm layers.
    assert any(isinstance(m, torch.nn.GroupNorm) for m in net.modules())


def test_default_generator_still_instance_norm():
    net = Generator(SMALL_ARCH).eval()  # default norm
    assert any(isinstance(m, torch.nn.InstanceNorm2d) for m in net.modules())
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in net.modules())


def test_squeeze_excite_preserves_shape_and_modulates():
    se = SqueezeExcite(8).eval()
    x = torch.randn(2, 8, 16, 16)
    y = se(x)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)  # SE actually rescales channels


def test_se_token_in_arch():
    arch = "c9s1-8,d16,d32,R32,E32,R32,U2,c3s1-16,U2,c9s1-3"
    net = Generator(arch).eval()
    out = net(torch.randn(1, IN_CHANNELS, 64, 64))
    assert out.shape == (1, 3, 64, 64)
    assert any(isinstance(m, SqueezeExcite) for m in net.modules())


def test_build_model_threads_norm():
    net = build_model(ModelConfig(arch=SMALL_ARCH, norm="group"))
    assert any(isinstance(m, torch.nn.GroupNorm) for m in net.modules())


def test_default_arch_unchanged_faithful():
    # Default config produces the same faithful instance-norm network as before.
    net = build_model(ModelConfig())
    assert any(isinstance(m, torch.nn.InstanceNorm2d) for m in net.modules())
    assert not any(isinstance(m, (torch.nn.GroupNorm, SqueezeExcite)) for m in net.modules())
