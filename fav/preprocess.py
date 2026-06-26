"""Image (de)preprocessing, a faithful port of ``fast_artistic_video/preprocess.lua``.

Two methods are provided, selected by name exactly like the Lua
``preprocess[opt.preprocessing]`` table:

``vgg`` (default, used by the original FAV models and the VGG-16 loss network)
    RGB[0,1] -> reorder channels to BGR -> scale to [0,255] -> subtract the
    Caffe ImageNet BGR mean ``{103.939, 116.779, 123.68}``. ``deprocess`` is the
    exact inverse.

``resnet``
    RGB[0,1] -> subtract per-channel mean ``{0.485, 0.456, 0.406}`` -> divide by
    std ``{0.229, 0.224, 0.225}``.

All functions operate on float tensors shaped ``(N, 3, H, W)`` and are
backend-agnostic (the mean/std constants are moved to the input's device/dtype).
"""

from __future__ import annotations

from typing import Callable

import torch

# Caffe ImageNet mean in **BGR** order, matching preprocess.lua (values are in
# the [0,255] range, i.e. applied after the *255 scaling).
_VGG_BGR_MEAN = (103.939, 116.779, 123.68)
_RESNET_MEAN = (0.485, 0.456, 0.406)
_RESNET_STD = (0.229, 0.224, 0.225)

# RGB <-> BGR channel permutation (0-indexed; Lua used 1-indexed [3,2,1]).
_RGB_TO_BGR = (2, 1, 0)


def _const(values, ref: torch.Tensor) -> torch.Tensor:
    """A ``(1, 3, 1, 1)`` constant on the same device/dtype as ``ref``."""
    return torch.tensor(values, dtype=ref.dtype, device=ref.device).view(1, 3, 1, 1)


def vgg_preprocess(img: torch.Tensor) -> torch.Tensor:
    """RGB[0,1] (N,3,H,W) -> VGG-caffe BGR, mean-subtracted, [0,255] range."""
    mean = _const(_VGG_BGR_MEAN, img)
    img = img.index_select(1, torch.tensor(_RGB_TO_BGR, device=img.device))
    return img.mul(255.0).sub(mean)


def vgg_deprocess(img: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`vgg_preprocess`, returning RGB[0,1]."""
    mean = _const(_VGG_BGR_MEAN, img)
    img = img.add(mean).div(255.0)
    # BGR -> RGB is its own inverse permutation.
    return img.index_select(1, torch.tensor(_RGB_TO_BGR, device=img.device))


def resnet_preprocess(img: torch.Tensor) -> torch.Tensor:
    """RGB[0,1] (N,3,H,W) -> (img - mean) / std (channels kept in RGB order)."""
    mean = _const(_RESNET_MEAN, img)
    std = _const(_RESNET_STD, img)
    return img.sub(mean).div(std)


def resnet_deprocess(img: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`resnet_preprocess`."""
    mean = _const(_RESNET_MEAN, img)
    std = _const(_RESNET_STD, img)
    return img.mul(std).add(mean)


# Name -> (preprocess, deprocess), mirroring the Lua ``preprocess`` module table.
_METHODS: dict[str, tuple[Callable, Callable]] = {
    "vgg": (vgg_preprocess, vgg_deprocess),
    "resnet": (resnet_preprocess, resnet_deprocess),
}


def get_methods(name: str = "vgg") -> tuple[Callable, Callable]:
    """Return ``(preprocess_fn, deprocess_fn)`` for ``name`` ('vgg' | 'resnet')."""
    if name not in _METHODS:
        raise ValueError(f"invalid preprocessing '{name}'; must be one of {sorted(_METHODS)}")
    return _METHODS[name]


def preprocess(img: torch.Tensor, method: str = "vgg") -> torch.Tensor:
    return get_methods(method)[0](img)


def deprocess(img: torch.Tensor, method: str = "vgg") -> torch.Tensor:
    return get_methods(method)[1](img)
