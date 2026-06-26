"""Convert the legacy ``vgg16.t7`` loss network to a PyTorch state_dict.

The original ``vgg16.t7`` is a plain ``nn.Sequential`` of
``SpatialConvolution`` / ``ReLU`` / ``SpatialMaxPooling`` modules (caffe
``VGG_ILSVRC_16_layers``). We read it with ``torchfile`` (a pure-Python ``.t7``
reader — no Torch7 runtime needed) and copy each conv's weight/bias, in order,
into a :class:`fav.losses.vgg_loss_net.VGG16Features`. The conv weight layout
``(out, in, kH, kW)`` is identical between Torch7 and PyTorch, so the copy is
direct. Input channel order is BGR (caffe), matching ``fav.preprocess.vgg``.

This is a one-off step; the resulting ``.pt`` is the artifact the loss net
loads. The 6 sample *style* models are intentionally not converted.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fav.losses.vgg_loss_net import VGG16Features, _VGG16_SPEC


def _iter_t7_convs(obj):
    """Yield modules from a torchfile-loaded nn.Sequential that look like convs."""
    modules = getattr(obj, "modules", None)
    if modules is None and isinstance(obj, dict):
        modules = obj.get(b"modules") or obj.get("modules")
    if modules is None:
        raise ValueError("could not find .modules in the .t7 object")
    for m in modules:
        weight = getattr(m, "weight", None)
        bias = getattr(m, "bias", None)
        if weight is not None and getattr(weight, "ndim", 0) == 4:
            yield weight, bias


def convert_vgg16_t7(t7_path: str | Path, out_path: str | Path) -> Path:
    """Convert ``t7_path`` -> a ``.pt`` state_dict at ``out_path``.

    Returns the output path. Requires the optional ``torchfile`` dependency.
    """
    try:
        import torchfile  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "converting vgg16.t7 requires the 'torchfile' package "
            "(pip install torchfile). Alternatively supply caffe VGG-16 weights."
        ) from e

    loaded = torchfile.load(str(t7_path))
    t7_convs = list(_iter_t7_convs(loaded))

    net = VGG16Features(max_index=len(_VGG16_SPEC))
    target_convs = net.conv_modules()
    if len(t7_convs) < len(target_convs):
        raise ValueError(
            f".t7 has {len(t7_convs)} conv layers but the loss net needs {len(target_convs)}"
        )

    with torch.no_grad():
        for conv, (w, b) in zip(target_convs, t7_convs):
            wt = torch.as_tensor(w, dtype=torch.float32)
            if wt.shape != conv.weight.shape:
                raise ValueError(f"conv weight shape {tuple(wt.shape)} != {tuple(conv.weight.shape)}")
            conv.weight.copy_(wt)
            if b is not None:
                conv.bias.copy_(torch.as_tensor(b, dtype=torch.float32))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_path)
    return out_path
