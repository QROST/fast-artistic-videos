"""Synthetic motion sources, ported from ``DataLoader_video_fake.lua``.

These fabricate temporally-consistent training tuples from single images with
*known* flow and occlusion (no estimator needed):

* ``shift`` — pan a window across an upscaled canvas (constant flow).
* ``zoom_out`` — progressively wider crops rescaled back up (linear flow).
* ``single_image`` — empty prior, all-occluded → stylize the first frame from
  scratch (the "mixed training" source).

Each returns ``(imgsList, flowList, certList)``: ``imgsList`` has ``num+1``
frames ``(b,3,h,w)``, ``flowList``/``certList`` have ``num`` entries
(``(b,2,h,w)`` flow in ``(dy,dx)`` order, ``(b,1,h,w)`` certainty in ``[0,1]``).
Inputs/outputs are in the chosen (VGG) preprocessed space, like the original.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MAX_DISPL = 16  # legacy: floor(rand*32) - 16 -> [-16, 15]


def _rand_displacement(generator=None) -> tuple[int, int]:
    r = torch.rand(2, generator=generator)
    dx = int(r[0].item() * 2 * _MAX_DISPL) - _MAX_DISPL
    dy = int(r[1].item() * 2 * _MAX_DISPL) - _MAX_DISPL
    return dx, dy


def _zero_border(cert: torch.Tensor, dx: int, dy: int) -> None:
    """Zero the occluded border (matches the legacy index ranges)."""
    _, _, h, w = cert.shape
    if dx > 0:
        cert[:, :, :, max(w - dx - 1, 0):] = 0
    elif dx < 0:
        cert[:, :, :, : max(-dx, 1)] = 0
    if dy > 0:
        cert[:, :, max(h - dy - 1, 0):, :] = 0
    elif dy < 0:
        cert[:, :, : max(-dy, 1), :] = 0


def make_shift(imgs_pre: torch.Tensor, num: int, displ=None, generator=None):
    """Pan a (h,w) window across an upscaled canvas; constant ``(dy,dx)`` flow."""
    b, c, h, w = imgs_pre.shape
    dx, dy = displ if displ is not None else _rand_displacement(generator)
    offs = _MAX_DISPL
    canvas = F.interpolate(
        imgs_pre, size=(h + offs * num, w + offs * num), mode="bilinear", align_corners=False
    )
    imgs_list = []
    for i in range(num + 1):
        ox = max(-dx * (num - i), 0) + max(dx * i, 0)
        oy = max(-dy * (num - i), 0) + max(dy * i, 0)
        imgs_list.append(canvas[:, :, oy : oy + h, ox : ox + w].contiguous())

    flow = torch.zeros(b, 2, h, w, device=imgs_pre.device, dtype=imgs_pre.dtype)
    flow[:, 0] = dy
    flow[:, 1] = dx
    cert = torch.ones(b, 1, h, w, device=imgs_pre.device, dtype=imgs_pre.dtype)
    _zero_border(cert, dx, dy)

    return imgs_list, [flow] * num, [cert] * num


def make_zoom_out(imgs_pre: torch.Tensor, num: int, displ=None, generator=None):
    """Progressively wider crops rescaled to (h,w); linear zoom flow."""
    b, c, h, w = imgs_pre.shape
    dx, dy = displ if displ is not None else _rand_displacement(generator)

    imgs_list = []
    for i in range(num + 1):
        cw = w - abs(dx * (num - i))
        ch = h - abs(dy * (num - i))
        ox = max(-dx * (num - i), 0)
        oy = max(-dy * (num - i), 0)
        crop = imgs_pre[:, :, oy : oy + ch, ox : ox + cw]
        imgs_list.append(
            F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False).contiguous()
        )

    # Linear flow field (legacy grid construction, square training crop).
    lin_y = torch.linspace(-max(-dy, 0), max(dy, 0), w, device=imgs_pre.device)
    lin_x = torch.linspace(-max(-dx, 0), max(dx, 0), h, device=imgs_pre.device)
    grid_dy = lin_y.view(1, w).expand(h, w)
    grid_dx = lin_x.view(h, 1).expand(h, w)
    flow = torch.stack([grid_dy, grid_dx], dim=0).unsqueeze(0).expand(b, 2, h, w).contiguous()
    flow = flow.to(imgs_pre.dtype)

    cert = torch.ones(b, 1, h, w, device=imgs_pre.device, dtype=imgs_pre.dtype)
    _zero_border(cert, dx, dy)

    return imgs_list, [flow] * num, [cert] * num


def make_single_image(imgs_pre: torch.Tensor):
    """Empty prior + all-occluded cert -> stylize the first frame from scratch."""
    b, c, h, w = imgs_pre.shape
    zeros_img = torch.zeros(b, c, h, w, device=imgs_pre.device, dtype=imgs_pre.dtype)
    flow = torch.zeros(b, 2, h, w, device=imgs_pre.device, dtype=imgs_pre.dtype)
    cert = torch.zeros(b, 1, h, w, device=imgs_pre.device, dtype=imgs_pre.dtype)
    return [zeros_img, imgs_pre], [flow], [cert]


class SyntheticSource:
    """Dispatch to the synthetic generators by mode name."""

    MODES = ("shift", "zoom_out", "single_image")

    def __init__(self, generator=None):
        self.generator = generator

    def sample(self, mode: str, imgs_pre: torch.Tensor, num: int):
        if mode == "shift":
            return make_shift(imgs_pre, num, generator=self.generator)
        if mode == "zoom_out":
            return make_zoom_out(imgs_pre, num, generator=self.generator)
        if mode == "single_image":
            return make_single_image(imgs_pre)
        raise ValueError(f"unknown synthetic mode {mode!r}")
