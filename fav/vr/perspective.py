"""Perspective warp maps for cube-face seam consistency (ports vr_helper.lua).

Each function returns a ``(2, H, W)`` **displacement** map (channel 0 = dy,
channel 1 = dx), identical in convention to the temporal flow, so the same
``fav.warp.warp`` samples them. Positions outside the transformed border carry a
large ``SENTINEL`` displacement that lands out of bounds and therefore samples
zero (matching the legacy fill of 99999).

Although the legacy ``for x = width-crop_w+1, width`` loops use a *float*
``width``, the write index ``x - (width-crop_w) + ...`` cancels ``width`` and is
always integer — so the port iterates the integer columns/rows directly while
keeping the float ``x`` for the resize-factor math.
"""

from __future__ import annotations

import torch

SENTINEL = 99999.0


def _resize_width(long_side: int, oversize: float) -> float:
    """The legacy effective 'width'/'height' used by the perspective math."""
    base = long_side / 2.0 / ((2 * oversize + long_side) / long_side)
    max_rf = (base + oversize) / base
    return base - (max_rf - 1) / max_rf * oversize


def make_perspective_warp_map_left(height, crop_w, orig_width, oversize_h=None, oversize_w=None):
    if oversize_h is None:
        oversize_h = crop_w / 2
    if oversize_w is None:
        oversize_w = crop_w / 2
    width = _resize_width(height, oversize_h)
    m = torch.full((2, height, orig_width), SENTINEL)
    mid_y = height / 2.0
    ys = torch.arange(1, height + 1, dtype=torch.float32)  # Lua 1-indexed y
    for k in range(crop_w):
        x = width - crop_w + 1 + k
        idx = orig_width - crop_w + k  # 0-indexed target column
        rf_h = (x + oversize_h) / width
        rf_w = (x + oversize_w) / width
        m[0, :, idx] = (mid_y - ys) * (-1.0 / rf_h + 1.0)
        m[1, :, idx] = (width - x - oversize_w) * (rf_w - 1) / rf_w - orig_width + crop_w
    return m


def make_perspective_warp_map_right(height, crop_w, orig_width, oversize_h=None, oversize_w=None):
    if oversize_h is None:
        oversize_h = crop_w / 2
    if oversize_w is None:
        oversize_w = crop_w / 2
    width = _resize_width(height, oversize_h)
    m = torch.full((2, height, orig_width), SENTINEL)
    mid_y = height / 2.0
    ys = torch.arange(1, height + 1, dtype=torch.float32)
    for k in range(crop_w):
        x = k + 1  # Lua x = 1..crop_w
        idx = k
        rf_h = (width - x + oversize_h) / width
        rf_w = (width - x + oversize_w) / width
        m[0, :, idx] = (mid_y - ys) * (-1.0 / rf_h + 1.0)
        m[1, :, idx] = -(x - oversize_w) * (rf_w - 1) / rf_w + orig_width - crop_w
    return m


def make_perspective_warp_map_top(width, crop_h, orig_height, oversize_w=None, oversize_h=None):
    if oversize_h is None:
        oversize_h = crop_h / 2
    if oversize_w is None:
        oversize_w = crop_h / 2
    height = _resize_width(width, oversize_w)
    m = torch.full((2, orig_height, width), SENTINEL)
    mid_x = width / 2.0
    xs = torch.arange(1, width + 1, dtype=torch.float32)
    for k in range(crop_h):
        y = height - crop_h + 1 + k
        row = orig_height - crop_h + k  # 0-indexed target row
        rf_w = (y + oversize_w) / height
        rf_h = (y + oversize_h) / height
        m[0, row, :] = (height - y - oversize_h) * (rf_h - 1) / rf_h - orig_height + crop_h
        m[1, row, :] = (mid_x - xs) * (-1.0 / rf_w + 1.0)
    return m


def make_perspective_warp_map_bottom(width, crop_h, orig_height, oversize_w=None, oversize_h=None):
    if oversize_h is None:
        oversize_h = crop_h / 2
    if oversize_w is None:
        oversize_w = crop_h / 2
    height = _resize_width(width, oversize_w)
    m = torch.full((2, orig_height, width), SENTINEL)
    mid_x = width / 2.0
    xs = torch.arange(1, width + 1, dtype=torch.float32)
    for k in range(crop_h):
        y = k + 1
        row = k
        rf_w = (height - y + oversize_w) / height
        rf_h = (height - y + oversize_h) / height
        m[0, row, :] = -(y - oversize_h) * (rf_h - 1) / rf_h + orig_height - crop_h
        m[1, row, :] = (mid_x - xs) * (-1.0 / rf_w + 1.0)
    return m
