"""Correctness tests for fav.warp.grid_sample — the highest-risk numeric port.

The warp must reproduce the legacy BilinearSamplerBDHW: source position
(y+dy, x+dx) in pixel units, bilinear, zero-padded out of bounds.
"""

import torch

from fav.warp.grid_sample import warp


def _ref_integer_shift(img, dy, dx):
    """Reference: shift content so output[y,x] = img[y+dy, x+dx], zero-filled.

    Mirrors warp semantics for *integer* (dy, dx): the sample location moves by
    (+dy, +dx), so the visible content shifts by (-dy, -dx).
    """
    n, c, h, w = img.shape
    out = torch.zeros_like(img)
    for y in range(h):
        for x in range(w):
            sy, sx = y + dy, x + dx
            if 0 <= sy < h and 0 <= sx < w:
                out[:, :, y, x] = img[:, :, sy, sx]
    return out


def test_integer_shift_matches_reference():
    torch.manual_seed(0)
    img = torch.rand(2, 3, 9, 11)
    for dy, dx in [(0, 0), (1, 0), (0, 2), (-3, 1), (2, -4)]:
        flow = torch.zeros(2, 2, 9, 11)
        flow[:, 0] = dy
        flow[:, 1] = dx
        got = warp(img, flow)
        ref = _ref_integer_shift(img, dy, dx)
        assert torch.allclose(got, ref, atol=1e-5), f"shift ({dy},{dx}) mismatch"


def test_zero_flow_is_identity():
    img = torch.rand(1, 3, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)
    assert torch.allclose(warp(img, flow), img, atol=1e-6)


def test_out_of_bounds_is_zero():
    img = torch.ones(1, 3, 6, 6)
    flow = torch.full((1, 2, 6, 6), 999.0)  # everything points far off-grid
    out = warp(img, flow)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_partial_oob_border_zeroed():
    # Shift the sample location down by 2: the bottom 2 rows read off-grid -> 0.
    img = torch.ones(1, 1, 5, 5)
    flow = torch.zeros(1, 2, 5, 5)
    flow[:, 0] = 2.0  # dy = +2
    out = warp(img, flow)[0, 0]
    assert torch.allclose(out[:3], torch.ones(3, 5), atol=1e-6)
    assert torch.allclose(out[3:], torch.zeros(2, 5), atol=1e-6)


def test_fractional_bilinear():
    # A horizontal ramp; sampling at x+0.5 averages neighbors.
    ramp = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).expand(1, 1, 3, 5).clone()
    flow = torch.zeros(1, 2, 3, 5)
    flow[:, 1] = 0.5  # dx = +0.5
    out = warp(ramp, flow)[0, 0]
    # Interior columns: (x + (x+1)) / 2 = x + 0.5; last column reads partly OOB.
    assert torch.allclose(out[:, 0], torch.full((3,), 0.5), atol=1e-5)
    assert torch.allclose(out[:, 3], torch.full((3,), 3.5), atol=1e-5)


def test_accepts_unbatched():
    img = torch.rand(3, 7, 7)
    flow = torch.zeros(2, 7, 7)
    out = warp(img, flow)
    assert out.shape == img.shape
    assert torch.allclose(out, img, atol=1e-6)
