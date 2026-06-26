"""Tests for VR geometry and per-face stylization."""

import torch

from fav.vr.perspective import (
    make_perspective_warp_map_left,
    make_perspective_warp_map_right,
    make_perspective_warp_map_top,
    make_perspective_warp_map_bottom,
    SENTINEL,
)
from fav.vr.cubemap import make_cube_to_equirectangular_map, PROC_ORDER
from fav.vr.stylize_vr import stylize_faces_over_time
from fav.models.generator import Generator

SMALL_ARCH = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"


def test_perspective_left_shape_and_sentinel():
    hplus, wplus, overlap = 96, 96, 20
    m = make_perspective_warp_map_left(hplus, overlap, wplus)
    assert m.shape == (2, hplus, wplus)
    # Only the rightmost `overlap` columns are filled; the rest stay sentinel.
    assert torch.all(m[:, :, : wplus - overlap] == SENTINEL)
    assert torch.all(m[:, :, wplus - overlap :] != SENTINEL)


def test_all_four_perspective_maps_shapes():
    hplus, wplus, ov = 96, 96, 20
    assert make_perspective_warp_map_left(hplus, ov, wplus).shape == (2, hplus, wplus)
    assert make_perspective_warp_map_right(hplus, ov, wplus).shape == (2, hplus, wplus)
    assert make_perspective_warp_map_top(wplus, ov, hplus).shape == (2, hplus, wplus)
    assert make_perspective_warp_map_bottom(wplus, ov, hplus).shape == (2, hplus, wplus)


def test_perspective_right_fills_left_columns():
    hplus, wplus, ov = 80, 80, 16
    m = make_perspective_warp_map_right(hplus, ov, wplus)
    # The right map fills the leftmost `overlap` columns.
    assert torch.all(m[:, :, :ov] != SENTINEL)
    assert torch.all(m[:, :, ov:] == SENTINEL)


def test_cube_to_equirect_map_shape_and_bounds():
    wplus = hplus = 68
    overlap = 4
    out_h, out_w = 64, 128
    m = make_cube_to_equirectangular_map(wplus, hplus, overlap, overlap, out_w, out_h)
    assert m.shape == (2, out_h, out_w)
    # Resolve to absolute source coords; they must stay within the 6-face strip.
    ys = torch.arange(out_h).view(out_h, 1)
    xs = torch.arange(out_w).view(1, out_w)
    src_y = m[0] + ys
    src_x = m[1] + xs
    strip_w = 6 * wplus
    assert float(src_x.min()) >= -1 and float(src_x.max()) <= strip_w + 1
    assert float(src_y.min()) >= -1 and float(src_y.max()) <= hplus + 1


def test_proc_order_is_permutation_of_faces():
    assert sorted(PROC_ORDER) == [1, 2, 3, 4, 5, 6]


def test_stylize_faces_over_time_smoke():
    model = Generator(SMALL_ARCH).eval()
    T, H, W = 3, 64, 64
    faces_seq = [{f: torch.rand(1, 3, H, W) for f in range(1, 7)} for _ in range(T)]
    flows_seq = [{f: torch.zeros(1, 2, H, W) for f in range(1, 7)} for _ in range(T - 1)]
    certs_seq = [{f: torch.ones(1, 1, H, W) for f in range(1, 7)} for _ in range(T - 1)]
    out = stylize_faces_over_time(model, faces_seq, flows_seq, certs_seq, median_filter_size=0)
    assert len(out) == T
    for t in range(T):
        assert set(out[t].keys()) == set(range(1, 7))
        for f in range(1, 7):
            assert out[t][f].shape == (1, 3, H, W)
            assert float(out[t][f].min()) >= 0 and float(out[t][f].max()) <= 1
