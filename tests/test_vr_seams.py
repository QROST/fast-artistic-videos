"""Tests for VR cross-face seam blending (ports fast_artistic_video_vr.lua)."""

import torch

from fav.vr.seams import (
    SeamGeometry,
    make_gradient_mask_w_inc,
    make_gradient_mask_w_dec,
    make_gradient_mask_h_inc,
    make_gradient_mask_h_dec,
    rotate90,
    rotate_minus90,
    rotate180,
    make_border_prior,
    make_border_cert,
    blend_border,
    reblend_all_faces,
    seam_prior_and_cert,
)
from fav.vr.cubemap import PROC_ORDER
from fav.vr.stylize_vr import stylize_faces_seams_over_time
from fav.models.generator import Generator

SMALL_ARCH = "c9s1-8,d16,d32,R32,R32,U2,c3s1-16,U2,c9s1-3"


# --- gradient masks --------------------------------------------------------

def test_gradient_mask_ramps():
    w = 5
    inc = make_gradient_mask_w_inc(3, w)
    dec = make_gradient_mask_w_dec(3, w)
    assert inc.shape == (1, 1, 3, w)
    # inc = (1..w)/(w+1) along width; dec is its reverse.
    torch.testing.assert_close(inc[0, 0, 0], torch.arange(1, w + 1).float() / (w + 1))
    torch.testing.assert_close(dec[0, 0, 0], torch.arange(w, 0, -1).float() / (w + 1))
    # h variants ramp along height and are constant across width.
    hinc = make_gradient_mask_h_inc(4, 3)
    assert hinc.shape == (1, 1, 4, 3)
    torch.testing.assert_close(hinc[0, 0, :, 0], torch.arange(1, 5).float() / 5)
    assert torch.allclose(hinc[0, 0, 0], hinc[0, 0, 0, 0].expand(3))
    hdec = make_gradient_mask_h_dec(4, 3)
    torch.testing.assert_close(hdec[0, 0, :, 0], torch.arange(4, 0, -1).float() / 5)


# --- rotations -------------------------------------------------------------

def test_rotations_match_lua_convention():
    # Distinct values so each rotation is uniquely checkable.
    t = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    # rotate90 == transpose(H,W) then flip along (new) H.
    torch.testing.assert_close(rotate90(t), t.transpose(2, 3).flip(2))
    torch.testing.assert_close(rotate_minus90(t), t.transpose(2, 3).flip(3))
    torch.testing.assert_close(rotate180(t), t.flip(2).flip(3))
    # rotate90 then rotate_minus90 is identity.
    torch.testing.assert_close(rotate_minus90(rotate90(t)), t)
    # rotate180 twice is identity.
    torch.testing.assert_close(rotate180(rotate180(t)), t)


# --- geometry --------------------------------------------------------------

def _geom(h=64, w=64, ov=20):
    return SeamGeometry.build(h, w, ov, ov)


def test_geometry_shapes_and_mask_ranges():
    g = _geom()
    assert g.warp_map_left.shape == (1, 2, 64, 64)
    for m in (g.mask_left, g.mask_right, g.mask_top, g.mask_bottom,
              g.mask_all, g.mask_all_div, g.grad_mask_all):
        assert m.shape == (1, 1, 64, 64)
    # masks are in [0,1]; mask_all_div >= 1 everywhere (safe divisor).
    for m in (g.mask_left, g.mask_right, g.mask_all):
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0 + 1e-5
    assert float(g.mask_all_div.min()) >= 1.0
    # left coverage lives on the right columns (warp_map_left writes there).
    assert float(g.mask_left[:, :, :, -1].mean()) > 0.5
    assert float(g.mask_left[:, :, :, 0].mean()) < 1e-4


def test_grad_mask_left_right_high_on_correct_edges():
    g = _geom()
    # grad_mask_left peaks at the left edge, grad_mask_right at the right edge.
    assert float(g.grad_mask_left[:, :, :, 0].mean()) > float(g.grad_mask_left[:, :, :, -1].mean())
    assert float(g.grad_mask_right[:, :, :, -1].mean()) > float(g.grad_mask_right[:, :, :, 0].mean())
    assert torch.all(g.grad_mask_all >= g.grad_mask_left - 1e-6)
    assert torch.all(g.grad_mask_all >= g.grad_mask_right - 1e-6)


# --- border prior ----------------------------------------------------------

def _segments(n, c=3, h=64, w=64, base=0.0):
    return [torch.full((1, c, h, w), base + i + 1.0) for i in range(n)]


def test_border_prior_front_is_empty():
    g = _geom()
    border, gm = make_border_prior(g, [], 0)
    assert torch.count_nonzero(border) == 0
    assert gm is None


def test_border_prior_pulls_neighbor_content_into_overlap():
    g = _geom()
    segs = _segments(1)  # segments[0] = front, constant 1.0
    border, gm = make_border_prior(g, segs, 1)  # bottom: warp front via left map
    assert gm is g.grad_mask_right
    # The border must carry the neighbour's content where the left-map covers
    # (the right overlap columns), and be ~0 outside that coverage.
    covered = g.mask_left > 0.5
    assert float(border[covered.expand_as(border)].mean()) > 0.5
    outside = g.mask_left < 1e-4
    assert float(border[outside.expand_as(border)].abs().mean()) < 1e-4


def test_border_prior_corner_face_combines_four_neighbors():
    g = _geom()
    segs = _segments(4)  # front,bottom,top,back present
    border, gm = make_border_prior(g, segs, 4)
    assert gm is g.grad_mask_all
    # All four overlap strips should carry content (non-zero) for a corner face.
    assert float(border[(g.mask_all > 0.5).expand_as(border)].mean()) > 0.1


def test_border_prior_modes_select_expected_gradmasks():
    g = _geom()
    segs = _segments(4)
    assert make_border_prior(g, segs, 2)[1] is g.grad_mask_left
    assert make_border_prior(g, segs, 3)[1] is g.grad_mask_left_right
    assert make_border_prior(g, segs, 5)[1] is g.grad_mask_all


# --- border certainty ------------------------------------------------------

def test_border_cert_strips_per_mode():
    g = _geom()
    assert torch.count_nonzero(make_border_cert(g, 0)) == 0  # front: nothing certain
    c1 = make_border_cert(g, 1)
    torch.testing.assert_close(c1, g.mask_left)
    c3 = make_border_cert(g, 3)
    torch.testing.assert_close(c3, torch.maximum(g.mask_left, g.mask_right))
    c4 = make_border_cert(g, 4)
    expect = torch.maximum(torch.maximum(g.mask_left, g.mask_right),
                           torch.maximum(g.mask_top, g.mask_bottom))
    torch.testing.assert_close(c4, expect)


# --- blend -----------------------------------------------------------------

def test_blend_front_passthrough():
    g = _geom()
    temporal = torch.rand(1, 3, 64, 64)
    border = torch.rand(1, 3, 64, 64)
    cert = torch.ones(1, 1, 64, 64)
    out = blend_border(g, 0, temporal, border, cert)
    torch.testing.assert_close(out, temporal)


def test_blend_uses_border_where_temporal_uncertain():
    g = _geom()
    temporal = torch.zeros(1, 3, 64, 64)
    border = torch.ones(1, 3, 64, 64)
    # Fully occluded (cert=0): over the coverage strip the blend weight -> mask,
    # so the border (1.0) wins there; outside the overlap temporal (0.0) stays.
    cert = torch.zeros(1, 1, 64, 64)
    out = blend_border(g, 1, temporal, border, cert)
    covered = g.mask_left > 0.5
    assert float(out[covered.expand_as(out)].mean()) > 0.5
    outside = g.mask_left < 1e-4
    assert float(out[outside.expand_as(out)].abs().mean()) < 1e-4


def test_blend_keeps_temporal_in_interior_when_certain():
    g = _geom()
    temporal = torch.full((1, 3, 64, 64), 0.3)
    border = torch.full((1, 3, 64, 64), 0.9)
    cert = torch.ones(1, 1, 64, 64)
    out = blend_border(g, 1, temporal, border, cert)
    # Deep interior (far from any overlap) keeps the temporal prior exactly.
    torch.testing.assert_close(out[:, :, 30:34, 5:9], temporal[:, :, 30:34, 5:9])


# --- orchestrator ----------------------------------------------------------

def test_seam_prior_and_cert_first_timestep_is_raw_border():
    g = _geom()
    segs = _segments(1)
    # occlusions_min_filter=1 disables erosion so cert == the raw border cert.
    prior, cert = seam_prior_and_cert(g, segs, 1, None, None, blend=False,
                                      occlusions_min_filter=1)
    border, _ = make_border_prior(g, segs, 1)
    torch.testing.assert_close(prior, border)
    torch.testing.assert_close(cert, make_border_cert(g, 1))


def test_seam_prior_and_cert_blends_and_marks_border_certain():
    g = _geom()
    segs = _segments(1)
    temporal = torch.zeros(1, 3, 64, 64)
    occ = torch.zeros(1, 1, 64, 64)  # everything occluded temporally
    prior, cert = seam_prior_and_cert(g, segs, 1, temporal, occ, blend=True,
                                      occlusions_min_filter=1)
    # Border strip is certain even though the temporal occlusion was all-zero.
    assert float((cert * g.mask_left).sum()) > 0
    # cert = combined(occlusion max border); min_filter=1 leaves it unchanged.
    torch.testing.assert_close(cert, torch.maximum(occ, make_border_cert(g, 1)))


def test_seam_prior_and_cert_combines_before_min_filter():
    # The min-filter must erode the COMBINED cert (occlusion max border), so a
    # covered border pixel adjacent to occluded interior gets eroded -- matching
    # the legacy core (filter applied once to the combined cert).
    g = _geom()
    segs = _segments(1)
    temporal = torch.zeros(1, 3, 64, 64)
    occ = torch.zeros(1, 1, 64, 64)
    _, cert7 = seam_prior_and_cert(g, segs, 1, temporal, occ, blend=True,
                                   occlusions_min_filter=7)
    _, cert1 = seam_prior_and_cert(g, segs, 1, temporal, occ, blend=True,
                                   occlusions_min_filter=1)
    # Erosion (r=7) shrinks the certain region vs the unfiltered combined cert.
    assert float(cert7.sum()) < float(cert1.sum())


def test_reblend_all_faces_smooths_with_neighbors():
    g = _geom()
    # 6 distinct constant faces; the re-blend pulls neighbour content into each
    # face's grad_mask_all overlap while leaving the deep interior untouched.
    segs = [torch.full((1, 3, 64, 64), float(i + 1)) for i in range(6)]
    out = reblend_all_faces(g, segs)
    assert len(out) == 6
    for p in range(6):
        assert out[p].shape == (1, 3, 64, 64)
        # Deep interior (grad_mask_all == 0 there) keeps the original face value.
        interior = g.grad_mask_all[:, :, 28:36, 28:36] < 1e-6
        assert torch.all(interior)
        torch.testing.assert_close(out[p][:, :, 28:36, 28:36], segs[p][:, :, 28:36, 28:36])
    # The front face (p=0) overlap should now carry some neighbour (!=1.0) content.
    front_overlap = out[0][(g.grad_mask_all > 0.3).expand_as(out[0])]
    assert float(front_overlap.std()) > 0


# --- end-to-end loop -------------------------------------------------------

def _make_faces_seq(T=2, h=48, w=48):
    torch.manual_seed(0)
    faces_seq = [{f: torch.rand(1, 3, h, w) for f in PROC_ORDER} for _ in range(T)]
    flows_seq = [{f: torch.zeros(1, 2, h, w) for f in PROC_ORDER} for _ in range(T - 1)]
    certs_seq = [{f: torch.ones(1, 1, h, w) for f in PROC_ORDER} for _ in range(T - 1)]
    return faces_seq, flows_seq, certs_seq


def test_seam_loop_runs_and_shapes():
    torch.manual_seed(0)
    model = Generator(SMALL_ARCH).eval()
    faces_seq, flows_seq, certs_seq = _make_faces_seq(T=2, h=48, w=48)
    out = stylize_faces_seams_over_time(
        model, faces_seq, flows_seq, certs_seq, overlap_w=20, overlap_h=20,
        median_filter_size=1,
    )
    assert len(out) == 2
    for t in range(2):
        assert set(out[t].keys()) == set(PROC_ORDER)
        for f in PROC_ORDER:
            assert out[t][f].shape == (1, 3, 48, 48)
            assert float(out[t][f].min()) >= 0 and float(out[t][f].max()) <= 1
