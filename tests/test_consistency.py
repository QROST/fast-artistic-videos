"""Tests for the occlusion consistency checker and filters."""

import torch

from fav.occlusion.consistency import check_consistency, compute_corners
from fav.occlusion.filters import min_filter, median_filter


def test_pure_translation_interior_reliable_border_occluded():
    # Forward flow shifts right by k; consistent backward flow shifts left by k.
    h, w, k = 12, 16, 3
    flow1 = torch.zeros(2, h, w)
    flow2 = torch.zeros(2, h, w)
    flow1[0] = k       # u = +k
    flow2[0] = -k      # u = -k (perfectly consistent)
    rel = check_consistency(flow1, flow2)
    # Columns whose sample lands off-grid (x + k + 1 >= w) are occluded, plus the
    # bottom row (no 2x2 neighborhood) — faithful to the C++ boundary rule.
    interior = rel[: h - 1, : w - 1 - k]
    assert torch.all(interior == 255.0)
    assert torch.all(rel[:, w - 1 - k :] == 0.0)  # right border occluded
    assert torch.all(rel[h - 1, :] == 0.0)        # bottom row occluded


def test_zero_flow_interior_reliable():
    # With zero flow the interior is reliable; the last row/column cannot form a
    # 2x2 sampling neighborhood and are occluded (matches consistencyChecker.cpp).
    rel = check_consistency(torch.zeros(2, 8, 8), torch.zeros(2, 8, 8))
    assert torch.all(rel[:7, :7] == 255.0)
    assert torch.all(rel[7, :] == 0.0)
    assert torch.all(rel[:, 7] == 0.0)


def test_injected_inconsistency_marks_occluded():
    h, w = 10, 10
    flow1 = torch.zeros(2, h, w)
    flow2 = torch.zeros(2, h, w)
    # A patch where backward flow disagrees badly with forward (round trip != 0).
    flow2[0, 3:6, 3:6] = 20.0
    rel = check_consistency(flow1, flow2)
    assert torch.all(rel[3:6, 3:6] == 0.0)
    # Far-away pixels remain reliable.
    assert rel[0, 0] == 255.0


def test_motion_boundary_detected():
    # A sharp step in the flow field creates a large gradient -> boundary kept
    # reliable (255), but we can detect it by using a distinct value.
    h, w = 8, 12
    flow1 = torch.zeros(2, h, w)
    flow1[0, :, w // 2 :] = 5.0  # step in u
    flow2 = torch.zeros(2, h, w)
    flow2[0, :, w // 2 :] = -5.0
    rel = check_consistency(flow1, flow2, motion_boundary_value=128.0)
    # Some column near the step should be flagged as a boundary (value 128).
    assert (rel == 128.0).any()


def test_compute_corners_range_and_flat_image():
    # A flat image has no corners -> all zeros after normalization.
    flat = torch.ones(3, 16, 16)
    corners = compute_corners(flat)
    assert corners.shape == (16, 16)
    assert torch.all(corners == 0.0)
    # A textured image yields values in [0,1].
    torch.manual_seed(0)
    tex = torch.rand(3, 16, 16)
    c = compute_corners(tex)
    assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0 + 1e-6


def test_structure_term_suppresses_false_positive():
    # A borderline-inconsistent flat region: with structure suppression it can
    # stay reliable, without it the threshold is tighter.
    h, w = 10, 10
    flow1 = torch.zeros(2, h, w)
    flow2 = torch.zeros(2, h, w)
    flow2[0] = 0.9  # small round-trip error everywhere
    without = check_consistency(flow1, flow2, structure=None)
    structure = torch.zeros(h, w)  # fully flat -> max suppression
    with_struct = check_consistency(flow1, flow2, structure=structure)
    # Structure suppression should not mark *more* pixels occluded than without.
    assert (with_struct == 0).sum() <= (without == 0).sum()


def _loop_reference(flow1, flow2):
    """Literal double-loop port of checkConsistency (no structure term)."""
    import math

    _, h, w = flow1.shape
    rel = torch.full((h, w), 255.0)
    for ay in range(h):
        for ax in range(w):
            bx = ax + float(flow1[0, ay, ax])
            by = ay + float(flow1[1, ay, ax])
            x1 = math.floor(bx)
            y1 = math.floor(by)
            x2, y2 = x1 + 1, y1 + 1
            if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
                rel[ay, ax] = 0.0
                continue
            ax_ = bx - x1
            ay_ = by - y1

            def samp(ch):
                a = (1 - ax_) * float(flow2[ch, y1, x1]) + ax_ * float(flow2[ch, y1, x2])
                b = (1 - ax_) * float(flow2[ch, y2, x1]) + ax_ * float(flow2[ch, y2, x2])
                return (1 - ay_) * a + ay_ * b

            u = samp(0)
            v = samp(1)
            cx = bx + u
            cy = by + v
            u2 = float(flow1[0, ay, ax])
            v2 = float(flow1[1, ay, ax])
            if ((cx - ax) ** 2 + (cy - ay) ** 2) >= 0.01 * (u2 * u2 + v2 * v2 + u * u + v * v) + 0.5:
                rel[ay, ax] = 0.0
    return rel


def test_vectorized_matches_loop_on_tile():
    torch.manual_seed(7)
    # Small flows so most pixels stay in-bounds and consistent-ish.
    flow1 = (torch.rand(2, 9, 11) - 0.5) * 2.0
    flow2 = (torch.rand(2, 9, 11) - 0.5) * 2.0
    vec = check_consistency(flow1, flow2)
    # Compare only the occlusion decision (ignore the boundary branch, which the
    # reference omits): both should agree on which pixels are 0.
    ref = _loop_reference(flow1, flow2)
    assert torch.equal((vec == 0), (ref == 0))


def test_min_filter_erodes_reliable_region():
    cert = torch.ones(1, 1, 11, 11)
    cert[0, 0, 5, 5] = 0.0  # single occluded pixel
    eroded = min_filter(cert, radius=7)
    # A 7-wide erosion grows the hole to a 7x7 block of zeros.
    assert eroded[0, 0, 5, 5] == 0.0
    assert float(eroded.sum()) < float(cert.sum())
    assert eroded.shape == cert.shape


def test_median_filter_removes_salt_pepper():
    img = torch.zeros(1, 1, 9, 9)
    img[0, 0, 4, 4] = 1.0  # isolated spike
    out = median_filter(img, radius=3)
    assert out[0, 0, 4, 4] == 0.0
    assert out.shape == img.shape
