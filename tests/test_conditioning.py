"""Tests for the Phase-3a diffusion reuse-bridge conditioning."""

import torch

from fav.diffusion.conditioning import (
    build_conditioning,
    first_frame_conditioning,
    flow_to_rgb,
    sobel_edges,
    stack_controls,
)
from fav.data.synthetic import make_shift
from fav.warp.grid_sample import warp


def test_build_conditioning_shapes_and_ranges():
    n, h, w = 2, 48, 64
    content = torch.rand(n, 3, h, w)
    prev = torch.rand(n, 3, h, w)
    flow = torch.zeros(n, 2, h, w)
    cert = torch.ones(n, 1, h, w)
    b = build_conditioning(content, prev, flow, cert)
    assert b.content.shape == (n, 3, h, w)
    assert b.warped_prev.shape == (n, 3, h, w)
    assert b.warped_prev_masked.shape == (n, 3, h, w)
    assert b.cert.shape == (n, 1, h, w)
    assert b.flow_image.shape == (n, 3, h, w)
    assert b.structure.shape == (n, 1, h, w)
    for t in (b.flow_image, b.structure, b.cert):
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0 + 1e-5


def test_warped_prev_uses_validated_warp():
    content = torch.rand(1, 3, 40, 40)
    prev = torch.rand(1, 3, 40, 40)
    flow = torch.zeros(1, 2, 40, 40)
    flow[:, 0] = 2.0  # dy
    flow[:, 1] = -3.0  # dx
    cert = torch.ones(1, 1, 40, 40)
    b = build_conditioning(content, prev, flow, cert, occlusions_min_filter=1)
    assert torch.allclose(b.warped_prev, warp(prev, flow), atol=1e-6)


def test_conditioning_temporal_consistency_via_synthetic_shift():
    # Cross-check through the bridge: with a known shift, the warped previous
    # frame matches the current frame in the reliable region (ties Phase-3a back
    # to the Phase-1 warp/flow convention).
    torch.manual_seed(0)
    img = torch.rand(1, 3, 80, 80)
    dx, dy = 5, -4
    imgs, flows, certs = make_shift(img, num=1, displ=(dx, dy))
    b = build_conditioning(imgs[1], imgs[0], flows[0], certs[0], occlusions_min_filter=1)
    margin = max(abs(dx), abs(dy)) + 2
    m = torch.zeros_like(certs[0])
    m[:, :, margin:-margin, margin:-margin] = certs[0][:, :, margin:-margin, margin:-margin]
    diff = ((b.warped_prev - imgs[1]) * m).abs().sum() / m.sum().clamp_min(1)
    assert diff < 1e-3


def test_flow_to_rgb():
    flow = torch.zeros(1, 2, 16, 16)
    img = flow_to_rgb(flow)
    assert img.shape == (1, 3, 16, 16)
    assert torch.allclose(img, torch.zeros_like(img), atol=1e-5)  # zero flow -> black
    flow[:, 1] = 5.0  # rightward motion
    img2 = flow_to_rgb(flow)
    assert float(img2.max()) > 0.0 and float(img2.max()) <= 1.0 + 1e-5


def test_sobel_edges():
    flat = torch.ones(1, 3, 16, 16) * 0.5
    e = sobel_edges(flat)
    assert e.shape == (1, 1, 16, 16)
    assert float(e.max()) < 1e-4  # no edges in a flat image
    step = torch.zeros(1, 3, 16, 16)
    step[:, :, :, 8:] = 1.0  # vertical edge
    es = sobel_edges(step)
    assert float(es.max()) > 0.5  # strong response at the boundary


def test_first_frame_conditioning():
    content = torch.rand(1, 3, 32, 32)
    b = first_frame_conditioning(content)
    assert torch.count_nonzero(b.warped_prev) == 0
    assert torch.count_nonzero(b.cert) == 0
    assert torch.count_nonzero(b.structure) > 0  # structure still derived from content


def test_stack_controls():
    content = torch.rand(1, 3, 24, 24)
    b = first_frame_conditioning(content)
    ctrl = stack_controls(b)  # default warped_prev_masked(3)+cert(1)+structure(1)=5
    assert ctrl.shape == (1, 5, 24, 24)
    ctrl2 = stack_controls(b, which=("flow_image", "content"))
    assert ctrl2.shape == (1, 6, 24, 24)
    try:
        stack_controls(b, which=("nope",))
    except ValueError as e:
        assert "nope" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_device_agnostic_and_dtype():
    content = torch.rand(1, 3, 16, 16)
    b = build_conditioning(content, torch.rand(1, 3, 16, 16),
                           torch.zeros(1, 2, 16, 16), torch.ones(1, 1, 16, 16))
    moved = b.to("cpu")
    assert moved.content.device.type == "cpu"
    assert b.flow_image.dtype == content.dtype
