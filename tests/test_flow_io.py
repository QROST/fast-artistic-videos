"""Round-trip and channel-convention tests for fav.warp.flow_io."""

import torch

from fav.warp import flow_io


def test_flo_roundtrip(tmp_path):
    h, w = 5, 7
    # Distinct, identifiable u and v fields.
    u = torch.arange(h * w, dtype=torch.float32).reshape(h, w)
    v = -torch.arange(h * w, dtype=torch.float32).reshape(h, w) * 0.5
    flow_uv = torch.stack([u, v], dim=0)  # (2,H,W)

    path = tmp_path / "sub" / "flow.flo"
    flow_io.write_flo(path, flow_uv)
    back = flow_io.read_flo(path)

    assert back.shape == (2, h, w)
    assert torch.allclose(back, flow_uv, atol=0)


def test_flo_bad_magic(tmp_path):
    path = tmp_path / "bad.flo"
    path.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 16)
    try:
        flow_io.read_flo(path)
    except ValueError as e:
        assert "magic" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for bad magic")


def test_uv_dydx_swap_roundtrip():
    flow_uv = torch.randn(2, 4, 6)
    dydx = flow_io.uv_to_dydx(flow_uv)
    # dy(channel 0) must equal v (channel 1 of uv); dx must equal u.
    assert torch.equal(dydx[0], flow_uv[1])
    assert torch.equal(dydx[1], flow_uv[0])
    assert torch.equal(flow_io.dydx_to_uv(dydx), flow_uv)


def test_uv_dydx_batched():
    flow_uv = torch.randn(3, 2, 4, 6)
    dydx = flow_io.uv_to_dydx(flow_uv)
    assert dydx.shape == flow_uv.shape
    assert torch.equal(dydx[:, 0], flow_uv[:, 1])
    assert torch.equal(dydx[:, 1], flow_uv[:, 0])


def test_pgm_roundtrip(tmp_path):
    gray = (torch.rand(6, 9) * 255).round()
    path = tmp_path / "reliable.pgm"
    flow_io.write_pgm(path, gray)
    back = flow_io.read_pgm(path)
    assert back.shape == (6, 9)
    assert torch.equal(back.float(), gray.clamp(0, 255))
