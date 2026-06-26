"""Read/write the legacy on-disk flow (``.flo``) and occlusion (``.pgm``) assets.

This keeps the modern pipeline byte-compatible with files produced by the
original tooling (``flowFileLoader.lua``, ``make_video_dataset.read_flow``, the
``consistencyChecker`` ``.pgm`` output) so existing datasets and stylize
pipelines keep working.

Flow convention
---------------
The Middlebury ``.flo`` format stores, per pixel, the pair ``(u, v)`` where
``u`` is the horizontal (x) displacement and ``v`` the vertical (y) displacement.
The warp sampler used throughout this package consumes flow as ``(dy, dx)``
(channel 0 = dy = v, channel 1 = dx = u) — this matches the channel swap done by
``DataLoader_video_real.lua`` and the order written directly by the synthetic
``shift`` source. Use :func:`uv_to_dydx` / :func:`dydx_to_uv` to convert.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Middlebury .flo sanity tag (float32 202021.25).
FLO_MAGIC = 202021.25


def read_flo(path: str | Path) -> torch.Tensor:
    """Read a Middlebury ``.flo`` file as a ``(2, H, W)`` float32 tensor.

    Channel order is ``(u, v)`` = ``(dx, dy)`` (the native ``.flo`` order).
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic.size == 0 or not np.isclose(float(magic[0]), FLO_MAGIC):
            raise ValueError(f"{path}: bad .flo magic number {magic!r} (expected {FLO_MAGIC})")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=2 * w * h)
    if data.size != 2 * w * h:
        raise ValueError(f"{path}: truncated .flo (got {data.size}, expected {2 * w * h})")
    # File layout is row-major, interleaved (u, v) per pixel -> (H, W, 2).
    flow_hw2 = data.reshape(h, w, 2)
    # -> (2, H, W) with channel 0 = u, channel 1 = v.
    return torch.from_numpy(np.ascontiguousarray(flow_hw2.transpose(2, 0, 1)))


def write_flo(path: str | Path, flow_uv: torch.Tensor) -> None:
    """Write a ``(2, H, W)`` ``(u, v)`` flow tensor to a Middlebury ``.flo`` file."""
    flow_uv = _as_2hw(flow_uv)
    _, h, w = flow_uv.shape
    arr = flow_uv.detach().cpu().to(torch.float32).numpy()
    # (2, H, W) -> (H, W, 2) interleaved.
    interleaved = np.ascontiguousarray(arr.transpose(1, 2, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.array([FLO_MAGIC], np.float32).tofile(f)
        np.array([w], np.int32).tofile(f)
        np.array([h], np.int32).tofile(f)
        interleaved.astype(np.float32).tofile(f)


def uv_to_dydx(flow_uv: torch.Tensor) -> torch.Tensor:
    """Convert ``(2,H,W)`` or ``(N,2,H,W)`` ``(u,v)`` flow to ``(dy,dx)``.

    This reproduces the channel swap in ``DataLoader_video_real.lua``:
    ``dy = v`` (the vertical displacement) goes to channel 0, ``dx = u`` to 1.
    """
    u, v = _split_uv(flow_uv)
    return torch.cat([v, u], dim=-3)


def dydx_to_uv(flow_dydx: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`uv_to_dydx`: ``(dy,dx)`` -> native ``(u,v)``."""
    dy, dx = _split_uv(flow_dydx)
    return torch.cat([dx, dy], dim=-3)


def _split_uv(flow: torch.Tensor):
    if flow.dim() not in (3, 4) or flow.shape[-3] != 2:
        raise ValueError(f"expected (...,2,H,W) flow, got shape {tuple(flow.shape)}")
    c0 = flow.narrow(-3, 0, 1)
    c1 = flow.narrow(-3, 1, 1)
    return c0, c1


def _as_2hw(flow: torch.Tensor) -> torch.Tensor:
    if flow.dim() == 4 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.dim() != 3 or flow.shape[0] != 2:
        raise ValueError(f"expected (2,H,W) flow, got shape {tuple(flow.shape)}")
    return flow


# --- Occlusion/reliability map I/O (binary PGM, P5) ------------------------- #


def read_pgm(path: str | Path) -> torch.Tensor:
    """Read a binary (P5) PGM grayscale map as a ``(H, W)`` uint8 tensor."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = _read_pgm_token(f)
        if magic != b"P5":
            raise ValueError(f"{path}: unsupported PGM magic {magic!r} (only binary P5)")
        width = int(_read_pgm_token(f))
        height = int(_read_pgm_token(f))
        maxval = int(_read_pgm_token(f))
        if maxval > 255:
            raise ValueError(f"{path}: 16-bit PGM not supported (maxval={maxval})")
        # The maxval token read above already consumed the single whitespace byte
        # that separates the header from the raster.
        raster = np.frombuffer(f.read(width * height), dtype=np.uint8)
    if raster.size != width * height:
        raise ValueError(f"{path}: truncated PGM raster")
    return torch.from_numpy(raster.reshape(height, width).copy())


def write_pgm(path: str | Path, gray: torch.Tensor) -> None:
    """Write a ``(H, W)`` tensor (0-255) to a binary (P5) PGM file."""
    if gray.dim() == 3 and gray.shape[0] == 1:
        gray = gray[0]
    if gray.dim() != 2:
        raise ValueError(f"expected (H,W) map, got shape {tuple(gray.shape)}")
    arr = gray.detach().cpu().round().clamp(0, 255).to(torch.uint8).numpy()
    h, w = arr.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def _read_pgm_token(f) -> bytes:
    """Read one whitespace-delimited token from a PGM header, skipping comments."""
    token = b""
    while True:
        ch = f.read(1)
        if ch == b"":
            break
        if ch == b"#":  # comment to end of line
            f.readline()
            continue
        if ch.isspace():
            if token:
                break
            continue
        token += ch
    return token
