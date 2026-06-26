"""Cube-face <-> equirectangular geometry, replacing the Transform360 ffmpeg filter.

* ``make_cube_to_equirectangular_map`` is a faithful vectorized port of
  ``vr_helper.make_cube_to_equirectangular_map`` (layout ``f, l, r, b, u, d`` as a
  horizontal strip). It returns a ``(2, out_h, out_w)`` displacement map sampling
  the cube strip, usable with ``fav.warp.warp``.
* ``equirect_to_cubefaces`` / ``cubefaces_to_equirect`` use ``py360convert`` (lazy
  import) as a self-contained, modern replacement for the legacy ffmpeg filter,
  for extracting faces from an equirectangular frame and back.
"""

from __future__ import annotations

import math

import torch

# Cube faces, numbered as in the legacy code. Processing order from the paper.
PROC_ORDER = [6, 1, 2, 5, 3, 4]


def make_cube_to_equirectangular_map(
    w_plus_overlap: int, h_plus_overlap: int, overlap_w: int, overlap_h: int,
    out_w: int, out_h: int,
) -> torch.Tensor:
    """Displacement map (2,out_h,out_w) sampling a cube strip into equirectangular.

    The cube strip is 6 faces wide in order ``f, l, r, b, d, u``, each
    ``w_plus_overlap`` px wide (``cubeFaceWidth = w_plus_overlap - overlap_w``).
    """
    cube_face_w = w_plus_overlap - overlap_w
    cube_face_h = h_plus_overlap - overlap_h

    js = torch.arange(out_h, dtype=torch.float64).view(out_h, 1)
    is_ = torch.arange(out_w, dtype=torch.float64).view(1, out_w)
    v = 1.0 - js / out_h
    theta = v * math.pi
    u = is_ / out_w
    phi = u * 2 * math.pi

    x = torch.sin(phi) * torch.sin(theta) * -1.0
    y = torch.cos(theta).expand(out_h, out_w)
    z = torch.cos(phi) * torch.sin(theta) * -1.0
    x = x.expand(out_h, out_w)
    z = z.expand(out_h, out_w)

    a = torch.maximum(torch.maximum(x.abs(), y.abs()), z.abs())
    xa, ya, za = x / a, y / a, z / a

    x_pixel = torch.zeros(out_h, out_w, dtype=torch.float64)
    y_pixel = torch.zeros(out_h, out_w, dtype=torch.float64)
    x_offset = torch.zeros(out_h, out_w, dtype=torch.float64)

    # Faces resolved by priority (matching the legacy if/elseif order):
    # right (xa==1), left (xa==-1), up (ya==1), down (ya==-1), front (za==1), back (za==-1).
    eps = 1e-9
    assigned = torch.zeros(out_h, out_w, dtype=torch.bool)

    def assign(mask, xp, xoff, yp):
        mask = mask & ~assigned
        x_pixel[mask] = xp[mask] if torch.is_tensor(xp) else xp
        y_pixel[mask] = yp[mask] if torch.is_tensor(yp) else yp
        x_offset[mask] = xoff
        assigned.logical_or_(mask)

    assign(xa >= 1 - eps, ((za + 1) / 2 - 1) * cube_face_w, 2 * w_plus_overlap, (ya + 1) / 2 * cube_face_h)
    assign(xa <= -1 + eps, (za + 1) / 2 * cube_face_w, 1 * w_plus_overlap, (ya + 1) / 2 * cube_face_h)
    assign(ya >= 1 - eps, (xa + 1) / 2 * cube_face_w, 5 * w_plus_overlap, ((za + 1) / 2 - 1) * cube_face_h)
    assign(ya <= -1 + eps, (xa + 1) / 2 * cube_face_w, 4 * w_plus_overlap, (za + 1) / 2 * cube_face_h)
    assign(za >= 1 - eps, (xa + 1) / 2 * cube_face_w, 0 * w_plus_overlap, (ya + 1) / 2 * cube_face_h)
    assign(za <= -1 + eps, ((xa + 1) / 2 - 1) * cube_face_w, 3 * w_plus_overlap, (ya + 1) / 2 * cube_face_h)

    x_pixel = x_pixel.abs() + x_offset + overlap_w / 2
    y_pixel = y_pixel.abs() + overlap_h / 2

    dy = (y_pixel - js).to(torch.float32)
    dx = (x_pixel - is_).to(torch.float32)
    return torch.stack([dy, dx], dim=0)


def equirect_to_cubefaces(equi_rgb: torch.Tensor, face_size: int):
    """Extract 6 cube faces from an equirectangular RGB ``(3,H,W)`` tensor.

    Returns a dict ``{'F','R','B','L','U','D'}`` of ``(3, face_size, face_size)``
    tensors. Requires ``py360convert``.
    """
    import numpy as np

    try:
        import py360convert  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError("VR face extraction requires py360convert (pip install py360convert)") from e

    arr = equi_rgb.permute(1, 2, 0).cpu().numpy()
    faces = py360convert.e2c(arr, face_w=face_size, cube_format="dict")
    return {k: torch.from_numpy(np.ascontiguousarray(v)).permute(2, 0, 1).float() for k, v in faces.items()}


def cubefaces_to_equirect(faces: dict, out_h: int, out_w: int) -> torch.Tensor:
    """Reassemble an equirectangular RGB ``(3,out_h,out_w)`` from 6 cube faces."""
    import numpy as np

    try:
        import py360convert  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError("VR equirect assembly requires py360convert (pip install py360convert)") from e

    np_faces = {k: v.permute(1, 2, 0).cpu().numpy() for k, v in faces.items()}
    equi = py360convert.c2e(np_faces, h=out_h, w=out_w, cube_format="dict")
    return torch.from_numpy(np.ascontiguousarray(equi)).permute(2, 0, 1).float()
