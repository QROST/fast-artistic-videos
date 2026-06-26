"""Real-video training tuples, replacing the monolithic HDF5 (DataLoader_video_real).

A "clip" is a list of consecutive RGB frames. For each adjacent pair we need the
backward flow (cur->prev) and a reliability mask; these are either computed on
the fly with a ``FlowEstimator`` or read from precomputed legacy assets
(``backward_{s}_{t}.flo`` / ``reliable_{s}_{t}.pgm``).

Output tuple matches the synthetic sources: ``imgsList`` (num+1 RGB frames),
``flowList`` (num ``(dy,dx)`` flows), ``certList`` (num ``[0,1]`` masks). The
flow is converted from native ``.flo`` ``(u,v)`` to the canonical ``(dy,dx)``
exactly as ``DataLoader_video_real.lua`` did with its channel swap.
"""

from __future__ import annotations

from pathlib import Path

import torch

from fav.flow.estimator import (
    FlowEstimator,
    compute_pair,
    flow_filename,
    occlusion_filename,
)
from fav.warp.flow_io import read_flo, read_pgm, uv_to_dydx


def video_tuple_from_frames(
    frames_rgb: list[torch.Tensor],
    estimator: FlowEstimator,
    use_structure: bool = True,
):
    """Build a training tuple by estimating flow/occlusion between frames.

    Args:
        frames_rgb: ``num+1`` RGB ``[0,1]`` tensors, each ``(1,3,H,W)``.
        estimator: flow estimator for the on-the-fly path.
    Returns:
        ``(imgsList rgb, flowList (1,2,H,W) dydx, certList (1,1,H,W) [0,1])``.
    """
    num = len(frames_rgb) - 1
    flow_list, cert_list = [], []
    for i in range(num):
        prev, cur = frames_rgb[i], frames_rgb[i + 1]
        backward_uv, reliable = compute_pair(estimator, prev, cur, use_structure=use_structure)
        flow_dydx = uv_to_dydx(backward_uv).unsqueeze(0)  # (1,2,H,W)
        cert = (reliable / 255.0).view(1, 1, *reliable.shape)
        flow_list.append(flow_dydx)
        cert_list.append(cert)
    return list(frames_rgb), flow_list, cert_list


def video_tuple_from_assets(
    frames_rgb: list[torch.Tensor],
    flow_dir: str | Path,
    frame_indices: list[int],
):
    """Build a tuple from precomputed legacy ``.flo`` / ``.pgm`` assets.

    ``frame_indices`` are the on-disk indices of ``frames_rgb`` so the asset
    filenames ``backward_{cur}_{prev}.flo`` can be located.
    """
    flow_dir = Path(flow_dir)
    num = len(frames_rgb) - 1
    flow_list, cert_list = [], []
    for i in range(num):
        prev_idx, cur_idx = frame_indices[i], frame_indices[i + 1]
        flow_uv = read_flo(flow_dir / flow_filename(cur_idx, prev_idx))
        cert_px = read_pgm(flow_dir / occlusion_filename(cur_idx, prev_idx)).float()
        flow_list.append(uv_to_dydx(flow_uv).unsqueeze(0))
        cert_list.append((cert_px / 255.0).view(1, 1, *cert_px.shape))
    return list(frames_rgb), flow_list, cert_list
