"""Spherical (360) video stylization over cube faces.

Each cube face is a planar video, so temporal consistency *within* a face reuses
the planar engine (warp the previous frame's same face by its optical flow). The
6 faces are processed in the paper's order. Equirectangular output is assembled
with ``py360convert``.

Cross-face seam consistency (the perspective-map border priors + gradient
blending from ``fast_artistic_video_vr.lua``) is the remaining VR refinement for
this phase; the geometry it needs is already provided in
``fav.vr.perspective`` / ``fav.vr.cubemap`` and the temporal path below gives the
primary (frame-to-frame) consistency.
"""

from __future__ import annotations

import torch

from fav.infer.core import stylize_sequence
from fav.vr.cubemap import PROC_ORDER, cubefaces_to_equirect


def stylize_faces_over_time(
    model_vid,
    faces_seq: list[dict[int, torch.Tensor]],
    flows_seq: list[dict[int, torch.Tensor]],
    certs_seq: list[dict[int, torch.Tensor]],
    model_img="self",
    occlusions_min_filter=7,
    median_filter_size=3,
    fill_occlusions="vgg-mean",
) -> list[dict[int, torch.Tensor]]:
    """Stylize a cube-face video.

    Args:
        faces_seq: list over time of ``{face_id: (1,3,H,W) rgb}``.
        flows_seq: list (len T-1) of ``{face_id: (1,2,H,W) (dy,dx)}`` temporal flow.
        certs_seq: list (len T-1) of ``{face_id: (1,1,H,W) [0,1]}``.
    Returns:
        list over time of ``{face_id: (1,3,H,W) stylized rgb}``.
    """
    face_ids = sorted(PROC_ORDER)
    T = len(faces_seq)
    out_seq: list[dict[int, torch.Tensor]] = [dict() for _ in range(T)]

    for face in PROC_ORDER:
        frames = [faces_seq[t][face] for t in range(T)]
        flows = [flows_seq[t][face] for t in range(T - 1)]
        certs = [certs_seq[t][face] for t in range(T - 1)]
        styled = stylize_sequence(
            model_vid, frames, flows, certs, model_img=model_img,
            occlusions_min_filter=occlusions_min_filter,
            median_filter_size=median_filter_size, fill_occlusions=fill_occlusions,
        )
        for t in range(T):
            out_seq[t][face] = styled[t]
    assert set(face_ids) == set(PROC_ORDER)
    return out_seq


# Map numeric face ids (1..6) to py360convert's dict keys for equirect assembly.
# Layout from the legacy code:  2=top, 1=bottom, 3=left, 6=front, 4=right, 5=back.
_FACE_KEY = {6: "F", 4: "R", 5: "B", 3: "L", 2: "U", 1: "D"}


def faces_to_equirect(faces: dict[int, torch.Tensor], out_h: int, out_w: int) -> torch.Tensor:
    """Assemble stylized cube faces ``{id: (1,3,H,W)}`` into an equirect image."""
    keyed = {_FACE_KEY[fid]: img[0] for fid, img in faces.items()}
    return cubefaces_to_equirect(keyed, out_h, out_w)
