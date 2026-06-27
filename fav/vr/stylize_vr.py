"""Spherical (360) video stylization over cube faces.

Each cube face is a planar video, so temporal consistency *within* a face reuses
the planar engine (warp the previous frame's same face by its optical flow). The
6 faces are processed in the paper's order. Equirectangular output is assembled
with ``py360convert``.

For cross-face seam consistency, :func:`stylize_faces_seams_over_time` processes
faces *time-major* (all 6 faces of a timestep before the next) so that each
face's seam strip can be primed from its already-stylized neighbours of the same
timestep, blended into the per-face temporal prior with the gradient masks ported
in :mod:`fav.vr.seams`. :func:`stylize_faces_over_time` keeps the simpler
temporal-only path (each face stylized independently over time).
"""

from __future__ import annotations

import torch

from fav.infer.core import stylize_first_frame, stylize_sequence, stylize_with_prior
from fav.occlusion.filters import median_filter
from fav.preprocess import get_methods
from fav.vr.cubemap import PROC_ORDER, cubefaces_to_equirect
from fav.vr.seams import SeamGeometry, reblend_all_faces, seam_prior_and_cert
from fav.warp.grid_sample import warp


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


def stylize_faces_seams_over_time(
    model_vid,
    faces_seq: list[dict[int, torch.Tensor]],
    flows_seq: list[dict[int, torch.Tensor]],
    certs_seq: list[dict[int, torch.Tensor]],
    overlap_w: int = 20,
    overlap_h: int = 20,
    model_img="self",
    preprocessing: str = "vgg",
    occlusions_min_filter: int = 7,
    median_filter_size: int = 3,
    fill_occlusions: str = "vgg-mean",
    precision: str = "fp32",
) -> list[dict[int, torch.Tensor]]:
    """Seam-consistent cube-face video stylization (ports ``fast_artistic_video_vr``).

    Processes time-major: at each timestep the 6 faces are stylized in
    :data:`PROC_ORDER`, and every face after the first is primed with a
    cross-face *border prior* warped from the neighbours already stylized this
    timestep (see :mod:`fav.vr.seams`). The border is blended into the face's own
    temporal prior (its previous output warped by its optical flow) over the
    overlap strip, and that strip is marked certain. After all 6 faces are done,
    a second :func:`~fav.vr.seams.reblend_all_faces` pass re-blends every face
    with its four neighbours; that re-blended result is what carries to the next
    timestep's temporal prior *and* is returned (median-filtered) as the output,
    matching the legacy ``blend_other_sides`` step.

    All seam geometry runs in deprocessed RGB space (as the legacy
    ``last_segments`` do); the blended prior is preprocessed just before the
    network. ``overlap_w``/``overlap_h`` are the cube-face overlap used to build
    the seam geometry. Returns a list over time of ``{face_id: (1,3,H,W) rgb}``.
    """
    preprocess_fn, _ = get_methods(preprocessing)
    face_ids = sorted(PROC_ORDER)
    T = len(faces_seq)
    out_seq: list[dict[int, torch.Tensor]] = [dict() for _ in range(T)]
    # Per-face previous output in RGB space (the re-blended result), keyed by id.
    prev_rgb: dict[int, torch.Tensor] = {}

    geom = None
    for t in range(T):
        # `segments` holds this timestep's stylized faces (RGB) in processing
        # order, so neighbour lookups in fav.vr.seams resolve correctly.
        segments: list = []
        for p, face in enumerate(PROC_ORDER):
            frame_rgb = faces_seq[t][face]
            if geom is None:
                _, _, hplus, wplus = frame_rgb.shape
                geom = SeamGeometry.build(
                    hplus, wplus, overlap_w, overlap_h,
                    device=frame_rgb.device, dtype=frame_rgb.dtype,
                )
            if t == 0 and p == 0:
                # Very first face of the first timestep: a plain single-image
                # stylization (no temporal prior, no neighbour processed yet).
                out_rgb, _ = stylize_first_frame(
                    model_vid, model_img, frame_rgb, preprocessing,
                    fill_occlusions, precision,
                )
            else:
                if t == 0:
                    # First timestep, later faces: raw border prior, no temporal.
                    prior_rgb, cert = seam_prior_and_cert(
                        geom, segments, p, None, None, blend=False,
                        occlusions_min_filter=occlusions_min_filter,
                    )
                else:
                    # Temporal prior: this face's previous (re-blended) output
                    # warped by its flow, blended with the cross-face border.
                    flow = flows_seq[t - 1][face]
                    temporal_prior = warp(prev_rgb[face], flow)
                    prior_rgb, cert = seam_prior_and_cert(
                        geom, segments, p, temporal_prior, certs_seq[t - 1][face],
                        blend=True, occlusions_min_filter=occlusions_min_filter,
                    )
                # The cert is already combined (occlusion max border) and
                # min-filtered, so disable the filter inside the step; preprocess
                # the RGB prior to feed the network's warped-previous channels.
                out_rgb, _ = stylize_with_prior(
                    model_vid, frame_rgb, preprocess_fn(prior_rgb), cert,
                    preprocessing, occlusions_min_filter, fill_occlusions,
                    precision, apply_min_filter=False,
                )
            segments.append(out_rgb)

        # Second pass: re-blend every face with its 4 neighbours. This feeds both
        # the next timestep's temporal prior and the (median-filtered) output.
        reblended = reblend_all_faces(geom, segments)
        for p, face in enumerate(PROC_ORDER):
            prev_rgb[face] = reblended[p]
            out_seq[t][face] = _post(reblended[p], median_filter_size)
    assert set(face_ids) == set(PROC_ORDER)
    return out_seq


def _post(out_rgb, median_filter_size):
    if median_filter_size and median_filter_size > 1:
        return median_filter(out_rgb[0], median_filter_size).unsqueeze(0).clamp(0, 1)
    return out_rgb.clamp(0, 1)


# Map numeric face ids (1..6) to py360convert's dict keys for equirect assembly.
# Layout from the legacy code:  2=top, 1=bottom, 3=left, 6=front, 4=right, 5=back.
_FACE_KEY = {6: "F", 4: "R", 5: "B", 3: "L", 2: "U", 1: "D"}


def faces_to_equirect(faces: dict[int, torch.Tensor], out_h: int, out_w: int) -> torch.Tensor:
    """Assemble stylized cube faces ``{id: (1,3,H,W)}`` into an equirect image."""
    keyed = {_FACE_KEY[fid]: img[0] for fid, img in faces.items()}
    return cubefaces_to_equirect(keyed, out_h, out_w)
