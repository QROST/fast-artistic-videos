"""Frame-by-frame stylization driver, ported from ``fast_artistic_video_core.lua``.

The previous stylized output is kept in VGG space (the model's output space),
warped into the current frame by the backward flow, masked by the certainty, and
concatenated with the current frame and the mask to form the 7-channel input.
The first frame is stylized either by a separate image model or by the video
model itself with an all-occluded prior ('self').
"""

from __future__ import annotations

import torch

from fav.occlusion.filters import median_filter, min_filter
from fav.preprocess import get_methods
from fav.warp.grid_sample import warp


def _fill(cert, fill_occlusions, preprocess_fn):
    if fill_occlusions == "vgg-mean":
        return torch.zeros(cert.shape[0], 3, *cert.shape[2:], device=cert.device, dtype=cert.dtype)
    if fill_occlusions == "uniform-random":
        b, _, h, w = cert.shape
        rnd = preprocess_fn(torch.rand(b, 3, h, w, device=cert.device, dtype=cert.dtype))
        return rnd * (1.0 - cert)
    raise ValueError(f"unknown fill_occlusions {fill_occlusions!r}")


@torch.no_grad()
def stylize_first_frame(model_vid, model_img, frame_rgb, preprocessing="vgg",
                        fill_occlusions="vgg-mean"):
    """Stylize the first frame; returns ``(out_rgb, out_pre)`` (out_pre in VGG space)."""
    preprocess_fn, deprocess_fn = get_methods(preprocessing)
    frame_pre = preprocess_fn(frame_rgb)
    b, _, h, w = frame_pre.shape
    if model_img is not None and model_img != "self":
        out_pre = model_img(frame_pre)
    else:
        # 'self': video model with an all-occluded empty prior.
        prior = _fill(torch.zeros(b, 1, h, w, device=frame_pre.device, dtype=frame_pre.dtype),
                      fill_occlusions, preprocess_fn)
        cert = torch.zeros(b, 1, h, w, device=frame_pre.device, dtype=frame_pre.dtype)
        out_pre = model_vid(torch.cat([frame_pre, prior, cert], dim=1))
    return deprocess_fn(out_pre).clamp(0, 1), out_pre


@torch.no_grad()
def stylize_next_frame(model_vid, prev_out_pre, frame_rgb, flow_dydx, cert,
                       preprocessing="vgg", occlusions_min_filter=7,
                       fill_occlusions="vgg-mean"):
    """Stylize a subsequent frame using the warped previous output as prior."""
    preprocess_fn, deprocess_fn = get_methods(preprocessing)
    frame_pre = preprocess_fn(frame_rgb)
    cert = min_filter(cert, occlusions_min_filter)
    warped = warp(prev_out_pre, flow_dydx)
    warped_masked = warped * cert
    fill = _fill(cert, fill_occlusions, preprocess_fn)
    inp = torch.cat([frame_pre, warped_masked + fill, cert], dim=1)
    out_pre = model_vid(inp)
    return deprocess_fn(out_pre).clamp(0, 1), out_pre


@torch.no_grad()
def stylize_sequence(model_vid, frames_rgb, flows, certs, model_img="self",
                     preprocessing="vgg", occlusions_min_filter=7, median_filter_size=3,
                     fill_occlusions="vgg-mean"):
    """Stylize a whole clip.

    Args:
        frames_rgb: list of ``(1,3,H,W)`` RGB frames.
        flows: list (len N-1) of ``(1,2,H,W)`` ``(dy,dx)`` backward flows (cur->prev).
        certs: list (len N-1) of ``(1,1,H,W)`` certainties in [0,1].
        model_img: separate first-frame model, or 'self'.
    Returns:
        list of stylized ``(1,3,H,W)`` RGB frames in [0,1].
    """
    outputs = []
    out_rgb, prev_pre = stylize_first_frame(
        model_vid, model_img, frames_rgb[0], preprocessing, fill_occlusions
    )
    outputs.append(_post(out_rgb, median_filter_size))
    for i in range(1, len(frames_rgb)):
        out_rgb, prev_pre = stylize_next_frame(
            model_vid, prev_pre, frames_rgb[i], flows[i - 1], certs[i - 1],
            preprocessing, occlusions_min_filter, fill_occlusions,
        )
        outputs.append(_post(out_rgb, median_filter_size))
    return outputs


def _post(out_rgb, median_filter_size):
    if median_filter_size and median_filter_size > 1:
        return median_filter(out_rgb[0], median_filter_size).unsqueeze(0).clamp(0, 1)
    return out_rgb
