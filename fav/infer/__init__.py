"""Inference: planar (and, in fav.vr, spherical) video stylization."""

from fav.infer.core import (
    stylize_first_frame,
    stylize_next_frame,
    stylize_sequence,
)

__all__ = ["stylize_first_frame", "stylize_next_frame", "stylize_sequence"]
