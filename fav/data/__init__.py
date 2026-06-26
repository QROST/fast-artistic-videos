"""Training data: synthetic motion sources, real video, and the mix sampler."""

from fav.data.synthetic import (
    make_shift,
    make_zoom_out,
    make_single_image,
    SyntheticSource,
)
from fav.data.mixed import DataMix

__all__ = [
    "make_shift",
    "make_zoom_out",
    "make_single_image",
    "SyntheticSource",
    "DataMix",
]
