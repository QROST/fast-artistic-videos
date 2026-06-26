"""Perceptual (content + style) loss, Gram matrices, and temporal/TV penalties."""

from fav.losses.gram import gram_matrix, GramMatrix
from fav.losses.temporal import tv_penalty, temporal_pixel_loss
from fav.losses.vgg_loss_net import VGG16Features, build_vgg16_loss_net
from fav.losses.perceptual import PerceptualCriterion

__all__ = [
    "gram_matrix",
    "GramMatrix",
    "tv_penalty",
    "temporal_pixel_loss",
    "VGG16Features",
    "build_vgg16_loss_net",
    "PerceptualCriterion",
]
