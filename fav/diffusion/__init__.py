"""Phase-3 diffusion path (additive, quality-first).

Step 3a — the *reuse bridge*: turn the Phase-1/2 optical flow + occlusion + warp
into ControlNet-style conditioning tensors for a diffusion/LoRA stylizer. Pure
tensor logic, device-agnostic (CPU/CUDA/**MPS**), no ``diffusers`` dependency.
"""

from fav.diffusion.conditioning import (
    ConditioningBundle,
    build_conditioning,
    first_frame_conditioning,
    flow_to_rgb,
    sobel_edges,
    stack_controls,
)
from fav.diffusion.pipeline import (
    DiffusionStylizer,
    DummyDiffusionStylizer,
    SDXLControlNetStylizer,
    stylize_video_diffusion,
    build_stylizer,
)

__all__ = [
    "ConditioningBundle",
    "build_conditioning",
    "first_frame_conditioning",
    "flow_to_rgb",
    "sobel_edges",
    "stack_controls",
    "DiffusionStylizer",
    "DummyDiffusionStylizer",
    "SDXLControlNetStylizer",
    "stylize_video_diffusion",
    "build_stylizer",
]
