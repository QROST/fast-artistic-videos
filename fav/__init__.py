"""fav — Fast Artistic Videos, faithful PyTorch/MPS port (Phase 1).

This package is a bit-faithful re-implementation of the 2018 Torch7/Lua
``fast-artistic-videos`` method (Ruder, Dosovitskiy, Brox). Phase 1 preserves
the original algorithm and every load-bearing constant; only dead native
dependencies (optical flow, the warp sampler, the C++ occlusion checker, the
VGG-16 loss network source, and the HDF5 storage) are swapped for modern
equivalents. See ``PHASE1_PLAN`` references in the repository for details.

The legacy ``.lua`` sources remain in the repository as the reference the port
must match.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
