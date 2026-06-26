"""Device / backend selection for CPU, CUDA, and Apple-Silicon MPS.

Phase-1 policy (see plan):

* fp32 everywhere — MPS autocast/bf16 is deferred to Phase 2 because the Gram
  matrices sum over ``H*W`` and style features are precision sensitive.
* ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set as early as possible so any op that
  is not yet implemented for the MPS backend silently falls back to CPU instead
  of raising.
* A few ops are still better run on CPU explicitly for correctness (e.g. the
  median used by ``median_filter``); helpers that need that decide per-op.
"""

from __future__ import annotations

import os

# Must be set before the first MPS op is dispatched. Importing this module is
# the canonical way to opt in.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (import after setting the env var)


def select_device(prefer: str | None = None) -> torch.device:
    """Return the best available device.

    Args:
        prefer: ``"mps" | "cuda" | "cpu"`` to force a choice, or ``None`` to
            auto-select in the order MPS -> CUDA -> CPU.
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "mps" and mps_available():
            return torch.device("mps")
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "cpu":
            return torch.device("cpu")
        # Fall through to auto if the requested device is unavailable.

    if mps_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mps_available() -> bool:
    """True if the Apple-Silicon MPS backend is built and usable."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def default_dtype() -> torch.dtype:
    """Phase-1 compute dtype: always fp32."""
    return torch.float32


def is_mps(device: torch.device | str) -> bool:
    return torch.device(device).type == "mps"


def synchronize(device: torch.device | str) -> None:
    """Block until queued work on ``device`` finishes (for timing)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
