"""Optical-flow estimation (swappable backends) and dataset flow/occlusion writing."""

from fav.flow.estimator import (
    FlowEstimator,
    DummyFlowEstimator,
    rescale_flow,
    flow_filename,
    occlusion_filename,
    compute_pair,
    write_pair,
)

__all__ = [
    "FlowEstimator",
    "DummyFlowEstimator",
    "rescale_flow",
    "flow_filename",
    "occlusion_filename",
    "compute_pair",
    "write_pair",
    "build_estimator",
    "PTLFLOW_SHORTHANDS",
    "RECOMMENDED_BACKEND",
]


# SOTA estimators exposed as first-class backend names (routed to ptlflow).
# SEA-RAFT is the recommended higher-quality default when ptlflow is installed.
PTLFLOW_SHORTHANDS = ("sea_raft", "flowformer", "gma", "gmflow", "rapidflow", "memflow")
RECOMMENDED_BACKEND = "sea_raft"


def build_estimator(backend: str = "raft", model: str | None = None,
                    ckpt: str = "things", device=None) -> FlowEstimator:
    """Construct a flow estimator.

    Backends:
      * ``dummy``  — zero flow (tests / analytic pipelines).
      * ``raft``   — torchvision RAFT (out-of-box default; ``model`` =
        ``raft_large``/``raft_small``).
      * ``ptlflow``— any ptlflow model via ``model`` (e.g. ``sea_raft``).
      * ``sea_raft`` / ``flowformer`` / ``gma`` / ... — shorthands that route to
        ptlflow with that model name. ``sea_raft`` is the recommended SOTA option.
    """
    backend = backend.lower()
    if backend == "dummy":
        return DummyFlowEstimator()
    if backend == "raft":
        from fav.flow.raft import RaftFlowEstimator

        return RaftFlowEstimator(model=model or "raft_large", device=device)
    if backend == "ptlflow" or backend in PTLFLOW_SHORTHANDS:
        from fav.flow.ptlflow_backend import PtlflowEstimator

        name = model if backend == "ptlflow" else backend
        return PtlflowEstimator(model=name or RECOMMENDED_BACKEND, ckpt=ckpt, device=device)
    raise ValueError(
        f"unknown flow backend {backend!r}; expected dummy|raft|ptlflow|" + "|".join(PTLFLOW_SHORTHANDS)
    )
