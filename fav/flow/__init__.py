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
]


def build_estimator(backend: str = "raft", **kwargs) -> FlowEstimator:
    """Construct a flow estimator by backend name ('raft' | 'ptlflow' | 'dummy')."""
    backend = backend.lower()
    if backend == "dummy":
        return DummyFlowEstimator()
    if backend == "raft":
        from fav.flow.raft import RaftFlowEstimator

        return RaftFlowEstimator(**kwargs)
    if backend == "ptlflow":
        from fav.flow.ptlflow_backend import PtlflowEstimator

        return PtlflowEstimator(**kwargs)
    raise ValueError(f"unknown flow backend {backend!r}")
