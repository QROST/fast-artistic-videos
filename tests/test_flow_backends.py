"""Tests for the swappable flow-backend dispatch (PR-D)."""

import pytest

from fav.flow import build_estimator, DummyFlowEstimator, PTLFLOW_SHORTHANDS, RECOMMENDED_BACKEND


def test_dummy_backend():
    assert isinstance(build_estimator("dummy"), DummyFlowEstimator)


def test_recommended_backend_is_sea_raft():
    assert RECOMMENDED_BACKEND == "sea_raft"
    assert "sea_raft" in PTLFLOW_SHORTHANDS
    assert "flowformer" in PTLFLOW_SHORTHANDS


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError):
        build_estimator("not_a_backend")


def test_sota_shorthands_route_to_ptlflow():
    # Without ptlflow installed, the shorthand must still ROUTE to the ptlflow
    # backend (RuntimeError about the missing dep), not fail as an unknown name.
    for name in ("sea_raft", "flowformer"):
        with pytest.raises(RuntimeError) as e:
            build_estimator(name)
        assert "ptlflow" in str(e.value).lower()


def test_raft_routes_to_torchvision():
    # Likewise, 'raft' routes to the torchvision backend (RuntimeError if absent),
    # not ValueError.
    try:
        est = build_estimator("raft")
    except RuntimeError as e:
        assert "torchvision" in str(e).lower()
    else:
        # torchvision present: we got a real estimator.
        from fav.flow.estimator import FlowEstimator

        assert isinstance(est, FlowEstimator)


def test_ptlflow_explicit_model_routing():
    with pytest.raises(RuntimeError):
        build_estimator("ptlflow", model="gma")
