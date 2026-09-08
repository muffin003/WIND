import numpy as np
import pytest

from wind_benchmark.benchmark import OptimizerInfo
from wind_benchmark.catalog import (
    OPTIMIZER_ALIASES,
    accepted_optimizer_names,
    canonical_optimizer_names,
    create_optimizer,
    resolve_optimizer_spec,
)
from wind_benchmark.experiment import HeavyBall, Nesterov
from wind_benchmark.oracle import Observation

EXPECTED_ALIASES = {
    "Nesterov": "HeavyBall",
    "AdamW": "CoupledL2Adam",
    "OnePointSPSA": "OnePointTemporalSecant",
    "FDSA": "NormalizedRandomSecant",
    "SPSA": "ScaledTemporalSecant",
    "ZOSGD": "GaussianTemporalSecant",
    "ZOSignSGD": "SignedGaussianTemporalSecant",
    "KieferWolfowitz": "ShrinkingTemporalSecant",
    "NedicSubgradient": "DiminishingGaussianSecant",
    "AcceleratedSPSA": "MomentumTemporalSecant",
    "CMAES": "EliteCovarianceSearch",
    "GPUCB": "DistanceScaledExploration",
}


def _gradient_observation(x, gradient):
    return Observation(
        x=np.asarray(x, dtype=float),
        t=0,
        value=0.0,
        grad=np.asarray(gradient, dtype=float),
        optimum_value=0.0,
        mode="first-order",
    )


def test_catalog_exposes_only_canonical_names():
    canonical = canonical_optimizer_names()
    assert len(canonical) == 25
    assert len(canonical) == len(set(canonical))
    assert not set(EXPECTED_ALIASES).intersection(canonical)
    assert accepted_optimizer_names() == frozenset((*canonical, *EXPECTED_ALIASES))


def test_legacy_aliases_resolve_to_declared_canonical_names():
    assert OPTIMIZER_ALIASES == EXPECTED_ALIASES
    for alias, canonical_name in EXPECTED_ALIASES.items():
        assert resolve_optimizer_spec(alias).canonical_name == canonical_name
        with pytest.warns(FutureWarning, match=f"{alias} is a legacy alias"):
            optimizer = create_optimizer(alias)
        assert optimizer.name == canonical_name
        assert optimizer.canonical_name == canonical_name
        assert optimizer.requested_alias == alias


def test_alias_creation_warns_and_exports_canonical_identity():
    with pytest.warns(FutureWarning, match="AdamW is a legacy alias"):
        optimizer = create_optimizer("AdamW", {"lr": 0.02})

    assert optimizer.name == "CoupledL2Adam"
    assert optimizer.canonical_name == "CoupledL2Adam"
    assert optimizer.requested_alias == "AdamW"
    assert optimizer.implementation_id == "coupled_l2_adam_v1"
    info = OptimizerInfo.from_optimizer(optimizer)
    assert info.name == "CoupledL2Adam"
    assert info.requested_alias == "AdamW"
    assert info.implementation_id == "coupled_l2_adam_v1"


def test_legacy_nesterov_preserves_historical_heavy_ball_update():
    heavy_ball = HeavyBall(lr=0.05, beta=0.9)
    with pytest.warns(FutureWarning, match="Nesterov is a legacy alias"):
        legacy = Nesterov(lr=0.05, beta=0.9)

    heavy_ball.reset()
    legacy.reset()
    observations = [
        _gradient_observation([0.0, 0.0], [1.0, -2.0]),
        _gradient_observation([-0.05, 0.1], [0.5, -0.25]),
    ]
    for observation in observations:
        np.testing.assert_allclose(
            heavy_ball.step(observation), legacy.step(observation)
        )


def test_lookahead_nesterov_is_distinct_from_heavy_ball():
    heavy_ball = create_optimizer("HeavyBall", {"lr": 0.05, "beta": 0.9})
    lookahead = create_optimizer("LookaheadNesterov", {"lr": 0.05, "beta": 0.9})
    first = _gradient_observation([0.0, 0.0], [1.0, -2.0])
    first_heavy = heavy_ball.step(first)
    first_lookahead = lookahead.step(first)
    np.testing.assert_allclose(first_heavy, first_lookahead)

    second = _gradient_observation(first_heavy, [0.5, -0.25])
    assert not np.allclose(heavy_ball.step(second), lookahead.step(second))
