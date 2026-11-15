"""Independent formula, accounting and development-selection checks for 0.7."""

from dataclasses import replace
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from wind_benchmark.algorithms.catalog import create_optimizer, OPTIMIZER_SPECS
from wind_benchmark.algorithms.protocol import TaskSpec
from wind_benchmark.core.oracle import Observation
from wind_benchmark.core.privacy import EvaluationSession, GroundTruthAccessError
from wind_benchmark.experiments import paper


class Client:
    def __init__(self, budget=100, value=lambda x: x @ x / 2, grads=None):
        self.budget, self.value, self.grads = budget, value, grads
        self.points = []

    @property
    def remaining(self):
        return self.budget - len(self.points)

    def query(self, x):
        assert self.remaining > 0
        self.points.append(x.copy())
        return Observation(
            x=x,
            t=len(self.points) - 1,
            query_index=len(self.points),
            value=self.value(x),
            grad=None if self.grads is None else self.grads[len(self.points) - 1],
        )


def start(name, params=None, seed=123, x=(1.0, -2.0)):
    alg = create_optimizer(name, params)
    alg.initialize(np.array(x), np.random.default_rng(seed), TaskSpec(len(x)))
    return alg


def check(x, y):
    assert_allclose(x, y, atol=1e-12, rtol=1e-10)


def test_adagrad_accumulates_without_forgetting():
    alg = start("AdaGrad", {"lr": 0.2, "eps": 0.1})
    c = Client(grads=[np.array([2.0, -4.0]), np.zeros(2), np.array([-1.0, 2.0])])
    first = np.array([1.0, -2.0]) - 0.2 * np.array([2.0, -4.0]) / np.array([2.1, 4.1])
    check(alg.advance(c).recommendation, first)
    check(alg.advance(c).recommendation, first)
    check(
        alg.advance(c).recommendation,
        first - 0.2 * np.array([-1.0, 2.0]) / (np.sqrt([5.0, 20.0]) + 0.1),
    )
    check(alg.squares, [5.0, 20.0])


def test_normalized_spsa_exact_scale_and_minus_plus_order():
    x = np.array([1.0, -2.0])
    estimates = []
    # Enumerate all Bernoulli signs independently; no Monte Carlo tolerance.
    for seed in range(100):
        delta = np.random.default_rng(seed).choice((-1.0, 1.0), size=2) / np.sqrt(2)
        alg, c = start("SPSA", seed=seed), Client()
        result = alg.advance(c)
        check(c.points[0], x - 0.1 * delta)
        check(c.points[1], x + 0.1 * delta)
        check(result.estimated_gradient, (x @ delta) * delta)
    for signs in itertools.product((-1.0, 1.0), repeat=2):
        u = np.array(signs) / np.sqrt(2)
        estimates.append((x @ u) * u)
    check(np.mean(estimates, axis=0), x / 2)


def test_acceleration_two_steps_and_constant_gamma():
    alg = start("AcceleratedSPSA")
    x = np.array([1.0, -2.0])
    z = x.copy()
    rng = np.random.default_rng(123)
    c = Client()
    for _ in range(3):
        u = (x + 0.1 * z) / 1.1
        direction = rng.choice((-1.0, 1.0), size=2) / np.sqrt(2)
        g = (u @ direction) * direction
        x = u - 0.005 * g
        z = 0.9 * z + 0.1 * u - (0.1 / 1.05) * g
        result = alg.advance(c)
        check(result.estimation_point, u)
        check(result.recommendation, x)
        check(alg.z, z)
        assert alg.gamma == 1.05
    alg = start("AcceleratedSPSA")
    check(alg.advance(Client(value=lambda x: 7.0)).recommendation, [1.0, -2.0])
    check(alg.z, [1.0, -2.0])


def test_one_point_formula_and_additive_constant():
    direction = np.random.default_rng(123).choice((-1.0, 1.0), size=2)
    point = np.array([1.0, -2.0]) + 0.1 * direction
    values = []
    for offset in (0.0, 7.0):
        alg, c = start("OnePointSPSA"), Client(value=lambda x: x @ x / 2 + offset)
        update = alg.advance(c)
        expected = (point @ point / 2 + offset) / (0.1 * direction)
        check(update.estimated_gradient, expected)
        check(update.recommendation, np.array([1.0, -2.0]) - 0.005 * expected)
        assert update.queries == len(c.points) == 1
        values.append(update.estimated_gradient)
    assert not np.array_equal(*values)


def test_zo_sign_shared_baseline_and_sign_of_average():
    alg, c = start("ZOSignSGD", {"m": 3}), Client()
    rng = np.random.default_rng(123)
    x = np.array([1.0, -2.0])
    expected = np.zeros(2)
    for _ in range(3):
        u = rng.normal(size=2)
        u /= np.sqrt(sum(u * u))
        expected += ((x + 0.01 * u) @ (x + 0.01 * u) / 2 - x @ x / 2) * u
    expected *= 2 / (3 * 0.01)
    result = alg.advance(c)
    assert len(c.points) == result.queries == 4
    check(c.points[0], x)
    check(result.estimated_gradient, expected)
    check(result.recommendation, x - 0.005 * np.sign(expected))
    second = alg.advance(c)
    check(c.points[4], result.recommendation)
    assert second.queries == 4


@pytest.mark.parametrize(
    "name,params,cost",
    [
        ("SPSA", {}, 2),
        ("AcceleratedSPSA", {}, 2),
        ("RandomSearch", {}, 2),
        ("ZOSignSGD", {"m": 3}, 4),
        ("KieferWolfowitz", {}, 4),
    ],
)
def test_partial_updates_preserve_all_recommendation_state(name, params, cost):
    alg = start(name, params)
    c = Client(budget=cost - 1)
    old_z = getattr(alg, "z", None)
    update = alg.advance(c)
    assert not update.update_complete and update.queries == cost - 1 and alg.t == 0
    check(update.recommendation, [1.0, -2.0])
    if old_z is not None:
        check(alg.z, old_z)
    assert alg.advance(c).queries == 0


def test_random_search_both_orders_fresh_values_ties_and_failures():
    orders = set()
    for seed in range(12):
        for accept in (True, False, None):
            alg = start("RandomSearch", seed=seed)
            x = alg.x.copy()
            c = Client(
                value=lambda p: (
                    1.0
                    if np.array_equal(p, x)
                    else (0.0 if accept is True else 2.0 if accept is False else 1.0)
                )
            )
            result = alg.advance(c)
            base_first = np.array_equal(c.points[0], x)
            orders.add(base_first)
            candidate = c.points[1 if base_first else 0]
            check(result.recommendation, candidate if accept is True else x)
            assert result.estimated_gradient is result.estimation_point is None
            alg.advance(c)
            assert len(c.points) == 4
    assert orders == {True, False}
    with pytest.raises((FloatingPointError, ValueError)):
        start("RandomSearch").advance(Client(value=lambda x: np.nan))


@pytest.mark.parametrize(
    "name,params",
    [
        ("SPSA", {"beta": 0}),
        ("AcceleratedSPSA", {"a": 0}),
        ("AcceleratedSPSA", {"delta": float("inf")}),
        ("AdaGrad", {"eps": 0}),
        ("OnePointSPSA", {"r": -1}),
        ("ZOSignSGD", {"m": True}),
        ("RandomSearch", {"sigma": 0}),
    ],
)
def test_new_parameter_validation(name, params):
    with pytest.raises(ValueError):
        create_optimizer(name, params)


def test_retired_coordinate_name_is_not_an_alias():
    with pytest.raises(KeyError):
        create_optimizer("CoordinateFiniteDifferenceSGD")


def test_static_catalog_matches_python_defaults():
    app = (Path(__file__).resolve().parents[1] / "web/app.js").read_text("utf-8")
    match = re.search(r"const OPTIMIZERS = (\[.*?\]);", app, re.S)
    entries = {v["name"]: v for v in json.loads(match[1])}
    assert set(entries) == {s.canonical_name for s in OPTIMIZER_SPECS}
    for spec in OPTIMIZER_SPECS:
        assert entries[spec.canonical_name]["params"] == spec.default_parameters


def test_equal_tuning_grid_and_quadratic_failure_status():
    profile = paper.ExperimentProfile.build("paper")
    for method in paper._zero_order_methods():
        grid = paper._zero_order_candidate_hyperparameters(profile, method, 20)
        assert len(grid) == (1 if method.family == "first_order" else 12)
        assert len({json.dumps(v, sort_keys=True) for v in grid}) == len(grid)
    method = paper.ZeroOrderMethodConfig("random", "random_search")
    _, result = paper._run_zero_order_method(
        method,
        "frozen",
        paper.ZeroOrderCase("x", 2, 1.0, "stationary"),
        10,
        np.arange(11),
        9001,
        None,
    )
    assert result["status"] == "TUNING_FAILED" and result["queries"] == 0
    assert result["implementation_version"] == 1
    assert result["rng_stream_id"] == method.rng_stream_id
    assert result["parameters"] is None


def test_all_failed_tuning_does_not_pick_first(monkeypatch):
    def failed(*args, **kwargs):
        return None, dict(
            failed=1, query_averaged_mse=np.nan, queries=1, status="NUMERICAL_FAILURE"
        )

    monkeypatch.setattr(paper, "_run_zero_order_method", failed)
    choice, rows = paper._tune_zero_order_methods(
        paper.ExperimentProfile.build("smoke"),
        [paper.ZeroOrderCase("x", 2, 1.0, "stationary")],
        [paper.ZeroOrderMethodConfig("SPSA", "spsa")],
    )
    assert choice[(2, "SPSA")] is None
    assert not rows.selected.any() and rows.selection_status.eq("TUNING_FAILED").all()


def test_tuning_cache_is_guarded_and_reuses_no_queries(monkeypatch, trusted_core_unit):
    seen = []

    def fake(*args, **kwargs):
        seen.append(args[5])
        assert args[5] in (9001, 9002)
        return None, dict(failed=0, query_averaged_mse=1.0, queries=5, status="SUCCESS")

    monkeypatch.setattr(paper, "_run_zero_order_method", fake)
    # Exercise cache through trusted orchestration, not a user-granted evaluator context.
    profile = replace(
        paper.ExperimentProfile.build("smoke"),
        zero_order_budget=4,
        zero_order_tuning_budget=5,
    )
    cases = [paper.ZeroOrderCase("x", 2, 1.0, "stationary")]
    methods = [paper.ZeroOrderMethodConfig("SPSA", "spsa")]
    with EvaluationSession():
        cache = paper._TuningCache()
        a, rows = paper._tune_zero_order_methods(profile, cases, methods, cache)
        b, reused = paper._tune_zero_order_methods(profile, cases, methods, cache)
    assert a == b and len(seen) == 24 and reused.charged_queries.sum() == 0
    assert a[(2, "SPSA")].parameters == {"alpha": 0.05, "beta": 0.03}


def test_tuning_cache_blocks_retained_access_and_foreign_owner():
    outside = paper._TuningCache()
    with EvaluationSession():
        guarded = paper._TuningCache()
        with pytest.raises(GroundTruthAccessError):
            _ = guarded.entries
        with pytest.raises(ValueError, match="current evaluation series"):
            paper.run_zero_order_study(
                paper.ExperimentProfile.build("smoke"), _tuning_cache=outside
            )
