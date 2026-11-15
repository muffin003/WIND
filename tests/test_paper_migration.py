"""Protocol integration plus numerical goldens captured before the migration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wind_benchmark.experiments import paper as paper
from wind_benchmark.core.environment import (
    DynamicEnvironment,
    GaussianNoise,
    LinearDrift,
    PrescribedDrift,
    QuadraticLandscape,
)
from wind_benchmark.core.execution import OracleSession, QueryBudgetExhausted
from wind_benchmark.core.oracle import FirstOrderOracle, ZeroOrderOracle

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/paper_migration_reference.json").read_text()
)


def make_session(protocol="streaming", budget=4):
    env = DynamicEnvironment(
        1,
        LinearDrift(np.ones(1)),
        QuadraticLandscape(1),
        np.zeros(1),
    )
    oracle = ZeroOrderOracle(env)
    return env, oracle, OracleSession(oracle, budget, protocol=protocol)


@pytest.mark.parametrize("protocol", ["frozen", "streaming"])
def test_session_time_budget_and_capability_closure(protocol):
    env, oracle, session = make_session(protocol)
    with session.client() as client:
        first = client.query(np.zeros(1))
        second = client.query(np.zeros(1))
        assert not hasattr(client, "environment")
        assert not hasattr(client, "finish")
        assert not hasattr(first, "theta") and not hasattr(first, "clean_value")
        assert first.grad is None
        with pytest.raises(PermissionError, match="active"):
            session.finish()
    assert env.t == 2
    assert first.value == 0
    assert second.value == (0 if protocol == "frozen" else 0.5)
    with pytest.raises(RuntimeError):
        client.query(np.zeros(1))
    with session.client() as last:
        third = last.query(np.zeros(1))
        last.query(np.zeros(1))
        with pytest.raises(QueryBudgetExhausted):
            last.query(np.zeros(1))
    assert third.value == 2.0
    assert env.t == 4
    with pytest.raises(PermissionError):
        _ = oracle._cached_theta
    records = session.finish()
    assert not oracle._step_active and not env._locked_by_oracle
    assert len(records) == 4
    assert session.counts == {"queries": 4, "value_queries": 4, "gradient_queries": 0}
    np.testing.assert_array_equal(
        [r.theta[0] for r in records],
        [0, 0, 2, 2] if protocol == "frozen" else [0, 1, 2, 3],
    )
    assert not records[0].theta.flags.writeable
    with pytest.raises(RuntimeError, match="closed"):
        session.observe(np.zeros(1))


@pytest.mark.parametrize("protocol", ["frozen", "streaming"])
def test_exception_closes_client_unlocks_and_accounts_for_consumed_queries(protocol):
    env, oracle, session = make_session(protocol)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        with session.client() as client:
            client.query(np.zeros(1))
            client.query(np.zeros(1))
            raise RuntimeError("optimizer failed")
    assert env.t == oracle.n_queries == 2
    with pytest.raises(PermissionError):
        _ = oracle._cached_theta
    with pytest.raises(PermissionError):
        session.finish()
    with pytest.raises(RuntimeError):
        client.query(np.zeros(1))


@pytest.mark.parametrize("protocol", ["frozen", "streaming"])
def test_failed_observation_still_counts_and_advances_time(protocol):
    env, oracle, session = make_session(protocol)
    with pytest.raises(ValueError, match="Non-finite"):
        session.observe(np.array([np.nan]))
    assert env.t == oracle.n_queries == 1
    with pytest.raises(PermissionError):
        _ = oracle._cached_theta
    with pytest.raises(PermissionError):
        session.finish()


def test_prescribed_drift_replays_resets_and_does_not_modify_input():
    path = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    env = paper._path_environment(QuadraticLandscape(2), path)
    path[:] = 99
    env.step()
    np.testing.assert_array_equal(env.get_current_theta(True), [0, 1])
    env.step()
    np.testing.assert_array_equal(env.get_current_theta(True), [0.5, 0.5])
    env.reset()
    np.testing.assert_array_equal(env.get_current_theta(True), [1, 0])
    with pytest.raises(ValueError):
        PrescribedDrift(np.zeros(2))


@pytest.mark.parametrize("study", ["calibration", "memory", "geometry"])
def test_original_study_runs_match_and_use_library_gradient_oracle(study, monkeypatch):
    queries = []
    original_query = FirstOrderOracle.query

    def query(self, point):
        result = original_query(self, point)
        assert result.value is None and result.grad is not None
        assert not hasattr(result, "theta") and not hasattr(result, "clean_value")
        queries.append(result.query_index)
        return result

    monkeypatch.setattr(FirstOrderOracle, "query", query)
    result = getattr(paper, f"run_{study}_study")(
        paper.ExperimentProfile.build("smoke")
    )
    reference = REFERENCE["studies"][study]
    expected = pd.DataFrame(reference["data"], columns=reference["columns"])
    # v3 intentionally uses its own RNG namespace. Historical exact values
    # remain in the frozen archive, not as references for the new streams.
    assert (
        len(result.runs[result.runs.method != "AdaGrad"]) == len(expected)
        if study == "memory"
        else len(result.runs) == len(expected)
    )
    numeric = result.runs.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()
    assert result.histories["index"]
    assert (
        len(queries) == result.runs.queries.sum() == result.runs.gradient_queries.sum()
    )
    assert result.runs.value_queries.eq(0).all()


@pytest.mark.parametrize(
    "reference",
    REFERENCE["zero_order"],
    ids=lambda r: f"{r['kind']}-{r['noise']}-{r['method']}-{r['protocol']}",
)
def test_v2_zero_order_replay_and_query_semantics(reference, monkeypatch):
    name = (
        reference["method"]
        .replace("Gaussian-m", "SphericalZOSGD-m")
        .replace("PairedSPSA-m1", "SPSA")
        .replace("CoordinateCentral", "KieferWolfowitz")
    )
    method = next(m for m in paper._zero_order_methods() if m.name == name)
    params = {"lr": 0.02}
    if method.family == "spsa":
        params = {"alpha": 0.02, "beta": 0.1}
    elif method.family != "first_order":
        params["h"] = 0.1
        if method.family == "spherical":
            params["m"] = method.directions
    oracle_class = (
        FirstOrderOracle if method.family == "first_order" else ZeroOrderOracle
    )
    calls = []
    original = oracle_class.query

    def query(self, point):
        result = original(self, point)
        if method.family != "first_order":
            assert result.grad is None
        calls.append(result)
        return result

    monkeypatch.setattr(oracle_class, "query", query)
    case = paper.ZeroOrderCase(
        "regression",
        3,
        3.0,
        reference["kind"],
        observation_noise=reference["noise"],
        drift_per_query=0.01,
        amplitude=1.0,
        period=17,
    )
    curve, metrics = paper._run_zero_order_method(
        method,
        reference["protocol"],
        case,
        72,
        np.linspace(0, 72, 13),
        123,
        paper.ZeroOrderHyperparameters(params),
    )
    # v1 numbers belong to the frozen archive. v2 changes noise streams and
    # query-grid metrics deliberately. Keep the old scenario coverage, and check
    # exact replay plus independence from rendering density. Independent metric
    # arithmetic is checked in test_reproducibility_v2.py.
    calls.clear()
    dense, repeated = paper._run_zero_order_method(
        method,
        reference["protocol"],
        case,
        72,
        np.arange(73),
        123,
        paper.ZeroOrderHyperparameters(params),
    )
    np.testing.assert_array_equal(curve, dense[::6])
    for key in metrics:
        np.testing.assert_equal(metrics[key], repeated[key])
    cost = method.query_cost(3)
    assert metrics["queries"] == len(calls) == 72
    assert metrics["unused_queries"] == 0
    if method.family == "first_order":
        assert (
            metrics["gradient_queries"] == len(calls) and metrics["value_queries"] == 0
        )
    else:
        assert (
            metrics["value_queries"] == len(calls) and metrics["gradient_queries"] == 0
        )


def test_first_order_trace_really_applies_library_noise(monkeypatch):
    class Inspect:
        def initialize(self, x0, rng, task_spec):
            self.x = x0.copy()

        def advance(self, client):
            from wind_benchmark.algorithms.protocol import AdvanceResult

            obs = client.query(self.x)
            assert obs.value is None and not hasattr(obs, "theta")
            np.testing.assert_array_equal(obs.grad, [7.0])
            return AdvanceResult(obs.x, True, 1)

    monkeypatch.setattr(GaussianNoise, "apply", lambda self, signal, t: signal + 7)
    env = DynamicEnvironment(1, LinearDrift(np.zeros(1)), QuadraticLandscape(1))
    records, counts = paper._first_order_trace(
        env, Inspect(), np.zeros(1), 2, GaussianNoise(0.1, 4)
    )
    assert len(records) == counts["gradient_queries"] == 2
    assert all(record.clean_value == 0 for record in records)


def test_main_defers_every_export_until_all_studies_complete(tmp_path, monkeypatch):
    completed = []
    exports = []

    def fake_study(name):
        def run(profile, **kwargs):
            assert not exports
            completed.append(name)
            return object()

        return run

    for name in paper.STUDY_NAMES:
        function_name = (
            "run_zero_order_study" if name == "zeroth" else f"run_{name}_study"
        )
        monkeypatch.setattr(paper, function_name, fake_study(name))

    def export(name, *args, **kwargs):
        assert completed == list(paper.STUDY_NAMES)
        exports.append(name)
        return {}

    monkeypatch.setattr(paper, "_write_study", export)
    assert (
        paper.main(
            ["--output-dir", str(tmp_path), "--run-name", "deferred", "--no-plots"]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "deferred/manifest.json").read_text())
    assert manifest["execution_backend"] == "library_oracle_session_v1"
    zo = {item["name"]: item for item in manifest["zero_order_optimizers"]}
    assert len(zo) == 10
    assert zo["SPSA"]["implementation_version"] == 2
    assert zo["KieferWolfowitz"]["implementation_version"] == 1
    assert zo["RandomSearch"]["rng_stream_id"] == "paper:random_search:fresh:v1"
    assert "core/execution.py" in manifest["source_sha256"]
    assert exports == completed
