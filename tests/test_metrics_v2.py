"""Independent mathematical examples for metrics_v2 (not legacy snapshots)."""

import io
import json
import random
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from wind_benchmark.core.runner import BenchmarkRunner, BatchRunner
from wind_benchmark.core.environment import (
    DynamicEnvironment,
    StationaryDrift,
    QuadraticLandscape,
)
from wind_benchmark.algorithms.optimizers import SGD
from wind_benchmark.algorithms.catalog import create_optimizer
from wind_benchmark.core.oracle import (
    FirstOrderOracle,
    ZeroOrderOracle,
    ScheduledOracle,
    Observation,
)
from wind_benchmark.core.metrics import (
    TrackingErrorMetric,
    MaxCoordinateErrorMetric,
    StiefelFrameMetric,
    GrassmannPrincipalAngleMetric,
    StiefelGeodesicMetric,
    LyapunovMetric,
    NormalizedLyapunovMetric,
    AsymptoticBoundMetric,
    InstantaneousLossMetric,
    DynamicRegretMetric,
    DriftAdaptationMetric,
    TimeToRecoveryMetric,
    AdaptivityMetric,
    QueryEfficiencyMetric,
    MetricsCollection,
    ensemble_metrics,
)
from wind_benchmark.core.geometry import canonical_stiefel_distance
from wind_benchmark.core.privacy import EvaluationSession, GroundTruthAccessError
from wind_benchmark.reporting.common import safe_print


def env():
    return DynamicEnvironment(2, StationaryDrift(), QuadraticLandscape(2, seed=5))


def runner(metrics, oracle=None, trajectory=False):
    task = env() if oracle is None else oracle.environment
    return BenchmarkRunner(
        task,
        oracle or FirstOrderOracle(task),
        MetricsCollection(metrics),
        record_trajectory=trajectory,
    )


def feed(metric, errors, targets=None):
    for t, error in enumerate(errors):
        theta = np.array(targets[t] if targets is not None else [0.0, 0.0])
        point = theta + np.array(error)
        metric.update(
            t, point, theta, Observation(point, t, None, None, query_index=t + 1), None
        )
    return metric


@pytest.mark.parametrize("norm,p", [("l1", 1), ("l2", 2), (1.5, 1.5), ("linf", np.inf)])
def test_normalization_matches_lp_definition(norm, p):
    m = feed(TrackingErrorMetric(norm, normalize_by_dim=True), [[3.0, 4.0]])
    expected = (3**p + 4**p) ** (1 / p) / 2 ** (1 / p) if np.isfinite(p) else 4.0
    assert m.get_result() == pytest.approx(expected)


def test_mahalanobis_spd_and_reset():
    for matrix in ([[1, 2], [0, 1]], [[1, 0], [0, -1]], [[1, 0], [0, np.nan]]):
        with pytest.raises(ValueError):
            TrackingErrorMetric("mahalanobis", matrix)
    m = TrackingErrorMetric("mahalanobis", [[4.0, 0.0], [0.0, 9.0]], True)
    feed(m, [[3.0, 4.0]])
    assert m.get_result() == pytest.approx(np.sqrt(180 / 2))
    automatic = TrackingErrorMetric("mahalanobis")
    for scale in (2.0, 5.0):
        automatic.reset()
        task = SimpleNamespace(landscape=SimpleNamespace(A=np.eye(2) * scale))
        automatic.update(0, np.ones(2), np.zeros(2), None, task)
        assert automatic.get_result() == pytest.approx(np.sqrt(2 * scale))
    automatic.reset()
    with pytest.raises(ValueError, match="requires"):
        feed(automatic, [[1.0, 1.0]])


def test_tail_applied_once_and_max_coordinate_mean():
    m = feed(MaxCoordinateErrorMetric(), [[1, 0], [10, 0], [2, 0], [6, 0]])
    assert m.get_result(1) == 4.75
    assert m.get_result(0.5) == 4.0
    peak = feed(MaxCoordinateErrorMetric("max"), [[1, 0], [10, 0], [2, 0], [6, 0]])
    assert peak.get_result(1) == 10.0
    for fraction in (0, -1, 1.1, np.inf, np.nan):
        with pytest.raises(ValueError):
            m.get_result(fraction)


@pytest.mark.parametrize("rho", [0, -1, 1.01, np.inf, np.nan])
def test_invalid_lyapunov_exponent(rho):
    with pytest.raises(ValueError):
        LyapunovMetric(rho)


def test_lyapunov_values_and_exact_identities():
    raw = feed(LyapunovMetric(0.5), [[4.0, 9.0]])
    assert raw.get_result() == pytest.approx(35.0)
    norm = feed(NormalizedLyapunovMetric(0.5, 4), [[4.0, 9.0]])
    assert norm.get_result() == pytest.approx(35 / 8)
    for A in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            NormalizedLyapunovMetric(1.0, A)
    a, b = NormalizedLyapunovMetric(1, 0.10001), NormalizedLyapunovMetric(1, 0.10002)
    collection = MetricsCollection([a, b])
    assert len(collection.get_definitions()) == 3  # raw V plus two normalizations
    assert a.metric_id != b.metric_id
    with pytest.raises(ValueError, match="Duplicate"):
        MetricsCollection([TrackingErrorMetric(), TrackingErrorMetric()])
    assert (
        len(
            MetricsCollection(
                [QueryEfficiencyMetric(0.1), QueryEfficiencyMetric(0.2)]
            ).get_results()
        )
        == 2
    )


def test_regret_is_full_sum_normalization_additional_and_loss_is_clean():
    task = SimpleNamespace(
        landscape=SimpleNamespace(
            loss=lambda x, theta: float(np.sum((x - theta) ** 2) + 7)
        )
    )
    regret = DynamicRegretMetric(True)
    loss = InstantaneousLossMetric()
    collection = MetricsCollection([regret, loss])
    for t, (x, theta) in enumerate(
        (([2.0, 0.0], [0.0, 0.0]), ([4.0, 0.0], [1.0, 0.0]))
    ):
        collection.update(t, np.array(x), np.array(theta), None, task)
    values = collection.get_results(0.01)
    assert values["dynamic_regret"] == 13
    assert values["dynamic_regret_path_normalized"] == 13
    assert values["instant_loss"] == 16
    assert collection.get_histories()["instantaneous_regret"] == [4.0, 9.0]
    collection.reset()
    collection.update(0, np.ones(2), np.zeros(2), None, task)
    assert collection.get_results()["dynamic_regret"] == 2
    assert np.isnan(collection.get_results()["dynamic_regret_path_normalized"])
    assert (
        collection.get_status()["dynamic_regret_path_normalized"]["state"]
        == "zero_path_variation"
    )


def test_drift_cosine_excludes_zero_motion():
    metric = DriftAdaptationMetric()
    points = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    targets = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    for t, (x, theta) in enumerate(zip(points, targets)):
        metric.update(t, np.array(x), np.array(theta), None, None)
    assert metric.get_result(1) == 0
    assert metric.status(1)["n_valid"] == 2
    metric.reset()
    metric.update(0, np.ones(2), np.zeros(2), None, None)
    assert np.isnan(metric.get_result())


def test_parameterized_regret_keeps_all_derived_identities():
    collection = MetricsCollection([DynamicRegretMetric(), DynamicRegretMetric(True)])
    task = SimpleNamespace(
        landscape=SimpleNamespace(loss=lambda x, theta: float(np.sum((x - theta) ** 2)))
    )
    collection.update(0, np.ones(2), np.zeros(2), None, task)
    definitions = collection.get_definitions()
    histories = collection.get_histories()
    assert len(histories) == len(definitions) == len(collection.get_status()) == 5
    assert len({definition["id"] for definition in definitions.values()}) == 5
    assert len(collection.get_results()) == 3
    for name in histories:
        assert name in definitions


def test_shared_recovery_events_immediate_censored_and_new_jump():
    # Jumps at 1,3,5: immediate recovery, recovery after one step, never recovered.
    targets = [[0, 0], [2, 0], [2, 0], [4, 0], [4, 0], [6, 0], [6, 0]]
    errors = [[1, 0], [0, 0], [1, 0], [1, 0], [0, 0], [1, 0], [1, 0]]
    ttr = feed(TimeToRecoveryMetric(1, 0.1), errors, targets)
    adapt = feed(AdaptivityMetric(1, 0.1, oracle_ttr=0.5), errors, targets)
    assert ttr.get_result() == 1  # mean(0,1,2), not successful-only .5
    assert ttr.status()["recovery_fraction"] == pytest.approx(2 / 3)
    assert adapt.get_result() == 0.75  # mean(1,.5), explicitly conditional
    assert adapt.status()["n_censored"] == 1
    assert [e["window"] for e in ttr.status()["events"]] == [2, 2, 2]
    interrupted = feed(
        TimeToRecoveryMetric(1, 0.1), [[1, 0], [1, 0], [0, 0]], [[0, 0], [2, 0], [4, 0]]
    )
    assert interrupted.status()["n_censored"] == 1
    assert interrupted.get_result() == 0.5
    ttr.reset()
    feed(ttr, [[1, 0], [0, 0]])
    assert np.isnan(ttr.get_result())
    assert ttr.status()["state"] == "no_events"


def test_threshold_initial_and_auxiliary_query_index():
    class ExtraQueries:
        query_cost_per_step = 3

        def initialize(self, x0, rng, task_spec):
            self.x = x0.copy()

        def advance(self, client):
            from wind_benchmark.algorithms.protocol import AdvanceResult

            obs = client.query(self.x)
            client.query(obs.x + 3)
            client.query(obs.x + 4)
            self.x = np.zeros(2)
            return AdvanceResult(self.x, True, 3)

    r = runner([QueryEfficiencyMetric()]).run(ExtraQueries(), 3, np.ones(2), seed=1)
    assert r.final_metrics["query_efficiency"] == 4  # not 6 after auxiliaries
    assert r.metadata["query_counts"]["total"] == 9
    assert r.metric_status["query_efficiency"]["n_valid"] == 10
    initial = runner([QueryEfficiencyMetric()]).run(SGD(), 2, np.zeros(2), seed=1)
    assert initial.final_metrics["query_efficiency"] == 0
    miss = runner([QueryEfficiencyMetric()]).run(SGD(lr=1e-15), 2, np.ones(2), seed=1)
    assert miss.metric_status["query_efficiency"]["restricted_queries"] == 2
    assert miss.to_dict()["final_metrics"]["query_efficiency"] is None


def test_threshold_zero_query_result_and_exact_observation_counts():
    metric = QueryEfficiencyMetric()
    assert metric.status()["n_total"] == 0
    assert metric.status()["state"] == "no_observations"
    metric.initial(np.zeros(2), np.zeros(2), None)
    assert metric.get_result() == 0
    assert metric.status()["n_total"] == metric.status()["n_valid"] == 1
    metric.reset()
    feed(metric, [[1.0, 1.0]])  # No initial observation was supplied.
    assert metric.status()["n_total"] == metric.status()["n_valid"] == 1


def test_query_ensemble_rejects_partial_or_heterogeneous_results():
    batch = BatchRunner(
        lambda seed: env(),
        lambda task: FirstOrderOracle(task),
        lambda: MetricsCollection([QueryEfficiencyMetric()]),
    )
    results = batch.run(lambda: SGD(), seeds=[12, 13], T=2)
    assert ensemble_metrics(results)["query_efficiency"]["state"] == "ok"
    partial = ensemble_metrics(results[:1])["query_efficiency"]
    assert partial["state"] == "incomplete_ensemble"
    assert partial["restricted_mean"] is None
    different = runner([TrackingErrorMetric()]).run(SGD(), 2, np.ones(2), seed=14)
    mixed = ensemble_metrics([results[0], different])["query_efficiency"]
    assert mixed["state"] == "incomparable_protocol_or_budget"
    assert mixed["restricted_mean"] is None
    assert mixed["n_valid"] == 1


def test_ensemble_mean_then_max_and_incomplete_series():
    batch = BatchRunner(
        lambda seed: env(),
        lambda task: FirstOrderOracle(task),
        lambda: MetricsCollection([AsymptoticBoundMetric()]),
    )
    results = batch.run(lambda: SGD(), seeds=[12, 13], T=2)
    name = next(iter(results[0].metrics_history))
    for result, trace in zip(results, ([10.0, 0.0], [0.0, 10.0])):
        result.metrics_history[name] = trace
        result.metadata["tail_fraction"] = 1.0
    assert np.isnan(results[0].final_metrics[name])
    summary = ensemble_metrics(results)[name]
    assert summary["state"] == "finite_horizon_estimate"
    assert summary["value"] == 5  # mean(max each) would incorrectly be 10
    assert ensemble_metrics(results[:1])[name]["value"] is None
    results[1].metadata["seed"] = 12
    assert ensemble_metrics(results)[name]["value"] is None


def query_ensemble(method="SPSA", parameters=None, tail=0.5):
    batch = BatchRunner(
        lambda seed: env(),
        FirstOrderOracle if method == "SGD" else ZeroOrderOracle,
        lambda: MetricsCollection([AsymptoticBoundMetric()]),
        record_trajectory=True,
        tail_fraction=tail,
    )
    results = batch.run(
        lambda: create_optimizer(method, parameters),
        seeds=[12, 13],
        T=2,
        initial_x=np.ones(2),
    )
    return batch, results, next(iter(results[0].metrics_history))


@pytest.mark.parametrize(
    "method,parameters,queries",
    [
        ("SGD", {}, 2),
        ("SPSA", {}, 4),
        ("KieferWolfowitz", {}, 8),
        ("SphericalZOSGD", {"m": 3}, 12),
    ],
)
@pytest.mark.parametrize("tail", [1.0, 0.5])
def test_ensemble_uses_query_grid_not_update_horizon(method, parameters, queries, tail):
    batch, results, name = query_ensemble(method, parameters, tail)
    before = [result.to_dict() for result in results]
    for result in results:
        assert result.status == "SUCCESS"
        assert result.metadata["total_steps"] == 2
        assert result.metadata["time_grid"] == {
            "kind": "query_index",
            "start": 1,
            "step": 1,
            "size": queries,
        }
        assert len(result.metrics_history[name]) == queries

    # Independent squared-distance arithmetic on the recorded queried points.
    per_query_means = []
    for records in zip(*(result.evaluation_log for result in results)):
        per_query_means.append(
            sum(
                sum(
                    (float(x) - float(theta)) ** 2
                    for x, theta in zip(record.observation.x, record.theta)
                )
                for record in records
            )
            / len(results)
        )
    expected = max(per_query_means[int(queries * (1 - tail)) :])
    summary = batch.summarize(results)[name]
    assert summary["state"] == "finite_horizon_estimate"
    assert summary["value"] == pytest.approx(expected, abs=1e-12, rel=1e-10)
    assert (summary["N"], summary["T"], summary["gamma"]) == (2, queries, tail)
    assert not summary["certified_bound"]
    assert [result.to_dict() for result in results] == before


def test_query_grid_ensemble_applies_tail_once_after_seed_mean():
    batch, results, name = query_ensemble()
    for result, values in zip(results, ([100, 0, 8, 0], [0, 100, 0, 4])):
        result.metrics_history[name] = values
    # Tail means are [4, 2], not individual maxima [8, 4] or the last value 2.
    assert batch.summarize(results)[name]["value"] == 4


@pytest.mark.parametrize("length", [0, 2, 3, 5])
def test_query_grid_ensemble_rejects_incomplete_or_excess_histories(length):
    batch, results, name = query_ensemble()
    for result in results:
        result.metrics_history[name] = [1.0] * length
    summary = batch.summarize(results)[name]
    assert summary["state"] == "incomplete_ensemble"
    assert summary["value"] is None


@pytest.mark.parametrize(
    "grid",
    [
        None,
        {},
        [],
        {"size": 0},
        {"size": -1},
        {"size": True},
        {"size": 4.0},
        {"size": "4"},
    ],
)
def test_query_grid_ensemble_rejects_invalid_declared_size(grid):
    batch, results, name = query_ensemble()
    for result in results:
        result.metadata["time_grid"] = grid
    assert batch.summarize(results)[name]["state"] == "incomplete_ensemble"


@pytest.mark.parametrize("field,value", [("start", 2), ("step", 2), ("size", 5)])
def test_query_grid_ensemble_rejects_misaligned_replicates(field, value):
    batch, results, name = query_ensemble()
    results[1].metadata["time_grid"][field] = value
    assert batch.summarize(results)[name]["value"] is None


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "matrix", "failed"])
def test_query_grid_ensemble_requires_complete_finite_scalar_histories(problem):
    batch, results, name = query_ensemble()
    if problem == "missing":
        del results[1].metrics_history[name]
    elif problem == "nonfinite":
        results[1].metrics_history[name][0] = np.nan
    elif problem == "matrix":
        results[1].metrics_history[name] = [[1.0]] * 4
    else:
        results[1].status = "ERROR"
    summary = batch.summarize(results)[name]
    assert summary["state"] == "incomplete_ensemble"
    assert summary["value"] is None


def test_ensemble_without_declared_grid_keeps_step_history_contract():
    batch, results, name = query_ensemble("SGD", tail=1.0)
    for result, values in zip(results, ([10.0, 0.0], [0.0, 10.0])):
        del result.metadata["time_grid"]
        result.metrics_history[name] = values
    assert batch.summarize(results)[name]["value"] == 5.0


def test_workbench_exports_multi_query_ensemble_on_query_grid(tmp_path):
    from wind_benchmark.interfaces.web_runner import run_workbench

    config = {
        "schema_version": 3,
        "environment": {
            "dim": 2,
            "landscape": {"type": "quadratic", "seed": 5},
            "drift": {"type": "stationary"},
            "initial_theta": [0.0, 0.0],
        },
        "oracle": {"type": "zero-order"},
        "optimizer": {"name": "SPSA", "params": {}},
        "runner": {
            "steps": 2,
            "seeds": [12, 13],
            "tail_fraction": 0.5,
            "initial_x": [1.0, 1.0],
            "output_dir": "web",
        },
        "metrics": ["asymptotic_bound"],
    }
    summary = run_workbench(config, tmp_path)
    name = AsymptoticBoundMetric().name
    row = summary["ensemble_metrics"]["SPSA"][name]
    assert row["state"] == "finite_horizon_estimate"
    assert row["T"] == 4
    histories = [
        json.loads((tmp_path / "web" / f"SPSA_seed{seed}.json").read_text())[
            "metrics_history"
        ][name]
        for seed in [12, 13]
    ]
    assert row["value"] == pytest.approx(
        max((a + b) / 2 for a, b in zip(histories[0][2:], histories[1][2:]))
    )

    def reject_nonfinite(value):
        raise AssertionError(f"Non-standard JSON constant: {value}")

    saved = json.loads(
        (tmp_path / "web" / "workbench_summary.json").read_text(),
        parse_constant=reject_nonfinite,
    )
    assert saved["ensemble_metrics"]["SPSA"][name] == row


def test_geometries_are_distinct_and_inputs_validated():
    X = np.eye(4)[:, :2]
    angle = 0.7
    Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Y = X @ Q
    metrics = [
        StiefelFrameMetric(4, 2),
        GrassmannPrincipalAngleMetric(4, 2),
        StiefelGeodesicMetric(4, 2),
    ]
    for metric in metrics:
        metric.update(0, X.ravel(), Y.ravel(), None, None)
    assert metrics[0].get_result() == pytest.approx(2 * np.sqrt(2) * np.sin(angle / 2))
    assert metrics[1].get_result() == pytest.approx(0, abs=3e-8)
    assert metrics[2].get_result() == pytest.approx(angle, abs=1e-8)
    with pytest.raises(ValueError):
        metrics[1].update(1, np.zeros(8), X.ravel(), None, None)
    value, diag = canonical_stiefel_distance(X, -X)
    assert np.isnan(value)
    assert diag["state"] != "converged"


@pytest.mark.parametrize("d,r", [(3, 2), (4, 2), (5, 3), (2, 2)])
def test_numerical_stiefel_known_small_canonical_geodesic(d, r):
    X = np.eye(d)[:, :r]
    A = np.zeros((r, r))
    A[0, 1] = -0.08
    A[1, 0] = 0.08
    B = np.full((d - r, r), 0.03)
    generator = np.block([[A, -B.T], [B, np.zeros((d - r, d - r))]])
    Y = expm(generator)[:, :r]
    expected = np.sqrt(0.5 * np.sum(A * A) + np.sum(B * B))
    value, diag = canonical_stiefel_distance(X, Y)
    reverse, _ = canonical_stiefel_distance(Y, X)
    assert value == pytest.approx(expected, abs=2e-7)
    assert reverse == pytest.approx(value, abs=2e-7)
    if d - r > 1:
        # Generic rank-two normal component requires iterations, unlike rank one.
        rng = np.random.default_rng(5)
        B = rng.normal(size=(d - r, r)) * 0.3
        Y = expm(np.block([[A, -B.T], [B, np.zeros((d - r, d - r))]]))[:, :r]
        value, diag = canonical_stiefel_distance(X, Y, max_iterations=1)
        assert np.isnan(value)


def test_reset_matches_fresh_for_all_accumulators():
    factories = [
        TrackingErrorMetric,
        MaxCoordinateErrorMetric,
        LyapunovMetric,
        NormalizedLyapunovMetric,
        AsymptoticBoundMetric,
        DriftAdaptationMetric,
        TimeToRecoveryMetric,
        AdaptivityMetric,
        QueryEfficiencyMetric,
    ]
    for factory in factories:
        reused = feed(factory(), [[1.0, 2.0], [3.0, 4.0]])
        reused.reset()
        feed(reused, [[5.0, 6.0]])
        fresh = feed(factory(), [[5.0, 6.0]])
        np.testing.assert_equal(reused.get_history(), fresh.get_history())
        np.testing.assert_equal(reused.get_result(), fresh.get_result())


def test_json_no_trajectory_csv_real_metrics(tmp_path):
    m = [
        TrackingErrorMetric("l1"),
        InstantaneousLossMetric(),
        DynamicRegretMetric(True),
    ]
    without = runner(m).run(SGD(), 3, np.ones(2), seed=1)
    path = tmp_path / "without.json"
    without.save_to_json(path)
    payload = json.loads(path.read_text())
    assert payload["_schema_version"] == 4
    assert payload["trajectory"] is None
    assert "NaN" not in path.read_text()
    recorded = runner(m, trajectory=True).run(SGD(), 3, np.ones(2), seed=1)
    csv = tmp_path / "all.csv"
    recorded.save_to_csv(csv)
    frame = pd.read_csv(csv)
    for metric, history in recorded.metrics_history.items():
        np.testing.assert_allclose(
            frame.loc[frame.metric_name == metric, "metric_value"], history
        )


def test_rng_restored_success_and_failure():
    class Failing(SGD):
        def step(self, obs):
            np.random.random()
            random.random()
            raise ValueError("expected")

    for optimizer in (SGD(), Failing()):
        np.random.seed(27)
        random.seed(28)
        before = np.random.get_state()
        py_before = random.getstate()
        try:
            runner([TrackingErrorMetric()]).run(optimizer, 3, np.ones(2), seed=90)
        except ValueError:
            pass
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
        assert py_before == random.getstate()


def test_scheduled_repeated_run_same_trace():
    task = env()
    oracle = ScheduledOracle(task, schedule=[("first-order", 1), ("zero-order", 2)])

    class Hold:
        query_cost_per_step = 1

        def initialize(self, x0, rng, task_spec):
            self.x = x0.copy()

        def advance(self, client):
            from wind_benchmark.algorithms.protocol import AdvanceResult

            client.query(self.x)
            return AdvanceResult(self.x, True, 1)

    engine = runner([TrackingErrorMetric()], oracle=oracle, trajectory=True)
    a = engine.run(Hold(), 7, np.ones(2), seed=1)
    b = engine.run(Hold(), 7, np.ones(2), seed=1)
    assert [r.observation.mode for r in a.evaluation_log] == [
        r.observation.mode for r in b.evaluation_log
    ]
    assert a.metadata["query_counts"] == b.metadata["query_counts"]
    assert a.metadata["query_counts"]["gradient"] == 1


def test_limited_console_encoding():
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="ascii")
    safe_print("✓ ρ = 1", file=stream)
    stream.flush()
    assert b"1" in buffer.getvalue()


def test_gym_abort_double_close_never_releases():
    pytest.importorskip("gymnasium")
    from wind_benchmark.integrations.gym_env import WindGymEnv

    task = env()
    gym = WindGymEnv(task, T=4)
    gym.reset(seed=1)
    gym.close()
    gym.close()
    with pytest.raises(GroundTruthAccessError):
        task.get_current_theta(for_analysis=True)


def test_export_failure_does_not_reclassify_completed_series(tmp_path, monkeypatch):
    from wind_benchmark.interfaces.web_runner import run_workbench

    config = {
        "schema_version": 3,
        "environment": {
            "dim": 2,
            "landscape": {"type": "quadratic"},
            "drift": {"type": "stationary"},
        },
        "oracle": {"type": "first-order", "blind_value": True},
        "optimizer": {"name": "SGD", "params": {}},
        "runner": {
            "steps": 2,
            "seeds": [1],
            "tail_fraction": 0.2,
            "output_dir": "output",
        },
        "metrics": ["tracking_error"],
    }

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "wind_benchmark.core.runner.ExperimentResult.save_to_json", fail
    )
    series = EvaluationSession(expected_runs=1)
    with pytest.raises(OSError, match="disk full"):
        with series:
            pending = run_workbench(config, tmp_path)
    assert series.state == "released"
    assert series.export_status == "failed"
    assert pending.result()["runs"][0]["status"] == "SUCCESS"


def test_constructor_failure_has_no_phantom_active_session():
    from wind_benchmark.core.execution import OracleSession
    from wind_benchmark.core.oracle import ZeroOrderOracle

    with EvaluationSession(expected_runs=1):
        with pytest.raises(ValueError):
            OracleSession(ZeroOrderOracle(env()), budget=0)
        session = OracleSession(ZeroOrderOracle(env()), budget=1)
        session.observe(np.zeros(2))
        records = session.finish()
        with pytest.raises(GroundTruthAccessError):
            session.abort()
    assert len(records) == 1


def test_frozen_transition_failure_aborts_and_closes_client(monkeypatch):
    from wind_benchmark.core.execution import OracleSession
    from wind_benchmark.core.oracle import ZeroOrderOracle

    task = env()
    session = OracleSession(ZeroOrderOracle(task), budget=1, protocol="frozen")

    def fail(self, *args, **kwargs):
        raise RuntimeError("transition failed")

    monkeypatch.setattr(DynamicEnvironment, "step", fail)
    with pytest.raises(RuntimeError, match="transition failed"):
        with session.client() as client:
            client.query(np.zeros(2))
    with pytest.raises(RuntimeError):
        client.query(np.zeros(2))
    with pytest.raises(GroundTruthAccessError):
        session.finish()
    with pytest.raises(GroundTruthAccessError):
        task.get_current_theta(True)


def test_catalog_does_not_import_legacy_orchestrator():
    import subprocess
    import sys

    command = 'import wind_benchmark.algorithms.catalog; import sys; assert "wind_benchmark.experiments.configurable" not in sys.modules'
    subprocess.run([sys.executable, "-B", "-c", command], check=True)


def test_environment_oracle_registration_does_not_retain_large_task_cycles():
    enabled = gc.isenabled()
    gc.disable()
    try:
        task = env()
        oracle = FirstOrderOracle(task)
        task_ref, oracle_ref = weakref.ref(task), weakref.ref(oracle)
        oracle.n_queries = 3
        task.reset()
        assert oracle.n_queries == 0
        assert task._registered_oracle is oracle
        del task, oracle
        assert task_ref() is None
        assert oracle_ref() is None
        task = env()
        oracle = FirstOrderOracle(task)
        del oracle
        assert task._registered_oracle is None
        task.reset()
    finally:
        if enabled:
            gc.enable()
        gc.collect()
