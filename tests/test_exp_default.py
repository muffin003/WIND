import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wind_benchmark.experiments.standard import (
    Method,
    Profile,
    Task,
    catalog_method,
    default_tasks,
    evaluate_records,
    feasibility,
    main,
    mean_interval,
    noise_parameters,
    run_one,
    run_suite,
    summarize,
    task_components,
    wilson_interval,
)
from wind_benchmark.core.oracle import EvaluationRecord, Observation
from wind_benchmark.algorithms.protocol import AdvanceResult


@pytest.fixture
def profile():
    return Profile("test", 40, (11, 22, 33), 100, 9)


def test_registry_covers_agreed_axes_and_matched_geometry_controls():
    tasks = default_tasks()
    assert len(tasks) == 32
    assert len({t.name for t in tasks}) == len(tasks)
    assert {t.noise for t in tasks} == {
        "none",
        "gaussian",
        "correlated",
        "heavy_tailed",
        "sparse",
        "multiplicative",
        "quantized",
    }
    assert {"quadratic", "pnorm", "rosenbrock", "multiextremal"} <= {
        t.landscape for t in tasks
    }
    for geometry in ("simplex", "stiefel", "grassmann"):
        assert len([t for t in tasks if t.geometry == geometry]) == 3
    assert Profile().budget == 4000
    assert len(Profile().seeds) == 50


def test_profile_validation(profile):
    for invalid in (
        replace(profile, seeds=(1, 1)),
        replace(profile, seeds=(1,)),
        replace(profile, budget=0),
        replace(profile, seeds=(-1, 2)),
    ):
        with pytest.raises(ValueError):
            invalid.validate()


@pytest.mark.parametrize("geometry", ["simplex", "stiefel", "grassmann"])
def test_constrained_targets_and_decisions_stay_feasible(profile, geometry):
    task = next(t for t in default_tasks() if t.name == f"{geometry}_moving_correlated")
    env, x, _ = task_components(task, profile, 11)
    assert feasibility(task, x) < 1e-12
    for _ in range(profile.budget):
        assert feasibility(task, env.get_current_theta(True)) < 1e-10
        env.step()
    method = catalog_method(
        "ExponentiatedGradient" if geometry == "simplex" else "RiemannianSGD"
    )
    row, _, _, _ = run_one(method, task, profile, 11)
    assert row["status"] == "SUCCESS"
    assert row["max_constraint_violation"] < 1e-10


def test_scaling_preserves_initial_norm_drift_norm_and_noise_power(profile):
    for dim in (10, 30, 100):
        task = Task("scale", "scaling", drift="linear", noise="gaussian", dim=dim)
        env, x, _ = task_components(task, profile, 11)
        theta = env.get_current_theta(True)
        assert np.linalg.norm(x - theta) == pytest.approx(1)
        env.step()
        assert np.linalg.norm(env.get_current_theta(True) - theta) == pytest.approx(
            0.005
        )
        assert noise_parameters(task, 11, gradient=True)[
            "sigma"
        ] ** 2 * dim == pytest.approx(0.01)


def test_exact_clean_sgd_metrics(profile):
    row, curves, _, _ = run_one(
        catalog_method("SGD", {"lr": 0.1}), Task("clean", "reference"), profile, 11
    )
    expected = 0.9 ** (2 * np.arange(profile.budget))
    assert row["status"] == "SUCCESS"
    assert row["queries"] == row["gradient_queries"] == profile.budget
    assert row["value_queries"] == 0
    assert row["mean_error"] == pytest.approx(expected.mean())
    assert row["tail_error"] == pytest.approx(expected[32:].mean())
    assert row["dynamic_regret"] == pytest.approx(expected.sum() / 2)
    for point in curves:
        assert point["squared_error"] == pytest.approx(expected[point["query"] - 1])


@pytest.mark.parametrize("oracle", ["first-order", "zero-order", "hybrid"])
def test_same_external_streams_are_paired_and_oracle_fields_isolated(profile, oracle):
    seen = []

    class Hold:
        def initialize(self, x0, rng, task_spec):
            self.x = x0.copy()

        def advance(self, client):
            obs = client.query(self.x)
            assert not hasattr(obs, "theta")
            assert not hasattr(obs, "optimum_value")
            assert (obs.value is None) == (oracle == "first-order")
            assert (obs.grad is None) == (oracle == "zero-order")
            seen.append((obs.value, None if obs.grad is None else obs.grad.copy()))
            return AdvanceResult(obs.x, True, 1)

    task = Task("paired", "interaction", noise="correlated", drift="random_walk")
    first = run_one(
        Method("A", oracle, ("euclidean",), lambda t, s: Hold()), task, profile, 11
    )
    split = len(seen)
    second = run_one(
        Method("B", oracle, ("euclidean",), lambda t, s: Hold()), task, profile, 11
    )
    assert first[0]["score"] == second[0]["score"]
    for a, b in zip(seen[:split], seen[split:]):
        assert a[0] == b[0]
        if a[1] is not None:
            np.testing.assert_array_equal(a[1], b[1])
    assert first[0]["queries"] == profile.budget
    assert first[0]["value_queries"] == (
        0 if oracle == "first-order" else profile.budget
    )
    assert first[0]["gradient_queries"] == (
        0 if oracle == "zero-order" else profile.budget
    )


def test_multiquery_streaming_cap_and_client_lifecycle(profile):
    clients, times, values = [], [], []

    class Multi:
        def initialize(self, x0, rng, task_spec):
            self.x = x0.copy()

        def advance(self, client):
            consumed = 1
            obs = client.query(self.x)
            clients.append(client)
            assert not hasattr(client, "environment")
            times.append(obs.t)
            values.append(obs.value)
            for _ in range(6):
                if client.remaining == 0:
                    return AdvanceResult(obs.x, False, consumed)
                extra = client.query(obs.x)
                consumed += 1
                times.append(extra.t)
                values.append(extra.value)
            return AdvanceResult(obs.x, True, consumed)

    method = Method("multi", "zero-order", ("euclidean",), lambda t, s: Multi())
    row, _, _, _ = run_one(
        method, Task("linear", "dynamics", drift="linear"), profile, 11
    )
    assert row["status"] == "SUCCESS"
    assert row["queries"] == profile.budget
    assert times == list(range(profile.budget))
    assert len(set(values)) > 1  # additional queries do not freeze the target
    for client in clients:
        with pytest.raises(RuntimeError, match="only available"):
            client.query(np.zeros(10))


def test_failures_not_silently_dropped_and_unsupported_geometry_skipped(profile):
    class Broken:
        def initialize(self, x0, rng, task_spec):
            self.x = x0.copy()

        def advance(self, client):
            obs = client.query(self.x)
            return AdvanceResult(np.full_like(obs.x, np.nan), True, 1)

    method = Method("broken", "first-order", ("euclidean",), lambda t, s: Broken())
    row, curves, _, _ = run_one(method, Task("clean", "reference"), profile, 11)
    assert row["status"] == "NUMERICAL_FAILURE"
    assert row["queries"] == 1
    assert "score" not in row
    assert curves == []
    row, _, _, _ = run_one(
        catalog_method("SGD"),
        Task("simplex", "geometry", geometry="simplex", landscape="simplex"),
        profile,
        11,
    )
    assert row["status"] == "SKIPPED"
    assert row["queries"] == 0


def test_declaring_geometry_does_not_project_invalid_decisions(profile):
    method = Method(
        "unconstrained",
        "first-order",
        ("grassmann",),
        lambda t, s: catalog_method("SGD").factory(t, s),
    )
    task = next(t for t in default_tasks() if t.geometry == "grassmann")
    row, _, _, _ = run_one(method, task, profile, 11)
    assert row["status"] == "ERROR_INTERFACE"
    assert "score" not in row


def test_jump_nonrecoveries_retained_and_windows_not_independent_samples(profile):
    task = Task("jump", "dynamics", drift="jump")
    records = [
        EvaluationRecord(
            Observation(np.ones(10), q, None, np.zeros(10), q + 1),
            np.zeros(10),
            5.0,
            0.0,
        )
        for q in range(profile.budget)
    ]
    metrics, _, events = evaluate_records(task, profile, records)
    assert len(events) == 4
    assert metrics["recovery_fraction"] == 0
    assert metrics["restricted_recovery_time"] == profile.budget // 10
    assert all(not e["recovered"] for e in events)


def test_paired_bootstrap_preserves_matching_and_reports_missing_pairs(profile):
    rows = []
    for seed in profile.seeds:
        for algorithm, offset in (("SGD", 0), ("B", 2)):
            rows.append(
                dict(
                    task="t",
                    algorithm=algorithm,
                    oracle="first-order",
                    geometry="euclidean",
                    seed=seed,
                    primary_metric="tail_error",
                    status="SUCCESS",
                    score=seed + offset,
                )
            )
    summary, paired = summarize(pd.DataFrame(rows), profile)
    assert paired.iloc[0].mean_difference_success_pairs_only == 2
    assert paired.iloc[0].ci_low == paired.iloc[0].ci_high == 2
    assert paired.iloc[0].n_pairs == 3
    rows[-1]["status"] = "ERROR"
    summary, paired = summarize(pd.DataFrame(rows), profile)
    assert paired.iloc[0].n_excluded_pairs == 1
    assert summary.iloc[1].n_failed == 1
    assert summary.iloc[1].failure_rate == pytest.approx(1 / 3)
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0)
    assert hi > 0  # bootstrap of all-success flags would falsely give [0, 0]
    assert mean_interval([1], 100, 0) == (1, None, None)


def test_end_to_end_deferred_exports_manifest_reproducibility_and_no_overwrite(
    profile, tmp_path
):
    output = tmp_path / "suite"

    class Inspect:
        def initialize(self, x0, rng, task_spec):
            self.x = x0.copy()
            assert {p.name for p in output.iterdir()} <= {
                "recommendations",
                "execution_status.json",
            }

        def advance(self, client):
            obs = client.query(self.x)
            assert {p.name for p in output.iterdir()} <= {
                "recommendations",
                "execution_status.json",
            }
            self.x = obs.x - 0.1 * obs.grad
            return AdvanceResult(self.x, True, 1)

    methods = [
        Method("inspect", "first-order", ("euclidean",), lambda t, s: Inspect()),
        catalog_method("Adam"),
    ]
    tasks = [
        Task("clean", "reference"),
        Task("linear", "dynamics", drift="linear", noise="correlated"),
    ]
    result = run_suite(output, profile, methods, tasks, plots=True, progress=False)
    assert len(result["runs"]) == 12
    assert set(result["runs"].status) == {"SUCCESS"}
    assert (output / "figures/curves_first-order_reference.png").stat().st_size > 1000
    for extension in ("svg", "pdf"):
        assert (output / f"figures/coverage.{extension}").stat().st_size > 1000
    trace_index = pd.read_csv(output / "trace_index.csv")
    with np.load(output / "trajectories.npz", allow_pickle=False) as traces:
        assert len(traces.files) == 12
        assert all(len(traces[key]) == profile.budget for key in trace_index.key)
    bands = pd.read_csv(output / "curve_summary.csv")
    assert len(bands) == 4 * profile.budget
    assert (bands.n_success == 3).all()
    for filename in (
        "runs.csv",
        "summary.csv",
        "metrics_summary.csv",
        "paired_comparisons.csv",
        "curves.csv",
        "recovery_events.csv",
        "report.md",
    ):
        assert (output / filename).is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["resolved_runs"]) == 12
    assert "experiments/standard.py" in manifest["source_sha256"]
    with pytest.raises(FileExistsError):
        run_suite(output, profile, methods, tasks, plots=False)
    again = run_one(catalog_method("Adam"), tasks[1], profile, 11)[0]
    match = (
        result["runs"]
        .query("algorithm == 'Adam' and task == 'linear' and seed == 11")
        .iloc[0]
    )
    assert again["score"] == match.score


def test_all_skipped_and_all_failed_reports_are_valid(profile, tmp_path):
    run_suite(
        tmp_path / "skip",
        profile,
        [catalog_method("ExponentiatedGradient")],
        [Task("clean", "reference")],
        progress=False,
    )

    class Error:
        def initialize(self, x0, rng, task_spec):
            raise RuntimeError("intentional")

        def advance(self, client):
            raise AssertionError("Unreachable")

    run = run_suite(
        tmp_path / "fail",
        profile,
        [Method("bad", "first-order", ("euclidean",), lambda t, s: Error())],
        [Task("clean", "reference")],
        progress=False,
    )
    assert set(run["runs"].status) == {"ERROR"}
    assert run["summary"].iloc[0].n_failed == len(profile.seeds)


def test_dry_run_does_not_create_output(tmp_path, capsys):
    output = tmp_path / "not-created"
    assert main(["--dry-run", "--profile", "smoke", "--output-dir", str(output)]) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["compatible_runs"] == 1347
    assert not output.exists()


def test_bootstrap_curves_preserves_whole_run_dependence():
    from wind_benchmark.reporting.standard import curve_interval

    data = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]])
    mean, low, high = curve_interval(data, 1000, 42)
    np.testing.assert_allclose(mean, data.mean(axis=0))
    np.testing.assert_allclose(low, low[0] * np.arange(1, 4))
    np.testing.assert_allclose(high, high[0] * np.arange(1, 4))
    _, lo, hi = curve_interval(data[:1], 100, 42)
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_tracking_plot_limits_include_intervals_without_negative_padding():
    from wind_benchmark.reporting.standard import plt, tracking_scale

    fig, axes = plt.subplots(1, 2, sharey=True)
    axes[0].plot([10, 100], [0.05, 0.9])
    axes[0].vlines([10, 100], [0.04, 0.8], [0.06, 1.1])
    axes[1].plot([10, 100], [0.2, 0.3])
    tracking_scale(axes)
    for ax in axes:
        lower, upper = ax.get_ylim()
        assert 0 < lower < 0.04 and upper > 1.1
    plt.close(fig)


def test_tracking_plot_limits_handle_zero_and_empty_curves():
    from wind_benchmark.reporting.standard import plt, tracking_scale

    fig, axes = plt.subplots(1, 2)
    axes[0].plot([1, 2], [0.0, 0.0])
    tracking_scale(axes)
    assert axes[0].get_ylim() == axes[1].get_ylim() == (0, 1e-10)
    plt.close(fig)


def test_dimension_scale_is_linear_for_a_narrow_positive_range():
    from wind_benchmark.reporting.standard import plt, tracking_scale

    fig, ax = plt.subplots()
    ax.plot([10, 30, 100], [120, 140, 160])
    ax.vlines([10, 30, 100], [110, 130, 150], [130, 150, 170])
    tracking_scale([ax], linear_when_narrow=True)
    assert ax.get_yscale() == "linear"
    assert ax.get_ylim()[0] < 110 and ax.get_ylim()[1] > 170
    plt.close(fig)


def test_interface_and_skip_logs_explain_required_contract(profile, tmp_path, capsys):
    task = Task("simplex", "geometry", geometry="simplex", landscape="simplex", dim=6)
    run_suite(tmp_path / "skip", profile, [catalog_method("SGD")], [task], plots=False)
    log = (tmp_path / "skip/execution.log").read_text()
    assert "REQUIRED:" in log and "sum=1" in log and "shape (6,)" in log
    assert "obs.grad only" in capsys.readouterr().out

    method = Method("invalid", "zero-order", ("simplex",), lambda t, s: object())
    result = run_suite(tmp_path / "invalid", profile, [method], [task], plots=False)
    assert set(result["runs"].status) == {"ERROR_INTERFACE"}
    log = (tmp_path / "invalid/execution.log").read_text()
    assert "initialize(x0, rng, task_spec)" in log and "obs.value only" in log
    assert "advance(client)" in log
