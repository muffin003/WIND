"""Series-level denial, retained aliases, side channels, and post-run release."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import pickle
import subprocess
import sys

import numpy as np
import pytest

from wind_benchmark import BenchmarkRunner, EvaluationSession, GroundTruthAccessError
from wind_benchmark.core.runner import BatchRunner, EnvironmentConfig
from wind_benchmark.core.environment import (
    DynamicEnvironment,
    GaussianNoise,
    LinearDrift,
    QuadraticLandscape,
)
from wind_benchmark.algorithms.optimizers import SGD
from wind_benchmark.algorithms.protocol import AdvanceResult
from wind_benchmark.core.metrics import MetricsCollection, TrackingErrorMetric
from wind_benchmark.core.oracle import FirstOrderOracle, ZeroOrderOracle
from wind_benchmark.core.serialization import environment_snapshot


def task():
    env = DynamicEnvironment(
        2, LinearDrift(np.array([0.01, 0.02])), QuadraticLandscape(2), np.ones(2)
    )
    oracle = FirstOrderOracle(env, grad_noise=GaussianNoise(0.01, 3))
    runner = BenchmarkRunner(
        env, oracle, MetricsCollection([TrackingErrorMetric()]), record_trajectory=True
    )
    return env, oracle, runner


def denied(function):
    with pytest.raises(GroundTruthAccessError):
        function()


def test_retained_aliases_serialization_and_threads_cannot_read_live_truth():
    env, oracle, runner = task()
    getter = env.get_current_theta
    take = oracle._take_evaluation
    drift, noise = env.drift, oracle.grad_noise

    class Inspector(SGD):
        def initialize(self, x0, rng, task_spec):
            probes = [
                lambda: getter(for_analysis=True),
                lambda: env._theta,
                lambda: env._initial_theta,
                lambda: vars(env),
                lambda: drift.velocity,
                lambda: noise.rng,
                lambda: oracle.get_current_theta(),
                lambda: oracle._cached_theta,
                take,
                lambda: vars(oracle),
                lambda: environment_snapshot(env),
                lambda: EnvironmentConfig.from_environment(env),
                lambda: pickle.dumps(env),
                lambda: runner.metrics,
                lambda: runner.trajectory,
                lambda: vars(runner),
            ]
            for probe in probes:
                denied(probe)
            with ThreadPoolExecutor(max_workers=1) as pool:
                denied(lambda: pool.submit(getter, True).result())
            super().initialize(x0, rng, task_spec)

        def advance(self, client):
            denied(lambda: getter(True))
            return super().advance(client)

    result = runner.run(Inspector(), T=4, x0=np.zeros(2), seed=17)
    assert len(result.evaluation_log) == 4
    assert np.isfinite(result.final_metrics["error_l2"])
    assert env.get_current_theta(True).shape == (2,)


def test_series_holds_first_run_until_all_declared_runs_finish(tmp_path):
    env, oracle, runner = task()
    with EvaluationSession(expected_runs=2) as series:
        first = runner.run(SGD(), T=3, x0=np.zeros(2), seed=17)
        for probe in (
            lambda: first.final_metrics,
            lambda: asdict(first),
            lambda: first.save_to_json(tmp_path / "early.json"),
            lambda: env.get_current_theta(True),
            series.finish,
            lambda: setattr(series, "_EvaluationSession__state", "released"),
        ):
            denied(probe)
        assert not (tmp_path / "early.json").exists()
        second = runner.run(SGD(), T=3, x0=np.zeros(2), seed=17)
        denied(lambda: second.trajectory)
    np.testing.assert_array_equal(first.trajectory["x"], second.trajectory["x"])
    assert series.state == "released"


def test_policy_cannot_release_or_open_privileged_sessions():
    from wind_benchmark.core import privacy as privacy
    from wind_benchmark.core.oracle import OracleClient

    env, _, runner = task()
    with EvaluationSession(expected_runs=1) as series:

        class Inspector(SGD):
            def advance(self, client):
                denied(series.finish)
                denied(
                    lambda: OracleClient(
                        lambda point: env.get_current_theta(True), lambda: 2
                    )
                )
                denied(lambda: EvaluationSession().__enter__())
                denied(lambda: privacy._internal(lambda: env.get_current_theta(True)))
                with pytest.raises(GroundTruthAccessError):
                    with privacy._evaluation():
                        env.get_current_theta(True)
                return super().advance(client)

        result = runner.run(Inspector(), T=2, x0=np.zeros(2))
    assert len(result.evaluation_log) == 2


def test_initial_array_conversion_does_not_leak_before_first_observation():
    env, _, runner = task()

    class InitialPoint:
        def __array__(self, dtype=None, copy=None):
            denied(lambda: env.get_current_theta(True))
            return np.zeros(2, dtype=dtype)

    assert runner.run(SGD(), T=2, x0=InitialPoint()).status == "SUCCESS"


def test_array_conversion_and_optimizer_properties_have_no_evaluator_rights():
    env, _, runner = task()

    class Inspector(SGD):
        @property
        def advance(self):
            denied(lambda: env.get_current_theta(True))
            return super().advance

    assert runner.run(Inspector(), T=2, x0=np.zeros(2)).status == "SUCCESS"


def test_batch_factories_do_not_inherit_access_to_previous_runs():
    saved = []

    def environment_factory(seed=None):
        if saved:
            denied(lambda: saved[-1].get_current_theta(True))
        env, _, _ = task()
        saved.append(env)
        return env

    batch = BatchRunner(
        environment_factory,
        FirstOrderOracle,
        lambda: MetricsCollection([TrackingErrorMetric()]),
    )
    results = batch.run(SGD, [17, 18], T=2)
    assert len(results) == 2


def test_custom_evaluator_callbacks_are_rejected_without_receiving_truth():
    received = []

    class LeakingNoise:
        def apply(self, clean, t):
            received.append(clean)
            return clean

    env, _, runner = task()
    runner.oracle = FirstOrderOracle(env, grad_noise=LeakingNoise())
    denied(lambda: runner.run(SGD(), T=2, x0=np.zeros(2)))
    assert received == []


def test_query_session_cannot_release_partial_runs_or_outlive_parent():
    from wind_benchmark.core.execution import OracleSession

    env, _, _ = task()
    with EvaluationSession() as parent:
        session = OracleSession(ZeroOrderOracle(env), 2)
        session.observe(np.zeros(2))
        denied(parent.finish)
        session.observe(np.zeros(2))
        records = session.finish()
        denied(lambda: records[0].theta)
    assert records[0].theta.shape == (2,)
    with pytest.raises(RuntimeError):
        session.observe(np.zeros(2))

    env, _, _ = task()
    partial = OracleSession(ZeroOrderOracle(env), 2)
    partial.observe(np.zeros(2))
    denied(partial.finish)
    denied(lambda: env.get_current_theta(True))


def test_gym_retained_core_and_array_hooks_are_blocked_until_series_finish():
    pytest.importorskip("gymnasium")
    from wind_benchmark.integrations.gym_env import WindGymEnv
    from wind_benchmark.integrations.gym_evaluation import GymEvaluationSession

    core, _, _ = task()
    oracle = ZeroOrderOracle(core)
    series = GymEvaluationSession()
    gym = WindGymEnv(core, oracle, T=1, evaluation_session=series)
    gym.reset()

    class Action:
        def __array__(self, dtype=None, copy=None):
            denied(lambda: core.get_current_theta(True))
            return np.zeros(2, dtype=dtype)

    denied(lambda: vars(gym))
    denied(lambda: gym.unwrapped._WindGymEnv__environment)
    gym.step(Action())
    denied(lambda: core.get_current_theta(True))
    series.finish()
    assert len(series.records[0]) == 2
    with pytest.raises(RuntimeError, match="closed"):
        gym.reset()


def test_default_exports_and_trace_callbacks_wait_for_outer_session(tmp_path):
    from wind_benchmark.experiments.standard import (
        Profile,
        Task,
        catalog_method,
        run_one,
        run_suite,
    )

    profile = Profile("smoke", 40, (101, 202), 100, 3)
    calls = []
    with EvaluationSession():
        pending = run_suite(
            tmp_path / "suite",
            profile,
            [catalog_method("SGD")],
            [Task("clean", "reference", dim=2)],
            plots=False,
            progress=False,
        )
        run_one(
            catalog_method("SGD"),
            Task("clean", "reference", dim=2),
            profile,
            101,
            trace_sink=lambda meta, error: calls.append(error),
        )
        denied(pending.result)
        assert {p.name for p in (tmp_path / "suite").iterdir()} <= {
            "recommendations",
            "execution_status.json",
        }
        assert calls == []
    assert (tmp_path / "suite/runs.csv").exists()
    assert len(pending.result()["runs"]) == 2
    assert len(calls) == 1


def test_abort_discards_pending_exports_and_keeps_results_locked(tmp_path):
    from wind_benchmark.experiments.standard import (
        Profile,
        Task,
        catalog_method,
        run_suite,
    )

    with pytest.raises(RuntimeError, match="stop"):
        with EvaluationSession():
            pending = run_suite(
                tmp_path / "aborted",
                Profile("smoke", 40, (1, 2), 100, 2),
                [catalog_method("SGD")],
                [Task("clean", "reference", dim=2)],
                plots=False,
                progress=False,
            )
            raise RuntimeError("stop")
    assert {p.name for p in (tmp_path / "aborted").iterdir()} <= {
        "recommendations",
        "execution_status.json",
    }
    denied(pending.result)


def test_web_exports_wait_for_all_runs_and_the_outer_boundary(tmp_path):
    from wind_benchmark.interfaces.web_runner import run_workbench

    config = {
        "schema_version": 3,
        "environment": {
            "dim": 2,
            "landscape": {"type": "quadratic"},
            "drift": {"type": "stationary"},
        },
        "oracle": {"type": "first-order", "blind_value": True},
        "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
        "runner": {
            "steps": 3,
            "seeds": [17, 18],
            "tail_fraction": 0.2,
            "output_dir": "web_output",
        },
        "metrics": ["tracking_error", "dynamic_regret"],
    }
    with EvaluationSession(expected_runs=2):
        pending = run_workbench(config, tmp_path)
        denied(pending.result)
        assert not list((tmp_path / "web_output").iterdir())
    assert len(pending.result()["runs"]) == 2
    assert (tmp_path / "web_output/workbench_summary.json").exists()


def test_cli_suite_has_no_intermediate_metric_logs_or_checkpoints(tmp_path, capsys):
    from wind_benchmark.experiments.configurable import run_full_experiment_suite

    with EvaluationSession(expected_runs=1):
        run_full_experiment_suite(
            str(tmp_path),
            seeds=[17],
            T=3,
            rho_values=[1.0],
            A_values=[0.01],
            dimensions=[2],
            optimizer_names=["SGD"],
        )
        assert not list(tmp_path.rglob("*.json"))
        assert not list(tmp_path.rglob("*.csv"))
        assert "Lyapunov=" not in capsys.readouterr().out
    assert (tmp_path / "full_results.csv").exists()


def test_outer_session_cannot_release_a_running_gym_episode():
    pytest.importorskip("gymnasium")
    from wind_benchmark.integrations.gym_env import WindGymEnv

    core, _, _ = task()
    with EvaluationSession() as series:
        gym = WindGymEnv(core, ZeroOrderOracle(core), T=1)
        gym.reset()
        denied(series.finish)
        gym.step(np.zeros(2))
        denied(lambda: core.get_current_theta(True))
    with pytest.raises(RuntimeError, match="closed"):
        gym.reset()


def test_cli_module_recognizes_builtin_components_defined_in_main(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "wind_benchmark.experiments.standard",
            "--profile",
            "smoke",
            "--tasks",
            "simplex_moving_gaussian",
            "--algorithms",
            "ExponentiatedGradient",
            "--no-plots",
            "--output-dir",
            str(tmp_path / "module_run"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "ERROR" not in process.stdout
    assert (tmp_path / "module_run/manifest.json").exists()


@pytest.mark.parametrize(
    "kind", ["first-order", "zero-order", "hybrid", "scheduled", "offline"]
)
def test_raw_oracle_cannot_bypass_a_managed_run(kind):
    from wind_benchmark.core.oracle import make_oracle

    env, _, _ = task()
    params = {}
    if kind == "scheduled":
        params["schedule"] = [("fo", 1), ("zo", 1)]
    elif kind == "offline":
        params.update(recorded_thetas=[np.zeros(2)], landscape=env.landscape)
    oracle = make_oracle(kind, env, **params)
    denied(lambda: oracle.start_step(0))
    denied(oracle._take_evaluation)
    assert oracle.n_queries == 0
