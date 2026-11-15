"""Invariants introduced by the 2026-11 reproducibility audit fixes."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from wind_benchmark.experiments import paper as paper
from wind_benchmark.core.runner import BatchRunner, BenchmarkRunner, EnvironmentConfig
from wind_benchmark.algorithms.catalog import create_optimizer
from wind_benchmark.core.environment import (
    DynamicEnvironment,
    RandomWalkDrift,
    SparseDrift,
    StationaryDrift,
    QuadraticLandscape,
    PNormLandscape,
    GaussianNoise,
    HeavyTailedNoise,
    CorrelatedNoise,
    QuantizedNoise,
)
from wind_benchmark.experiments.standard import (
    Profile,
    Task,
    catalog_method,
    run_one,
    summarize,
)
from wind_benchmark.algorithms.optimizers import SGD
from wind_benchmark.experiments.configurable import run_single_experiment
from wind_benchmark.core.metrics import MetricsCollection, TrackingErrorMetric
from wind_benchmark.core.oracle import FirstOrderOracle, HybridOracle, ZeroOrderOracle
from wind_benchmark.core.execution import OracleSession
from wind_benchmark.core.randomness import stream_seed


def metrics():
    return MetricsCollection([TrackingErrorMetric(norm="l2")])


def hyperparameters(method):
    return paper.ZeroOrderHyperparameters(
        dict(
            create_optimizer(method.catalog_name).parameters,
            **({"m": method.directions} if method.family == "spherical" else {}),
        )
    )


@pytest.mark.parametrize("method", paper._zero_order_methods(), ids=lambda m: m.name)
def test_paper_label_and_plot_grid_cannot_change_results(method):
    case = paper.ZeroOrderCase("invariant", 3, 2.0, "linear", drift_per_query=0.001)
    hp = hyperparameters(method)
    first, scalar = paper._run_zero_order_method(
        method, "frozen", case, 80, np.arange(81), 101, hp
    )
    other, renamed = paper._run_zero_order_method(
        replace(method, name="Any display name"),
        "frozen",
        case,
        80,
        np.array([0, 40, 80]),
        101,
        hp,
    )
    np.testing.assert_array_equal(other, first[[0, 40, 80]])
    for key in scalar:
        np.testing.assert_equal(scalar[key], renamed[key])


@pytest.mark.parametrize("method", paper._zero_order_methods(), ids=lambda m: m.name)
def test_query_metrics_match_independent_full_tick_calculation(method, monkeypatch):
    updates = []
    cls = type(create_optimizer(method.catalog_name))
    original = cls.advance
    total = 0

    def traced(original):
        def advance(self, client):
            nonlocal total
            previous = self.recommendation.copy()
            result = original(self, client)
            total += result.queries
            updates.append((total, previous, result.recommendation.copy()))
            return result

        return advance

    monkeypatch.setattr(cls, "advance", traced(original))
    budget = 80
    _, result = paper._run_zero_order_method(
        method,
        "frozen",
        paper.ZeroOrderCase("metric", 3, 2.0, "stationary"),
        budget,
        np.array([0, 80]),
        123,
        hyperparameters(method),
    )
    point = updates[0][1]
    values = []
    changes = {q: nxt for q, _, nxt in updates}
    for q in range(budget + 1):
        if q in changes:
            point = changes[q]
        values.append(sum(float(v) ** 2 for v in point))
    np.testing.assert_allclose(
        result["query_averaged_mse"], sum(values[:-1]) / budget, rtol=1e-14
    )
    np.testing.assert_allclose(result["tail_mse"], np.mean(values[64:80]), rtol=1e-14)
    np.testing.assert_allclose(result["final_mse"], values[80], rtol=1e-14)
    hits = [q for q in range(1, budget + 1) if values[q] <= 0.01]
    np.testing.assert_equal(result["queries_to_threshold"], hits[0] if hits else np.nan)


def test_plot_density_cannot_change_tuning():
    profile = replace(
        paper.ExperimentProfile.build("smoke"),
        zero_order_tuning_budget=40,
        curve_points=3,
    )
    case = paper.ZeroOrderCase("tune", 3, 2.0, "stationary")
    methods = [paper.ZeroOrderMethodConfig("name", "spsa")]
    selected_a, table_a = paper._tune_zero_order_methods(profile, [case], methods)
    selected_b, table_b = paper._tune_zero_order_methods(
        replace(profile, curve_points=61), [case], methods
    )
    assert selected_a == selected_b
    pd.testing.assert_frame_equal(table_a, table_b)


def test_estimator_draws_do_not_advance_observation_noise(monkeypatch):
    tapes = []
    original = GaussianNoise.apply

    def apply(self, signal, t):
        observed = original(self, signal, t)
        tapes[-1].append(float((observed - signal).item()))
        return observed

    monkeypatch.setattr(GaussianNoise, "apply", apply)
    for directions in (1, 5):
        tapes.append([])
        method = paper.ZeroOrderMethodConfig("same", "spherical", directions)
        paper._run_zero_order_method(
            method,
            "frozen",
            paper.ZeroOrderCase("noise_tape", 3, 2.0, "stationary"),
            40,
            np.arange(41),
            100,
            hyperparameters(method),
        )
    np.testing.assert_allclose(tapes[0], tapes[1], rtol=0, atol=1e-14)


def test_default_display_name_is_inert_and_current_identity_exported(tmp_path):
    from wind_benchmark.experiments.standard import run_suite

    profile = Profile("test", 40, (101, 202), 100, 5)
    task = Task("clean", "reference", dim=3)
    method = catalog_method("SPSA")
    a = run_one(method, task, profile, 101)
    b = run_one(replace(method, name="Alternative legend"), task, profile, 101)
    assert a[0]["status"] == b[0]["status"] == "SUCCESS"
    assert a[0]["score"] == b[0]["score"]
    assert a[3]["algorithm_seed"] == b[3]["algorithm_seed"]
    assert "requested_alias" not in a[0] and "requested_alias" not in b[0]
    method = catalog_method("Nesterov")
    assert method.parameters["lr"] == 0.05
    # Full export check, not only the wrapper object.
    run_suite(tmp_path / "run", profile, [method], [task], plots=False, progress=False)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["methods"][0]["name"] == "Nesterov"


def test_noise_selection_and_capabilities_do_not_come_from_names():
    gaussian = run_single_experiment(SGD(), 1.0, 0.01, 3, T=8, seed=101)
    heavy = run_single_experiment(
        SGD(), 1.0, 0.01, 3, T=8, seed=101, noise_type="heavy_tailed"
    )
    assert heavy.environment_config.noise_type == "HeavyTailedNoise"
    assert not np.array_equal(gaussian.trajectory["x"], heavy.trajectory["x"])
    with pytest.raises(ValueError, match="noise_type"):
        run_single_experiment(SGD(), 1.0, 0.01, 3, noise_type="typo")
    for label in ("SPSA", "AdamW", "a new legend"):
        optimizer = SGD()
        optimizer.name = label
        result = run_single_experiment(optimizer, 1.0, 0.01, 3, T=8, seed=101)
        assert result.optimizer_info.implementation_id == "wind060.SGD.v1"
        np.testing.assert_array_equal(result.trajectory["x"], gaussian.trajectory["x"])
    optimizer = create_optimizer("Nesterov")
    result = run_single_experiment(optimizer, 1.0, 0.01, 3, T=3)
    assert result.optimizer_info.name == "Nesterov"
    assert result.optimizer_info.hyperparameters["lr"] == 0.05


@pytest.mark.parametrize("seeded", [False, True])
def test_batch_factory_seeds_replay_and_are_order_invariant(seeded):
    received = []

    def make(seed=None):
        received.append(seed)
        return DynamicEnvironment(
            3, RandomWalkDrift(0.01, seed=seed), QuadraticLandscape(3, 2.0, seed=seed)
        )

    factory = make if seeded else lambda: make()
    batch = BatchRunner(
        factory,
        lambda env, seed=None: FirstOrderOracle(
            env, grad_noise=GaussianNoise(0.01, seed)
        ),
        metrics,
        record_trajectory=True,
    )
    a = batch.run(SGD, [17, 18], T=5)
    b = batch.run(SGD, [18, 19, 17], T=5)
    np.testing.assert_array_equal(a[0].trajectory["x"], b[2].trajectory["x"])
    np.testing.assert_array_equal(a[1].trajectory["theta"], b[0].trajectory["theta"])
    assert not np.array_equal(a[0].trajectory["theta"], a[1].trajectory["theta"])
    if seeded:
        assert received[0] == stream_seed(17, "environment")


def test_gym_seed_controls_episode_and_implicit_resets_advance_streams():
    pytest.importorskip("gymnasium")
    from wind_benchmark.integrations.gym_env import WindGymEnv

    env = DynamicEnvironment(
        3, RandomWalkDrift(0.05, seed=1), QuadraticLandscape(3, 2.0, seed=2)
    )
    gym = WindGymEnv(env, ZeroOrderOracle(env, value_noise=GaussianNoise(0.01, 3)), T=4)

    def episode(seed=None, options=None):
        obs, _ = gym.reset(seed=seed, options=options)
        history = [obs]
        for _ in range(4):
            obs, _, _, _, info = gym.step(np.zeros(3))
            assert not {"theta", "clean_value", "tracking_error"}.intersection(info)
            history.append(obs)
        return np.array(history)

    a = episode(111)
    b = episode(222)
    c = episode()
    assert not np.array_equal(a, b) and not np.array_equal(b, c)
    np.testing.assert_array_equal(a, episode(111))
    np.testing.assert_array_equal(a, episode(options={"replay": True}))


def test_seed_aware_factory_preserves_explicit_landscape_realization():
    matrix = np.diag([1.0, 3.0, 9.0])

    def make(seed=None):
        landscape = QuadraticLandscape(3, 9.0, seed)
        landscape.A = matrix.copy()
        return DynamicEnvironment(3, RandomWalkDrift(0.01, seed=seed), landscape)

    batch = BatchRunner(make, lambda env: FirstOrderOracle(env), metrics)
    result = batch.run(SGD, [17], T=3)[0]
    restored, _ = result.environment_config.restore()
    np.testing.assert_array_equal(restored.landscape.A, matrix)


def test_runner_reseeds_private_algorithm_generator():
    from wind_benchmark import BenchmarkRunner

    class RandomStep:
        name = "private-rng"

        def __init__(self):
            self.rng = np.random.default_rng()

        query_cost_per_step = 1

        def initialize(self, x0, rng, task_spec):
            self.x, self.rng = x0.copy(), rng

        def advance(self, client):
            from wind_benchmark.algorithms.protocol import AdvanceResult

            client.query(self.x)
            self.x = self.rng.normal(size=3)
            return AdvanceResult(self.x, True, 1)

    env = DynamicEnvironment(
        3, RandomWalkDrift(0.01, seed=1), QuadraticLandscape(3, 2.0, 2)
    )
    runner = BenchmarkRunner(
        env, FirstOrderOracle(env), metrics(), record_trajectory=True
    )
    optimizer = RandomStep()
    a = runner.run(optimizer, T=4, x0=np.zeros(3), seed=17)
    b = runner.run(optimizer, T=4, x0=np.zeros(3), seed=17)
    np.testing.assert_array_equal(a.trajectory["x"], b.trajectory["x"])


@pytest.mark.parametrize(
    "landscape", [QuadraticLandscape(3, 100.0, 7), PNormLandscape(1.5, 7.0, 8)]
)
@pytest.mark.parametrize(
    "noise",
    [
        HeavyTailedNoise(2.2, 0.1, 10),
        CorrelatedNoise(0.1, 0.7, 3, 12),
        QuantizedNoise(0.17),
    ],
)
def test_export_roundtrip_restores_both_noise_channels_and_full_task(landscape, noise):
    env = DynamicEnvironment(
        3, SparseDrift(3, 2, 0.04, 8), landscape, np.array([1.0, 2.0, 3.0])
    )
    oracle = HybridOracle(env, GaussianNoise(0.01, 9), noise)
    env.reset()  # Construction snapshot reproduces the start, not a mid-run state.
    config = EnvironmentConfig.from_environment(env)
    serialized = json.loads(json.dumps(asdict(config), allow_nan=False))
    restored_env, restored_oracle = EnvironmentConfig(**serialized).restore()
    assert config.landscape_parameters["condition_number"] in (7.0, 100.0)
    assert restored_oracle.value_noise.sigma == 0.01
    assert type(restored_oracle.grad_noise) is type(noise)
    left = OracleSession(oracle, 5)
    right = OracleSession(restored_oracle, 5)
    try:
        for q in range(5):
            point = np.full(3, 0.1 * q)
            a, b = left.observe(point), right.observe(point)
            assert a.value == b.value
            np.testing.assert_array_equal(a.grad, b.grad)
    finally:
        left.finish()
        right.finish()


def test_bootstrap_is_invariant_to_names_groups_and_row_order():
    data = pd.DataFrame({"method": ["B"] * 5, "score": [0.0, 1.0, 2.0, 20.0, 100.0]})
    a = paper._summarize_runs(data, ["method"], ["score"], 100, 123).drop(
        columns="method"
    )
    renamed = data.assign(method="New label").iloc[::-1]
    b = paper._summarize_runs(
        pd.concat([data.assign(method="A"), renamed]), ["method"], ["score"], 100, 123
    )
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True),
        b[b.method == "New label"].drop(columns="method").reset_index(drop=True),
    )


def test_import_does_not_disable_user_runtime_warnings():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import warnings; warnings.simplefilter('default', RuntimeWarning); import wind_benchmark; "
            "warnings.warn('audit-visible-warning', RuntimeWarning)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "audit-visible-warning" in result.stderr


def test_rl_policy_ids_and_reordered_arms_preserve_fine_tuning():
    pytest.importorskip("gymnasium")
    from wind_benchmark.experiments import rl as rl
    from wind_benchmark.integrations.rl_env import ResourceAllocationEnv

    profile = replace(rl.PROFILES["smoke"], transfer_horizon=12)

    def run(order):
        result = {}
        for condition in order:
            seed = stream_seed(
                101, "rl_transfer_v3", "adaptation", "QLearning", 0, condition
            )
            policy = rl.QLearning(seed)
            specs = rl.episode_schedule(profile, 202, 0, "adaptation", 0, 2, condition)
            env = ResourceAllocationEnv(specs)
            rl._q_train(policy, env)
            env.close()
            result[condition] = policy.q_values.copy()
        return result

    a = run(["curriculum", "stationary_only"])
    b = run(["no_pretraining", "stationary_only", "curriculum"])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
