"""Public feedback, deferred release, and equivalence to the legacy metric loop."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from wind_benchmark.core.runner import BenchmarkRunner, BatchRunner
from wind_benchmark.core.environment import (
    DynamicEnvironment,
    QuadraticLandscape,
    LinearDrift,
    JumpDrift,
    RandomWalkDrift,
    StationaryDrift,
    GaussianNoise,
)
from wind_benchmark.core.oracle import (
    FirstOrderOracle,
    ZeroOrderOracle,
    HybridOracle,
    ScheduledOracle,
    OfflineOracle,
)
from wind_benchmark.core.metrics import (
    MetricsCollection,
    TrackingErrorMetric,
    MaxCoordinateErrorMetric,
    LyapunovMetric,
    NormalizedLyapunovMetric,
    AsymptoticBoundMetric,
    InstantaneousLossMetric,
    DynamicRegretMetric,
    TimeToRecoveryMetric,
    DriftAdaptationMetric,
    AdaptivityMetric,
    QueryEfficiencyMetric,
)


def environment(kind="linear"):
    drift = {
        "linear": lambda: LinearDrift([0.02, -0.01]),
        "random": lambda: RandomWalkDrift(sigma=0.03, seed=19),
        "jump": lambda: JumpDrift(jump_magnitude=1.5, interval=12, dim=2, seed=19),
    }[kind]()
    return DynamicEnvironment(
        dim=2,
        drift=drift,
        landscape=QuadraticLandscape(2, 3, seed=21),
    )


class LegacyRegret(DynamicRegretMetric):
    """The pre-change update, including its reference field in observation."""

    def update(self, t, x, theta, observation, environment):
        regret = environment.landscape.loss(x, theta) - observation.optimum_value
        self._cumulative_regret += regret
        if self.normalize_by_path and self._prev_theta is not None:
            self._path_variation += np.linalg.norm(theta - self._prev_theta)
        self._prev_theta = theta.copy()
        self._increments.append(float(regret))
        self._path_history.append(self._path_variation)
        self._history.append(self._cumulative_regret)
        self._current_value = self._cumulative_regret


def metrics(normalized=False, legacy=False):
    return MetricsCollection(
        [
            TrackingErrorMetric(),
            TrackingErrorMetric(norm=1.5, normalize_by_dim=True),
            MaxCoordinateErrorMetric(),
            LyapunovMetric(rho=0.5),
            NormalizedLyapunovMetric(rho=0.5, drift_magnitude_A=0.1),
            AsymptoticBoundMetric(rho=0.5),
            InstantaneousLossMetric(),
            (LegacyRegret if legacy else DynamicRegretMetric)(
                normalize_by_path=normalized
            ),
            TimeToRecoveryMetric(jump_threshold=0.5, epsilon=0.2),
            DriftAdaptationMetric(),
            AdaptivityMetric(jump_threshold=0.5, epsilon=0.2),
            QueryEfficiencyMetric(),
        ]
    )


from wind_benchmark.algorithms.protocol import AdvanceResult


class SGD:
    name = "privacy-test-sgd"

    query_cost_per_step = 1

    def initialize(self, x0, rng, task_spec):
        self.x = x0.copy()

    def advance(self, client):
        observation = client.query(self.x)
        self.x = self.inspect_observation(observation)
        return AdvanceResult(self.x, True, 1)

    def inspect_observation(self, observation):
        assert set(asdict(observation)) == {
            "x",
            "t",
            "value",
            "grad",
            "query_index",
            "mode",
        }
        return observation.x - 0.1 * observation.grad


@pytest.mark.parametrize("kind", ["linear", "random", "jump"])
@pytest.mark.parametrize("normalized", [False, True])
def test_metric_histories_match_legacy_loop(kind, normalized, trusted_core_unit):
    # Independently run the old timing / aggregation path, with a fixed noise tape.
    env = environment(kind)
    oracle = HybridOracle(
        env, GaussianNoise(sigma=0.8, seed=31), GaussianNoise(sigma=0.04, seed=32)
    )
    old = metrics(normalized, legacy=True)
    env.reset()
    oracle.reset()
    x = np.array([1.0, -0.5])
    xs, thetas = [], []
    for t in range(60):
        oracle.start_step(t)
        obs = oracle.query(x)
        oracle.end_step()
        theta = env.get_current_theta(for_analysis=True)
        old_obs = SimpleNamespace(
            **asdict(obs), optimum_value=env.landscape.loss(theta, theta)
        )
        old.update(t, x, theta, old_obs, env)
        xs.append(x.copy())
        thetas.append(theta.copy())
        env.step(action=x)
        x = x - 0.1 * obs.grad

    env = environment(kind)
    oracle = HybridOracle(
        env, GaussianNoise(sigma=0.8, seed=31), GaussianNoise(sigma=0.04, seed=32)
    )
    runner = BenchmarkRunner(env, oracle, metrics(normalized), record_trajectory=True)
    # This test isolates the information-barrier migration, using the original
    # component streams; root-seeded v2 runs are tested separately.
    result = runner.run(
        SGD(), T=60, x0=np.array([1.0, -0.5]), seed=123, reseed_components=False
    )
    np.testing.assert_array_equal(result.trajectory["x"][1:], xs)
    np.testing.assert_array_equal(result.trajectory["theta"][1:], thetas)
    for name, history in old.get_histories().items():
        np.testing.assert_allclose(
            result.metrics_history[name], history, rtol=0, atol=0
        )
        if name in result.final_metrics:
            np.testing.assert_allclose(
                result.final_metrics[name], old.get_results()[name], rtol=0, atol=0
            )
    assert len(result.evaluation_log) == 60
    # The real JSON export must contain numeric arrays, not their string repr.
    exported = json.loads(json.dumps(result.to_dict()))
    assert exported["evaluation_log"][0]["theta"] == list(thetas[0])


@pytest.mark.parametrize(
    "mode", ["fo", "blind", "zo", "hybrid", "scheduled", "offline"]
)
def test_every_oracle_returns_only_public_fields(mode, trusted_core_unit):
    env = environment()
    factories = {
        "fo": lambda: FirstOrderOracle(env, blind_value=False),
        "blind": lambda: FirstOrderOracle(env),
        "zo": lambda: ZeroOrderOracle(env),
        "hybrid": lambda: HybridOracle(env),
        "scheduled": lambda: ScheduledOracle(
            env, [("first-order", 1), ("zero-order", 1)]
        ),
        "offline": lambda: OfflineOracle(
            env, [np.ones(2), np.ones(2) * 2], env.landscape
        ),
    }
    oracle = factories[mode]()
    for t in range(2):
        oracle.start_step(t)
        obs = oracle.query(np.zeros(2))
        record = oracle._take_evaluation()
        assert set(asdict(obs)) == {"x", "t", "value", "grad", "query_index", "mode"}
        assert record.clean_value == env.landscape.loss(obs.x, record.theta)
        assert record.optimum_value == env.landscape.loss(record.theta, record.theta)
        oracle.end_step()
        env.step()


def test_results_are_withheld_during_run_and_discarded_on_failure():
    env = environment()
    runner = BenchmarkRunner(
        env, FirstOrderOracle(env), metrics(), record_trajectory=True
    )

    class InspectingSGD(SGD):
        fail = False

        def inspect_fields(self):
            for field in ("trajectory", "metrics_history", "metrics"):
                with pytest.raises(PermissionError):
                    getattr(runner, field)

        def inspect_observation(self, obs):
            self.inspect_fields()
            if self.fail and obs.t == 3:
                raise RuntimeError("intentional failure")
            return super().inspect_observation(obs)

    opt = InspectingSGD()
    first = runner.run(opt, T=8, x0=np.ones(2))
    second = runner.run(opt, T=8, x0=np.ones(2))
    assert (
        first.final_metrics["dynamic_regret"] == second.final_metrics["dynamic_regret"]
    )
    opt.fail = True
    with pytest.raises(RuntimeError, match="intentional failure"):
        runner.run(opt, T=8, x0=np.ones(2))
    for field in ("trajectory", "metrics_history", "metrics"):
        with pytest.raises(PermissionError):
            getattr(runner, field)
    assert runner.oracle.n_queries == 4


@pytest.mark.parametrize("fail", [False, True])
def test_multi_query_client_is_scoped_and_accounts_for_each_query(fail):
    env = environment()
    runner = BenchmarkRunner(
        env, ZeroOrderOracle(env), metrics(), record_trajectory=True
    )

    class TwoPoint:
        query_cost_per_step = 3

        def initialize(self, x0, rng, task_spec):
            self.client = None
            self.x = x0.copy()

        def advance(self, client):
            self.client = client
            obs = client.query(self.x)
            for attr in (
                "environment",
                "get_current_theta",
                "optimum_value",
                "start_step",
                "end_step",
            ):
                assert not hasattr(client, attr)
            plus = client.query(obs.x + 0.01)
            minus = client.query(obs.x - 0.01)
            assert not hasattr(plus, "optimum_value")
            assert plus.query_index + 1 == minus.query_index
            if fail:
                raise RuntimeError("query failure")
            self.x = obs.x - 0.01 * (plus.value - minus.value) / 0.02
            return AdvanceResult(self.x, True, 3)

    opt = TwoPoint()
    if fail:
        with pytest.raises(RuntimeError, match="query failure"):
            runner.run(opt, T=4, x0=np.ones(2))
        with pytest.raises(PermissionError):
            _ = runner.trajectory
    else:
        result = runner.run(opt, T=4, x0=np.ones(2))
        assert runner.oracle.n_queries == 12
        assert len(result.evaluation_log) == 12
        for i in range(0, 12, 3):
            records = result.evaluation_log[i : i + 3]
            for r in records[1:]:
                np.testing.assert_array_equal(r.theta, records[0].theta)
        assert np.isnan(result.final_metrics["query_efficiency"])
        status = result.metric_status["query_efficiency"]
        assert status["state"] == "right_censored"
        assert status["restricted_queries"] == 12
    with pytest.raises(RuntimeError, match="only available"):
        opt.client.query(np.zeros(2))
    if not fail:
        assert not runner.oracle._step_active


def test_offline_metrics_use_replayed_state_and_landscape():
    env = DynamicEnvironment(
        dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(2)
    )
    landscape = QuadraticLandscape(2, 4, seed=42)
    path = [np.array([1.0, 2.0]), np.array([2.0, 3.0])]
    oracle = OfflineOracle(env, path, landscape)
    runner = BenchmarkRunner(env, oracle, metrics(), record_trajectory=True)
    result = runner.run(SGD(), T=2, x0=np.zeros(2))
    expected = [landscape.loss(r.x, r.theta) for r in result.evaluation_log]
    np.testing.assert_array_equal(result.metrics_history["instant_loss"], expected)
    assert result.final_metrics["dynamic_regret"] == sum(expected)


def test_batch_does_not_print_true_metrics(capsys):
    batch = BatchRunner(environment, FirstOrderOracle, metrics, record_trajectory=True)
    results = batch.run(SGD, [1, 2], T=3)
    assert len(results) == 2
    assert all(len(r.evaluation_log) == 3 for r in results)
    assert "error=" not in capsys.readouterr().out
