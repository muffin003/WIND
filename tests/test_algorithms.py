import pytest
import numpy as np
import json
import tempfile
import random
from pathlib import Path
from unittest.mock import Mock

# Core components
from wind_benchmark.core import (
    StationaryDrift,
    LinearDrift,
    RandomWalkDrift,
    CyclicDrift,
    JumpDrift,
    AdaptiveDrift,
    SparseDrift,
    QuadraticLandscape,
    PNormLandscape,
    RosenbrockLandscape,
    MultiExtremalLandscape,
    RobustLandscape,
    SimplexLandscape,
    GaussianNoise,
    HeavyTailedNoise,
    CorrelatedNoise,
    QuantizedNoise,
    MultiplicativeNoise,
    SparseNoise,
    DynamicEnvironment,
    make_drift,
    make_landscape,
    make_noise,
)
from wind_benchmark.oracle import (
    FirstOrderOracle,
    ZeroOrderOracle,
    HybridOracle,
    ScheduledOracle,
    OfflineOracle,
    Observation,
    OracleProtocol,
    make_oracle,
)
from wind_benchmark.metrics import (
    MetricsCollection,
    TrackingErrorMetric,
    LyapunovMetric,
    NormalizedLyapunovMetric,
    DynamicRegretMetric,
    InstantaneousLossMetric,
    AdaptivityMetric,
)
from wind_benchmark.benchmark import (
    BenchmarkRunner,
    ExperimentResult,
    OptimizerInfo,
    EnvironmentConfig,
    OptimizerProtocol,
    BatchRunner,
)


# ============================================================================
# 1. LANDSCAPE INVARIANTS (Critical for valid dynamic regret)
# ============================================================================
def test_landscape_invariant_loss_at_optimum_is_zero():
    """
    CRITICAL INVARIANT: L(theta, theta) MUST be exactly 0 for all landscapes.

    Scientific basis: Dynamic regret R_T = Σ[f_t(x_t) - f_t(theta_t)] requires
    f_t(theta_t) = 0 to be meaningful. Violation invalidates all regret analysis.
    """
    dim = 5
    theta = np.random.randn(dim)

    landscapes = [
        QuadraticLandscape(dim=dim, condition_number=10.0),
        PNormLandscape(p=2.0),
        PNormLandscape(p=1.5),
        RosenbrockLandscape(),
        MultiExtremalLandscape(k_centers=3, width=1.0),
        RobustLandscape(delta=1.0),
    ]

    for landscape in landscapes:
        loss = landscape.loss(theta, theta)
        assert (
            abs(loss) < 1e-8
        ), f"Landscape {landscape.__class__.__name__} violates L(theta,theta)=0: got {loss}"
        x_random = theta + np.random.randn(dim) * 2.0
        assert (
            landscape.loss(x_random, theta) >= -1e-8
        ), f"Landscape {landscape.__class__.__name__} produces negative loss"


def test_landscape_gradient_at_optimum_is_zero():
    """
    Gradient must vanish at global minimum (first-order optimality condition).

    Scientific basis: Necessary condition for unconstrained minimization.
    Critical for gradient-based optimizers to converge to optimum.
    """
    dim = 3
    theta = np.random.randn(dim)

    landscapes = [
        QuadraticLandscape(dim=dim, condition_number=5.0),
        PNormLandscape(p=2.0),
        RobustLandscape(delta=1.0),
    ]

    for landscape in landscapes:
        grad = landscape.grad(theta, theta)
        assert np.allclose(
            grad, np.zeros(dim), atol=1e-8
        ), f"Landscape {landscape.__class__.__name__} gradient not zero at optimum: {grad}"


# ============================================================================
# 2. Temporal consistency
# ============================================================================
def test_protocol_a_locking_mechanism():
    """
    Environment must refuse advancement while Oracle is active.

    Scientific basis requires strict temporal separation:
      1. Oracle locks environment state theta_t
      2. All queries use identical theta_t (no lookahead)
      3. Environment advances only after Oracle unlocks

    Violation would enable cheating via future information leakage.
    """
    env = DynamicEnvironment(
        dim=2,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=2, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env)

    oracle.start_step(0)
    assert env._locked_by_oracle, "Environment not locked after start_step()"

    with pytest.raises(
        RuntimeError, match="Cannot advance environment while Oracle is active"
    ):
        env.step()

    oracle.end_step()
    assert not env._locked_by_oracle, "Environment not unlocked after end_step()"
    env.step()  # Should succeed now


def test_temporal_consistency_same_theta_for_multiple_queries():
    """
    All queries within same step must observe identical theta_t.

    Scientific basis: Non-anticipative interaction (Besbes et al. 2015).
    Critical for fair comparison: optimizers must not receive inconsistent feedback.
    """
    env = DynamicEnvironment(
        dim=3,
        drift=LinearDrift(velocity=np.array([0.1, 0.1, 0.1])),
        landscape=QuadraticLandscape(dim=3, condition_number=2.0),
        initial_theta=np.zeros(3),
    )
    oracle = FirstOrderOracle(
        environment=env, blind_value=False
    )  # this test reads value

    # Step 0: multiple queries observe same theta
    oracle.start_step(0)
    obs1 = oracle.query(np.array([1.0, 0.0, 0.0]))
    obs2 = oracle.query(np.array([0.0, 1.0, 0.0]))
    obs3 = oracle.query(np.array([0.0, 0.0, 1.0]))
    assert obs1.t == obs2.t == obs3.t == 0
    oracle.end_step()

    # Advance environment
    env.step()

    # Step 1: new theta produces different function values at same x
    oracle.start_step(1)
    obs4 = oracle.query(np.array([1.0, 0.0, 0.0]))
    oracle.end_step()

    # Critical invariant: f_t(x) must change because theta changed
    assert not np.isclose(obs1.value, obs4.value, atol=1e-3), (
        f"Function value did not change after environment advance: "
        f"step0={obs1.value:.6f}, step1={obs4.value:.6f}"
    )


# ============================================================================
# 3. INFORMATION BARRIER (No theta leakage)
# ============================================================================
def test_observation_information_barrier():
    """
    Observation must NEVER expose theta_t directly to optimizer.

    Scientific basis: Information barrier prevents cheating.
    Optimizer must learn theta_t only through function evaluations, not direct access.
    """
    env = DynamicEnvironment(
        dim=2,
        drift=RandomWalkDrift(sigma=0.1),
        landscape=QuadraticLandscape(dim=2, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env)

    x_original = np.array([1.0, 2.0])
    oracle.start_step(0)
    obs = oracle.query(x_original)
    oracle.end_step()

    # Verify forbidden attributes are absent
    forbidden_attrs = ["theta", "optimal_point", "minimum_location", "_theta"]
    for attr in forbidden_attrs:
        assert not hasattr(
            obs, attr
        ), f"Information leak: Observation has '{attr}' attribute"

    # Verify original query point immutability
    x_original_copy = x_original.copy()
    obs.x[0] = 999.0  # Modify the COPY inside Observation

    # Original array MUST remain unchanged (prevents reference leakage)
    assert np.array_equal(
        x_original, x_original_copy
    ), "Original query point was mutated through Observation.x!"


def test_blind_value_mode_hides_function_value():
    """
    Strict first-order mode must return value=None.

    Scientific basis: Table 1 requirement "Value unavailable" for pure gradient oracles.
    Ensures fair comparison between algorithms with different oracle access.
    """
    env = DynamicEnvironment(
        dim=1,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=1, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env, blind_value=True)

    oracle.start_step(0)
    obs = oracle.query(np.array([1.0]))
    oracle.end_step()

    assert obs.value is None, "Blind value mode leaked function value"
    assert obs.grad is not None, "Blind value mode incorrectly hid gradient"


# ============================================================================
# 4. LYAPUNOV METRICS (Theoretical correctness)
# ============================================================================
def test_lyapunov_metric_rho_consistency():
    """
    Lyapunov metric must use correct p-norm = rho + 1.

    Scientific basis: Definition 1 requires V_n(x) = ||x - theta_n||_{rho+1}^{rho+1}
    for rho-Hölder smooth landscapes. Incorrect norm invalidates stabilization analysis.
    """
    rho = 0.5
    metric = LyapunovMetric(rho=rho)

    x = np.array([2.0, 0.0])
    theta = np.array([0.0, 0.0])

    p = rho + 1.0
    distance = np.linalg.norm(x - theta, ord=p)
    expected = distance**p

    mock_env = Mock()
    mock_env.landscape = Mock()
    mock_obs = Mock()
    mock_obs.optimum_value = 0.0

    metric.update(0, x, theta, mock_obs, mock_env)

    assert np.isclose(metric._current_value, expected, rtol=1e-5)


def test_normalized_lyapunov_drift_invariance():
    """
    Normalized Lyapunov must produce consistent denormalized values.

    Scientific basis: Condition D requires E[||theta_n - theta_{n-1}||_{rho+1}^{rho+1}] <= A^{rho+1}.
    Normalization by A^{rho+1} enables fair comparison across drift magnitudes.
    """
    rho = 1.0
    A1, A2 = 0.1, 0.2

    metric1 = NormalizedLyapunovMetric(rho=rho, drift_magnitude_A=A1)
    metric2 = NormalizedLyapunovMetric(rho=rho, drift_magnitude_A=A2)

    x = np.array([0.3, 0.0])
    theta = np.array([0.0, 0.0])

    mock_env = Mock()
    mock_env.landscape = Mock()
    mock_obs = Mock()
    mock_obs.optimum_value = 0.0

    metric1.update(0, x, theta, mock_obs, mock_env)
    metric2.update(0, x, theta, mock_obs, mock_env)

    # Verify denormalization recovers identical raw Lyapunov value
    raw_value_1 = metric1._current_value * (A1 ** (rho + 1.0))
    raw_value_2 = metric2._current_value * (A2 ** (rho + 1.0))
    expected_raw = 0.09  # ||[0.3,0]||_2^2

    assert np.isclose(
        raw_value_1, expected_raw, rtol=1e-5
    ), f"Denormalization failed for A1: got {raw_value_1}, expected {expected_raw}"
    assert np.isclose(
        raw_value_2, expected_raw, rtol=1e-5
    ), f"Denormalization failed for A2: got {raw_value_2}, expected {expected_raw}"


# ============================================================================
# 5. DYNAMIC REGRET INVARIANT
# ============================================================================
def test_dynamic_regret_optimum_value_consistency():
    """
    Regret calculation must use f_t(theta_t) computed with SAME theta_t as f_t(x_t).

    Scientific basis: Dynamic regret R_T = Σ[f_t(x_t) - f_t(theta_t)] requires consistent
    evaluation points. Using different theta_t would invalidate regret bounds.
    """
    env = DynamicEnvironment(
        dim=2,
        drift=LinearDrift(velocity=np.array([0.1, 0.1])),
        landscape=QuadraticLandscape(dim=2, condition_number=3.0),
        initial_theta=np.array([1.0, 1.0]),
    )
    oracle = FirstOrderOracle(environment=env)

    oracle.start_step(0)
    theta_t0 = env.get_current_theta(for_analysis=True)
    x = np.array([2.0, 2.0])
    obs = oracle.query(x)
    oracle.end_step()

    # Critical invariant: optimum_value MUST be 0 (within tolerance)
    assert abs(obs.optimum_value) < 1e-8


# ============================================================================
# 6. BENCHMARK RUNNER (End-to-end execution)
# ============================================================================
class MockOptimizer(OptimizerProtocol):
    """Simple mock optimizer for testing benchmark infrastructure."""

    def __init__(self, dim: int, lr: float = 0.1):
        self.dim = dim
        self.lr = lr
        self.name = "MockSGD"

    def step(self, observation) -> np.ndarray:
        if observation.grad is not None:
            return observation.x - self.lr * observation.grad
        else:
            # Zero-order fallback (reproducible when global seed is fixed)
            return observation.x + np.random.randn(self.dim) * 0.01

    def reset(self) -> None:
        pass


def test_benchmark_runner_full_execution():
    """
    End-to-end test of BenchmarkRunner with trajectory recording.

    Validates:
      - Successful execution without exceptions
      - Correct trajectory dimensions (T+1 points including initial state)
      - All required metrics computed and available
      - Non-negative dynamic regret (guaranteed by L(theta,theta)=0 invariant)
    """
    dim = 3
    T = 50

    env = DynamicEnvironment(
        dim=dim,
        drift=RandomWalkDrift(sigma=0.05, seed=42),
        landscape=QuadraticLandscape(dim=dim, condition_number=5.0, seed=43),
        initial_theta=np.zeros(dim),
    )
    oracle = FirstOrderOracle(
        environment=env,
        value_noise=GaussianNoise(sigma=0.01, seed=44),
        seed=45,
    )
    metrics = MetricsCollection(
        [
            TrackingErrorMetric(norm="l2"),
            DynamicRegretMetric(),
            LyapunovMetric(rho=1.0),
        ]
    )

    runner = BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        record_trajectory=True,
        tail_fraction=0.2,
    )

    optimizer = MockOptimizer(dim=dim, lr=0.2)
    result = runner.run(optimizer, T=T, x0=np.ones(dim) * 2.0, seed=42)

    # Validate result structure
    assert result.status == "SUCCESS"
    assert result.trajectory is not None
    assert len(result.trajectory["x"]) == T + 1
    assert len(result.trajectory["theta"]) == T + 1

    # Verify metrics computed
    assert "error_l2" in result.final_metrics
    assert "dynamic_regret" in result.final_metrics
    assert "lyapunov_rho1.00" in result.final_metrics

    # Verify regret non-negative (landscape invariant)
    assert result.final_metrics["dynamic_regret"] >= -1e-6


def test_batch_runner_statistical_aggregation():
    """
    BatchRunner must correctly aggregate metrics across multiple seeds.

    Validates statistical aggregation pipeline for significance testing.
    """

    def env_factory():
        return DynamicEnvironment(
            dim=2,
            drift=StationaryDrift(),
            landscape=QuadraticLandscape(dim=2, condition_number=2.0),
        )

    def oracle_factory(env):
        return FirstOrderOracle(environment=env)

    def metric_factory():
        return MetricsCollection([TrackingErrorMetric(norm="l2")])

    def optimizer_factory():
        return MockOptimizer(dim=2, lr=0.1)

    batch_runner = BatchRunner(
        environment_factory=env_factory,
        oracle_factory=oracle_factory,
        metric_factory=metric_factory,
        record_trajectory=False,
    )

    seeds = [42, 43]
    results = batch_runner.run(optimizer_factory, seeds=seeds, T=20)

    assert len(results) == len(seeds)

    errors = [
        r.final_metrics["error_l2"] for r in results if "error_l2" in r.final_metrics
    ]
    assert len(errors) == len(seeds)


# ============================================================================
# 7. OFFLINE ORACLE (Deterministic replay)
# ============================================================================
def test_offline_oracle_reproducibility():
    """
    OfflineOracle must replay identical theta_t sequence across runs.

    Scientific basis: Reproducibility requirement (Pineau et al. 2021).
    Enables fair algorithm comparison under identical environment evolution.
    """
    dim = 2
    T = 5

    # Record theta sequence from live environment
    live_env = DynamicEnvironment(
        dim=dim,
        drift=RandomWalkDrift(sigma=0.1, seed=2024),
        landscape=QuadraticLandscape(dim=dim, condition_number=1.0),
        initial_theta=np.zeros(dim),
    )

    thetas_recorded = [live_env.get_current_theta(for_analysis=True).copy()]
    for _ in range(T):
        live_env.step()
        thetas_recorded.append(live_env.get_current_theta(for_analysis=True).copy())

    # Create OfflineOracle with recorded sequence
    dummy_env = DynamicEnvironment(
        dim=dim,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=dim, condition_number=1.0),
    )

    offline_oracle = OfflineOracle(
        environment=dummy_env,
        recorded_thetas=thetas_recorded,
        landscape=QuadraticLandscape(dim=dim, condition_number=1.0),
        seed=999,
    )

    # Verify identical theta_t across two runs
    def run_with_oracle(oracle, T):
        thetas_observed = []
        for t in range(T + 1):
            oracle.start_step(t)
            thetas_observed.append(oracle.get_current_theta().copy())
            oracle.query(np.zeros(dim))
            oracle.end_step()
        return thetas_observed

    run1 = run_with_oracle(offline_oracle, T)
    offline_oracle.reset()
    run2 = run_with_oracle(offline_oracle, T)

    for t in range(T + 1):
        assert np.allclose(
            run1[t], run2[t], atol=1e-10
        ), f"Theta mismatch between runs at t={t}"
        assert np.allclose(run1[t], thetas_recorded[t], atol=1e-10), (
            f"OfflineOracle theta mismatch at t={t}: "
            f"recorded={thetas_recorded[t]}, observed={run1[t]}"
        )


# ============================================================================
# 8. EDGE CASES & ERROR HANDLING
# ============================================================================
def test_metrics_collection_rho_validation():
    """
    MetricsCollection must validate consistent rho values across metrics.

    Scientific basis: Theoretical validity of Lyapunov analysis requires
    consistent Hölder exponent rho across all rho-aware metrics.
    """
    # Valid: consistent rho values
    valid_metrics = [
        LyapunovMetric(rho=0.7),
        NormalizedLyapunovMetric(rho=0.7, drift_magnitude_A=0.1),
    ]
    MetricsCollection(valid_metrics)

    # Invalid: inconsistent rho values
    invalid_metrics = [
        LyapunovMetric(rho=0.5),
        NormalizedLyapunovMetric(rho=0.8, drift_magnitude_A=0.1),
    ]
    with pytest.raises(ValueError, match="Inconsistent Hölder exponents"):
        MetricsCollection(invalid_metrics)


def test_oracle_query_outside_active_step():
    """
    Querying Oracle outside active step must raise clear error.

    Enforces start_step() → query() → end_step() sequence.
    Prevents accidental lookahead or state corruption.
    """
    env = DynamicEnvironment(
        dim=1,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=1, condition_number=1.0),
    )
    oracle = FirstOrderOracle(environment=env)

    with pytest.raises(RuntimeError, match="outside active step"):
        oracle.query(np.array([1.0]))

    oracle.start_step(0)
    oracle.query(np.array([1.0]))
    oracle.end_step()

    with pytest.raises(RuntimeError, match="outside active step"):
        oracle.query(np.array([1.0]))


# ============================================================================
# 9. JSON EXPORT (Reproducibility signature)
# ============================================================================
def test_experiment_result_json_export():
    """
    Exported JSON must contain complete reproducibility signature.

    Scientific basis: Reproducibility requirement for scientific publication.
    Single-file export enables independent verification of results.
    """
    dim = 2
    T = 10

    env = DynamicEnvironment(
        dim=dim,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(dim=dim, condition_number=2.0),
    )
    oracle = FirstOrderOracle(environment=env)
    metrics = MetricsCollection([TrackingErrorMetric(norm="l2")])

    runner = BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        record_trajectory=True,
    )

    optimizer = MockOptimizer(dim=dim, lr=0.1)
    result = runner.run(optimizer, T=T, x0=np.array([1.0, 0.0]), seed=123)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "experiment.json"
        result.save_to_json(str(filepath))

        assert filepath.exists()
        with open(filepath) as f:
            exported = json.load(f)

        # Verify critical reproducibility fields
        assert exported["_format"] == "sthrd-v1.0"
        assert exported["_has_ground_truth"] is True
        assert "optimizer_info" in exported
        assert "environment_config" in exported
        assert "trajectory" in exported
        assert len(exported["trajectory"]["x"]) == T + 1


# ============================================================================
# 10. REPRODUCIBILITY (Independent runs with identical seeds)
# ============================================================================
def test_reproducibility_independent_runs():
    """
    Verify reproducibility across independent runs with identical seed configuration.

    Scientific basis: Reproducibility is fundamental requirement for scientific benchmarks.
    Components using np.random.default_rng() require explicit seed propagation at init time.
    """
    dim = 3
    T = 30
    seed = 12345

    # Factory for identical environment configuration
    def make_env():
        return DynamicEnvironment(
            dim=dim,
            drift=RandomWalkDrift(sigma=0.05, seed=seed),
            landscape=QuadraticLandscape(dim=dim, condition_number=5.0, seed=seed + 1),
            initial_theta=np.zeros(dim),
        )

    # Run 1
    env1 = make_env()
    oracle1 = FirstOrderOracle(
        environment=env1,
        value_noise=GaussianNoise(sigma=0.01, seed=seed),
        seed=seed,
    )
    metrics1 = MetricsCollection([TrackingErrorMetric(norm="l2")])
    runner1 = BenchmarkRunner(env1, oracle1, metrics1, record_trajectory=True)
    opt1 = MockOptimizer(dim=dim, lr=0.2)
    result1 = runner1.run(opt1, T=T, x0=np.ones(dim) * 2.0, seed=seed)

    # Run 2 (independent instance with identical seed configuration)
    env2 = make_env()
    oracle2 = FirstOrderOracle(
        environment=env2,
        value_noise=GaussianNoise(sigma=0.01, seed=seed),
        seed=seed,
    )
    metrics2 = MetricsCollection([TrackingErrorMetric(norm="l2")])
    runner2 = BenchmarkRunner(env2, oracle2, metrics2, record_trajectory=True)
    opt2 = MockOptimizer(dim=dim, lr=0.2)
    result2 = runner2.run(opt2, T=T, x0=np.ones(dim) * 2.0, seed=seed)

    # Verify identical trajectories
    assert np.allclose(
        np.array(result1.trajectory["x"]),
        np.array(result2.trajectory["x"]),
        atol=1e-10,
    ), "Non-reproducible execution across independent runs with identical seeds"

    # Verify identical ground truth trajectories
    assert np.allclose(
        np.array(result1.trajectory["theta"]),
        np.array(result2.trajectory["theta"]),
        atol=1e-10,
    ), "Non-reproducible ground truth trajectory"


# ============================================================================
# 11. REGRESSION TESTS (guard against previously-fixed defects)
# ============================================================================
def test_drift_reset_restores_rng_state():
    """
    REGRESSION (bug 1): Drift.reset() must restore the RNG so a re-run reproduces
    the identical theta_t sequence. Previously reset() was a no-op, so re-running
    the same environment instance silently diverged. Works even for seed=None.
    """
    drift = RandomWalkDrift(sigma=0.1, seed=None)
    theta0 = np.zeros(3)

    def roll():
        th = theta0.copy()
        out = []
        for t in range(8):
            th = drift.step(th, t)
            out.append(th.copy())
        return np.array(out)

    seq1 = roll()
    drift.reset()
    seq2 = roll()
    assert np.allclose(seq1, seq2), "RandomWalkDrift.reset() did not restore RNG state"


def test_noise_reset_restores_rng_state():
    """
    REGRESSION (bug 1): Noise.reset() must restore the RNG so repeated runs apply
    an identical noise sequence. Previously additive noises had no reset().
    """
    noise = GaussianNoise(sigma=1.0, seed=None)
    signal = np.zeros(4)

    def roll():
        return np.array([noise.apply(signal.copy(), t) for t in range(8)])

    a1 = roll()
    noise.reset()
    a2 = roll()
    assert np.allclose(a1, a2), "GaussianNoise.reset() did not restore RNG state"


def test_runner_rerun_same_instances_reproducible():
    """
    REGRESSION (bug 1): Re-running the SAME runner/env/oracle instances with the
    same seed must reproduce identical trajectories (theta and x). Previously the
    environment RNG was never reset, so the second run diverged.
    """
    dim = 3
    T = 25
    env = DynamicEnvironment(
        dim=dim,
        drift=RandomWalkDrift(sigma=0.05, seed=7),
        landscape=QuadraticLandscape(dim=dim, condition_number=4.0, seed=8),
        initial_theta=np.zeros(dim),
    )
    oracle = FirstOrderOracle(
        environment=env, value_noise=GaussianNoise(sigma=0.02, seed=9), seed=10
    )
    metrics = MetricsCollection([TrackingErrorMetric(norm="l2")])
    runner = BenchmarkRunner(env, oracle, metrics, record_trajectory=True)
    opt = MockOptimizer(dim=dim, lr=0.2)

    r1 = runner.run(opt, T=T, x0=np.ones(dim) * 2.0, seed=42)
    r2 = runner.run(opt, T=T, x0=np.ones(dim) * 2.0, seed=42)

    assert np.allclose(
        np.array(r1.trajectory["theta"]), np.array(r2.trajectory["theta"])
    ), "theta_t not reproducible on re-run with same instances"
    assert np.allclose(
        np.array(r1.trajectory["x"]), np.array(r2.trajectory["x"])
    ), "x_t not reproducible on re-run with same instances"


def test_environment_config_captures_noise():
    """
    REGRESSION (bug 2): EnvironmentConfig.from_environment must record the oracle's
    noise model. Previously it read a non-existent attribute (env._oracle instead of
    env._registered_oracle), so noise_type was ALWAYS 'none'.
    """
    env = DynamicEnvironment(
        dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(dim=2)
    )
    _ = FirstOrderOracle(environment=env, value_noise=GaussianNoise(sigma=0.1))
    cfg = EnvironmentConfig.from_environment(env)

    assert (
        cfg.noise_type == "GaussianNoise"
    ), f"Noise model not captured in config: got noise_type={cfg.noise_type!r}"
    assert cfg.noise_parameters.get("sigma") == 0.1


def test_multiextremal_first_order_optimality_at_theta():
    """
    REGRESSION (bug 3): MultiExtremalLandscape must have its global minimum exactly
    at theta: L(theta, theta) = 0 AND grad L(theta, theta) = 0. The previous
    Gaussian-mixture formulation had a nonzero gradient at theta (random offsets did
    not cancel), so theta was not even a critical point.
    """
    dim = 4
    rng = np.random.default_rng(0)
    for amp in [0.0, 1.0, 5.0]:
        land = MultiExtremalLandscape(k_centers=3, width=amp, seed=1)
        for _ in range(10):
            theta = rng.normal(size=dim) * 3.0
            assert abs(land.loss(theta, theta)) < 1e-9
            assert np.allclose(land.grad(theta, theta), np.zeros(dim), atol=1e-9)


def test_multiextremal_nonnegative_on_grid():
    """
    REGRESSION (bug 3): L(x, theta) >= 0 for all x, including wide ripple amplitude.
    The previous formulation could go negative near clustered local centers.
    """
    land = MultiExtremalLandscape(k_centers=4, width=5.0, seed=2)
    theta = np.array([0.3, -0.7])
    grid = np.linspace(-4.0, 4.0, 41)
    for a in grid:
        for b in grid:
            assert land.loss(np.array([a, b]), theta) >= -1e-9


def test_dynamic_regret_metric_has_no_stray_collection_method():
    """
    REGRESSION (bug 4): DynamicRegretMetric must NOT carry a copy-pasted
    get_current_values() that references self.metrics (which it does not have).
    """
    metric = DynamicRegretMetric()
    assert not hasattr(
        metric, "get_current_values"
    ), "DynamicRegretMetric still exposes the stray MetricsCollection method"


def test_batch_metadata_has_no_bogus_drift_rho():
    """
    REGRESSION (bug 5): batch metadata must not contain the always-zero, mislabeled
    'drift_speed_rho' field; it should expose the real drift parameters instead.
    """

    def env_factory():
        return DynamicEnvironment(
            dim=2,
            drift=LinearDrift(velocity=np.array([0.1, 0.0])),
            landscape=QuadraticLandscape(dim=2, condition_number=2.0),
        )

    batch = BatchRunner(
        environment_factory=env_factory,
        oracle_factory=lambda env: FirstOrderOracle(environment=env),
        metric_factory=lambda: MetricsCollection([TrackingErrorMetric(norm="l2")]),
    )
    results = batch.run(lambda: MockOptimizer(dim=2, lr=0.1), seeds=[1], T=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "batch.json"
        batch.save_batch_to_json(results, str(path), algorithm_name="MockSGD")
        data = json.loads(path.read_text(encoding="utf-8"))

    meta = data["batch_metadata"]
    assert "drift_speed_rho" not in meta, "Bogus 'drift_speed_rho' field reintroduced"
    assert "drift_parameters" in meta


# ============================================================================
# 12. SPEC-CONFORMANCE TESTS (Tables 2-3 alignment)
# ============================================================================
def test_pnorm_conditioning_and_invariant():
    """
    Table 3 (p-norm): conditioning matrix M_kappa is applied; kappa=1 recovers the
    plain p-norm; the optimum stays at theta (L=0, grad=0).
    """
    dim = 5
    theta = np.random.randn(dim)

    # kappa = 1 -> identical to the plain (1/p)||x-theta||_p^p
    plain = PNormLandscape(p=1.5, condition_number=1.0)
    x = theta + np.random.randn(dim)
    expected = np.sum(np.abs(x - theta) ** 1.5) / 1.5
    assert np.isclose(plain.loss(x, theta), expected)

    # kappa > 1 -> conditioned, but invariant at theta must hold
    cond = PNormLandscape(p=1.5, condition_number=100.0, seed=3)
    assert abs(cond.loss(theta, theta)) < 1e-9
    assert np.allclose(cond.grad(theta, theta), np.zeros(dim), atol=1e-9)
    # Conditioning actually changes the geometry
    assert not np.isclose(cond.loss(x, theta), plain.loss(x, theta))


def test_adaptive_drift_uses_coordinatewise_sign():
    """
    Table 2 (adaptive): evasion mode must implement theta - alpha*sign(x - theta)
    coordinate-wise (not an L2-normalized direction).
    """
    alpha = 0.1
    drift = AdaptiveDrift(alpha=alpha, mode="evasion")
    theta = np.array([0.0, 0.0, 0.0])
    x = np.array([2.0, -5.0, 0.0])  # different magnitudes per coordinate

    nxt = drift.step(theta, t=0, action=x)
    expected = theta - alpha * np.sign(x - theta)  # [-0.1, +0.1, 0.0]
    assert np.allclose(nxt, expected)


def test_sparse_drift_updates_exactly_k_coords():
    """
    Table 2 (sparse): exactly k coordinates change each step; reset restores RNG.
    """
    dim, k = 6, 2
    drift = SparseDrift(dim=dim, k=k, sigma=0.5, seed=11)
    theta = np.zeros(dim)
    nxt = drift.step(theta, t=0)
    assert np.count_nonzero(nxt - theta) == k

    # Reachable via the factory registry
    d2 = make_drift("sparse", dim=dim, k=k, seed=11)
    assert d2.step(np.zeros(dim), 0).shape == (dim,)


def test_simplex_landscape_invariant_and_projection():
    """
    Table 3 (simplex): L(theta,theta)=0, grad=0, and project() lands on Delta^{d-1}.
    """
    dim = 4
    theta = np.array([0.25, 0.25, 0.25, 0.25])
    land = SimplexLandscape()
    assert abs(land.loss(theta, theta)) < 1e-12
    assert np.allclose(land.grad(theta, theta), np.zeros(dim))

    p = SimplexLandscape.project(np.array([0.5, 1.2, -0.3, 2.0]))
    assert np.all(p >= -1e-12)
    assert np.isclose(np.sum(p), 1.0)


# ============================================================================
# 13. ORACLE / METRIC SPEC-CONFORMANCE (Tables 1, 5)
# ============================================================================
def test_first_order_oracle_blind_by_default():
    """
    Table 1: a pure First-Order oracle exposes ONLY the gradient by default
    (value=None). Opt in with blind_value=False to also get value.
    """
    env = DynamicEnvironment(
        dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(dim=2)
    )
    oracle = FirstOrderOracle(environment=env)  # default
    oracle.start_step(0)
    obs = oracle.query(np.array([1.0, 1.0]))
    oracle.end_step()
    assert obs.value is None, "Default FO oracle must hide value (Table 1)"
    assert obs.grad is not None

    oracle2 = FirstOrderOracle(environment=env, blind_value=False)
    oracle2.start_step(0)
    obs2 = oracle2.query(np.array([1.0, 1.0]))
    oracle2.end_step()
    assert obs2.value is not None


def test_hybrid_oracle_returns_value_and_grad():
    """
    Table 1 (Hybrid): simultaneous value+grad with independent noise sources.
    """
    env = DynamicEnvironment(
        dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(dim=2)
    )
    oracle = HybridOracle(
        environment=env,
        value_noise=GaussianNoise(sigma=0.1, seed=1),
        grad_noise=GaussianNoise(sigma=0.1, seed=2),
    )
    oracle.start_step(0)
    obs = oracle.query(np.array([1.0, 0.5]))
    oracle.end_step()
    assert obs.value is not None and obs.grad is not None
    assert obs.mode == "hybrid"


def test_scheduled_oracle_time_multiplexes_modes():
    """
    ScheduledOracle (renamed from the old HybridOracle) alternates FO/ZO by schedule:
    FO phase returns value+grad, ZO phase returns value only (grad=None).
    """
    env = DynamicEnvironment(
        dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(dim=2)
    )
    oracle = ScheduledOracle(
        environment=env, schedule=[("first-order", 1), ("zero-order", 1)]
    )
    oracle.start_step(0)
    fo = oracle.query(np.array([1.0, 0.0]))
    oracle.end_step()
    oracle.start_step(1)
    zo = oracle.query(np.array([1.0, 0.0]))
    oracle.end_step()
    assert fo.grad is not None  # FO phase
    assert zo.grad is None  # ZO phase


def test_adaptivity_metric_in_unit_interval():
    """
    Table 5 (Adaptivity): TTR_oracle/TTR_algo in (0, 1], 'maximize' direction.
    Faster recovery -> closer to 1.
    """
    metric = AdaptivityMetric(jump_threshold=1.0, epsilon=0.1, oracle_ttr=1.0)
    assert metric.direction == "maximize"

    # Synthetic scenario: a jump at t=1, then x converges to theta over a few steps.
    theta_seq = [
        np.zeros(2),
        np.array([5.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([5.0, 0.0]),
    ]
    x_seq = [
        np.zeros(2),
        np.zeros(2),
        np.array([3.0, 0.0]),
        np.array([4.9, 0.0]),
        np.array([5.0, 0.0]),
    ]
    for t, (x, th) in enumerate(zip(x_seq, theta_seq)):
        metric.update(t, x, th, observation=None, environment=None)

    val = metric.get_result(tail_fraction=1.0)
    assert 0.0 < val <= 1.0


# ============================================================================
# 14. COMPONENT COVERAGE (drifts, noises, factories) + RANGE VALIDATION
# ============================================================================
def test_cyclic_drift_follows_sine():
    """Table 2: theta_t = center + A*sin(2*pi*(t+1)/T) applied uniformly."""
    center = np.zeros(2)
    drift = CyclicDrift(amplitude=1.0, period=4, center=center)
    nxt = drift.step(center, t=0)  # phase = 2*pi*1/4 = pi/2 -> sin = 1
    assert np.allclose(nxt, center + 1.0)


def test_jump_drift_jumps_on_interval_only():
    """Table 2: theta changes only when (t+1) % interval == 0."""
    drift = JumpDrift(interval=3, jump_magnitude=2.0, dim=3, seed=5)
    theta = np.zeros(3)
    assert np.allclose(drift.step(theta, t=0), theta)  # t+1=1
    assert np.allclose(drift.step(theta, t=1), theta)  # t+1=2
    moved = drift.step(theta, t=2)  # t+1=3 -> jump
    assert np.isclose(np.linalg.norm(moved - theta), 2.0)


def test_heavy_tailed_noise_reset_reproducible():
    noise = HeavyTailedNoise(alpha=2.0, scale=1.0, seed=7)
    sig = np.zeros(5)
    a1 = noise.apply(sig.copy(), 0)
    noise.reset()
    a2 = noise.apply(sig.copy(), 0)
    assert np.allclose(a1, a2)


def test_correlated_noise_reset_clears_state_and_rng():
    noise = CorrelatedNoise(sigma=1.0, phi=0.8, seed=3)
    sig = np.zeros(4)
    seq1 = [noise.apply(sig.copy(), t) for t in range(6)]
    noise.reset()
    seq2 = [noise.apply(sig.copy(), t) for t in range(6)]
    assert np.allclose(seq1, seq2)


def test_quantized_noise_snaps_to_grid():
    q = QuantizedNoise(delta=0.5)
    out = q.apply(np.array([0.26, 0.74, -0.4]), 0)
    assert np.allclose(out, np.array([0.5, 0.5, -0.5]))


def test_factories_build_new_components():
    d = make_drift("sparse", dim=4, k=2, seed=1)
    assert isinstance(d, SparseDrift)
    land = make_landscape("simplex")
    assert isinstance(land, SimplexLandscape)
    n = make_noise("multiplicative", sigma_rel=0.1)
    assert isinstance(n, MultiplicativeNoise)
    env = DynamicEnvironment(dim=2, drift=StationaryDrift(), landscape=land)
    assert isinstance(make_oracle("hybrid", env), HybridOracle)
    assert isinstance(
        make_oracle("scheduled", env, schedule=[("first-order", 5)]), ScheduledOracle
    )


def test_parameter_range_validation():
    with pytest.raises(ValueError):
        QuadraticLandscape(dim=2, condition_number=0.5)  # kappa < 1
    with pytest.raises(ValueError):
        HeavyTailedNoise(alpha=0.0)  # tail index must be > 0
    with pytest.raises(ValueError):
        CorrelatedNoise(sigma=1.0, phi=1.0)  # non-stationary
    with pytest.raises(ValueError):
        SparseDrift(dim=3, k=5)  # k > dim
    with pytest.raises(ValueError):
        CyclicDrift(amplitude=1.0, period=0, center=np.zeros(2))  # period <= 0


def test_batch_runner_adaptive_stopping():
    """Item 8: run_adaptive stops once the 95% CI relative half-width < tol."""

    def env_factory():
        return DynamicEnvironment(
            dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(dim=2)
        )

    batch = BatchRunner(
        environment_factory=env_factory,
        oracle_factory=lambda env: FirstOrderOracle(environment=env),
        metric_factory=lambda: MetricsCollection([TrackingErrorMetric(norm="l2")]),
    )
    results = batch.run_adaptive(
        lambda: MockOptimizer(dim=2, lr=0.1),
        T=15,
        min_runs=3,
        max_runs=20,
        tol=1.0,  # loose -> stops at min_runs
        metric_key="error_l2",
    )
    assert 3 <= len(results) <= 20
    mean, half = BatchRunner.ci95([r.final_metrics["error_l2"] for r in results])
    assert np.isfinite(mean) and np.isfinite(half)


def test_optimizer_info_uses_explicit_oracle_type():
    """Item 13: explicit optimizer.oracle_type overrides the signature heuristic."""

    class ExplicitOpt:
        name = "ExplicitZO"
        oracle_type = "zero-order"

        def step(self, observation):
            return observation.x

        def reset(self):
            pass

    info = OptimizerInfo.from_optimizer(ExplicitOpt())
    assert info.oracle_type == "zero-order"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
