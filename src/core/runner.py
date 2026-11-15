"""
Benchmark orchestration module for WIND.
Manages experiment execution, result collection, and unified export to single files.
"""

import json
import random
import numpy as np
import datetime
import inspect
import time
from copy import deepcopy
from pathlib import Path
from typing import Protocol, Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict, replace

from wind_benchmark.core.oracle import EvaluationRecord, OracleClient
from wind_benchmark.core.privacy import (
    TruthGuard,
    _run,
    _series,
    _internal,
    _bind,
    _bind_environment,
    _user_call,
    _validate_component,
)
from wind_benchmark.core.randomness import (
    preserve_global_rng,
    seeded_factory,
    factory_accepts_seed,
    stream_seed,
    reseed_component,
    reseed_environment,
    reseed_oracle,
)
from wind_benchmark.reporting.common import json_safe, write_json, safe_print as print
from wind_benchmark.core.metrics import validate_tail, ensemble_metrics
from wind_benchmark.core.serialization import (
    environment_snapshot,
    restore_environment,
    snapshot_restorable,
)

# ============================================================================
# OPTIMIZER PROTOCOL (Interface for all algorithms)
# ============================================================================
from wind_benchmark.algorithms.protocol import (
    OptimizerProtocol,
    TaskSpec,
    AdvanceResult,
    require_algorithm,
)


@dataclass
class OptimizerInfo:
    """
    Complete optimizer signature for scientific reproducibility.

    Contains everything needed to reconstruct the exact algorithm configuration:
      - Identity (name, class, module)
      - Hyperparameters (lr, betas, momentum, etc.)
      - Computational properties (oracle type, query cost, memory complexity)
    """

    name: str
    class_name: str
    module: str
    hyperparameters: Dict[str, Any]
    oracle_type: str  # 'first-order', 'zero-order', 'hybrid'
    query_cost_per_step: int | str  # Integer or declared dimension/batch formula
    memory_complexity: str  # Asymptotic memory requirement (e.g., 'O(d)', 'O(d²)')
    implementation_id: str = "unspecified"
    update_schedule: str = "one-gradient-per-update"
    source: str = "user supplied"
    variant: str = "user supplied"
    implementation_version: int = 1
    rng_stream_id: str = "user supplied"

    @classmethod
    def from_optimizer(cls, optimizer: Any) -> "OptimizerInfo":
        require_algorithm(optimizer)
        parameters = dict(getattr(optimizer, "parameters", {}))
        return cls(
            name=getattr(optimizer, "canonical_name", type(optimizer).__name__),
            class_name=type(optimizer).__name__,
            module=type(optimizer).__module__,
            hyperparameters=parameters,
            oracle_type=getattr(optimizer, "oracle_type", "unknown"),
            query_cost_per_step=getattr(
                optimizer, "query_cost_per_step", "declared by algorithm"
            ),
            memory_complexity=getattr(optimizer, "memory_complexity", "unknown"),
            implementation_id=getattr(optimizer, "implementation_id", "user supplied"),
            update_schedule=getattr(optimizer, "update_schedule", "user supplied"),
            source=getattr(optimizer, "source", "user supplied"),
            variant=getattr(optimizer, "variant", "user supplied"),
            implementation_version=getattr(optimizer, "implementation_version", 1),
            rng_stream_id=getattr(optimizer, "rng_stream_id", "user supplied"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


# ============================================================================
# ENVIRONMENT CONFIGURATION (Reproducible environment signature)
# ============================================================================
@dataclass
class EnvironmentConfig(TruthGuard):
    """
    Complete environment configuration for reproducibility.

    Captures all parameters needed to reconstruct the exact experimental setting:
      - Dimensionality and bounds
      - Drift dynamics (type and parameters)
      - Landscape geometry (type and parameters)
      - Noise characteristics (type and parameters)
    """

    dim: int
    drift_type: str
    drift_parameters: Dict[str, Any]
    landscape_type: str
    landscape_parameters: Dict[str, Any]
    noise_type: str
    noise_parameters: Dict[str, Any]
    bounds: Optional[Tuple[float, float]] = None
    reconstruction: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_environment(cls, env: Any) -> "EnvironmentConfig":
        """
        Extract complete configuration from environment instance.

        Automatically collects parameters from drift, landscape, and noise objects.
        """
        # Drift configuration
        drift_type = env.drift.__class__.__name__
        drift_params = {}
        drift_attrs = [
            "velocity",
            "sigma",
            "alpha",
            "threshold",
            "mode",
            "period",
            "amplitude",
            "center",
            "interval",
            "jump_magnitude",
        ]
        for attr in drift_attrs:
            if hasattr(env.drift, attr):
                try:
                    val = getattr(env.drift, attr)
                    if isinstance(val, (int, float, str, list, dict, bool, np.ndarray)):
                        # Convert numpy arrays to lists for serialization
                        if isinstance(val, np.ndarray):
                            drift_params[attr] = val.tolist()
                        else:
                            drift_params[attr] = val
                except Exception:
                    continue

        # Landscape configuration
        landscape_type = env.landscape.__class__.__name__
        landscape_params = {}
        landscape_attrs = [
            "condition_number",
            "p",
            "k_centers",
            "width",
            "delta",
            "a",
            "b",
            "scale",
        ]
        for attr in landscape_attrs:
            if hasattr(env.landscape, attr):
                try:
                    val = getattr(env.landscape, attr)
                    if isinstance(val, (int, float, str, bool)):
                        landscape_params[attr] = val
                except Exception:
                    continue
        if hasattr(env.landscape, "kappa"):
            landscape_params["condition_number"] = float(env.landscape.kappa)

        # Noise configuration (from oracle if available)
        noise_type = "none"
        noise_params = {}

        # Try to get noise from environment's registered oracle reference
        oracle = getattr(env, "_registered_oracle", None)
        if oracle is not None:
            # Prefer value noise; fall back to gradient noise if value noise absent
            noise_obj = getattr(oracle, "value_noise", None) or getattr(
                oracle, "grad_noise", None
            )
            if noise_obj is not None:
                noise_type = noise_obj.__class__.__name__
                noise_attrs = ["sigma", "alpha", "scale", "phi", "p", "lam"]
                for attr in noise_attrs:
                    if hasattr(noise_obj, attr):
                        try:
                            val = getattr(noise_obj, attr)
                            if isinstance(val, (int, float, str, bool)):
                                noise_params[attr] = val
                        except Exception:
                            continue

        return cls(
            dim=env.dim,
            drift_type=drift_type,
            drift_parameters=drift_params,
            landscape_type=landscape_type,
            landscape_parameters=landscape_params,
            noise_type=noise_type,
            noise_parameters=noise_params,
            bounds=getattr(env, "bounds", None),
            reconstruction=environment_snapshot(env),
        )

    def restore(self):
        """Reconstruct the built-in environment and both oracle noise channels."""
        if not self.reconstruction:
            raise ValueError(
                "Historical metadata lacks a complete construction snapshot"
            )
        return restore_environment(self.reconstruction)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)


# ============================================================================
# EXPERIMENT RESULT (Unified export container)
# ============================================================================
@dataclass
class ExperimentResult(TruthGuard):
    """
    Complete experiment result with unified export capabilities.

    Scientific guarantees:
       Full reproducibility (optimizer + environment signatures)
       Ground truth inclusion (theta_t trajectory for regret calculation)
       Metric consistency (tail-processed final metrics + full history)
       Single-file export (JSON/CSV with all data)

    Usage:
        result = runner.run(optimizer, T=500, record_trajectory=True)
        result.save_to_json("experiment.json")  # Single file with everything
        result.save_to_csv("experiment.csv")    # Long-format for analysis tools
    """

    # Execution status
    status: str  # "SUCCESS" or "ERROR"
    error_message: Optional[str]

    # Reproducibility signatures
    optimizer_info: OptimizerInfo
    environment_config: EnvironmentConfig

    # Performance metrics
    final_metrics: Dict[str, float]  # Tail-processed aggregates
    metrics_history: Dict[str, List[float]]  # Full time series

    # Trajectories (x_t = algorithm path, theta_t = ground truth optimum)
    trajectory: Optional[Dict[str, List[Union[np.ndarray, float]]]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime: float = 0.0  # Execution time in seconds
    evaluation_log: Optional[List[EvaluationRecord]] = None
    metric_definitions: Dict[str, Any] = field(default_factory=dict)
    metric_status: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate critical invariants."""
        # Ensure ground truth is available if trajectory is recorded
        if self.trajectory is not None:
            assert "x" in self.trajectory, "Trajectory missing 'x' (algorithm path)"
            assert (
                "theta" in self.trajectory
            ), "Trajectory missing 'theta' (ground truth optimum)"
            assert len(self.trajectory["x"]) == len(
                self.trajectory["theta"]
            ), "Trajectory length mismatch between x and theta"

    def metric_value(self, metric_id: str) -> float:
        """Resolve a scalar by versioned identity, independent of display labels."""
        keys = [
            key
            for key, definition in self.metric_definitions.items()
            if definition["id"] == metric_id and key in self.final_metrics
        ]
        if len(keys) != 1:
            raise KeyError(f"No unique scalar result for metric ID {metric_id}")
        return self.final_metrics[keys[0]]

    def to_dict(self) -> Dict[str, Any]:
        """Schema 2: one traversal, numeric arrays, strict JSON-compatible nulls."""
        result = json_safe(self)
        result["_format"] = "wind-result-v2"
        result["_schema_version"] = 4
        result["_has_ground_truth"] = self.trajectory is not None
        result["_environment_restorable"] = snapshot_restorable(
            self.environment_config.reconstruction
        )
        result["_reproducible"] = (
            result["_environment_restorable"]
            and self.optimizer_info.implementation_id != "unspecified"
            and self.metadata.get("seed") is not None
        )
        return result

    def save_to_json(self, filepath: str) -> Path:
        path = write_json(filepath, self.to_dict(), prepared=True)
        print(f"Saved experiment: {path}")
        print(
            f"Optimizer: {self.optimizer_info.name}; trajectory: {self.trajectory is not None}"
        )
        return path

    def save_to_csv(self, filepath: str) -> Path:
        """Long schema 4: actual coordinate rows and every actual metric history."""
        import pandas as pd

        if self.trajectory is None:
            raise ValueError(
                "Trajectory not recorded. Use record_trajectory=True in BenchmarkRunner."
            )
        steps = max((len(v) for v in self.metrics_history.values()), default=0)
        if not steps:
            raise ValueError("Metric history is empty")
        xs = np.asarray(self.trajectory["x"][-steps:]).reshape(steps, -1)
        thetas = np.asarray(self.trajectory["theta"][-steps:]).reshape(steps, -1)
        common = {
            "experiment_id": self.metadata.get("experiment_id", "unknown"),
            "optimizer_name": self.optimizer_info.name,
            "optimizer_class": self.optimizer_info.class_name,
            "optimizer_implementation_id": self.optimizer_info.implementation_id,
            "optimizer_update_schedule": self.optimizer_info.update_schedule,
            "oracle_type": self.optimizer_info.oracle_type,
            "schema_version": 4,
        }
        common.update(
            {
                f"optimizer_{k}": v
                for k, v in self.optimizer_info.hyperparameters.items()
                if isinstance(v, (int, float, str, bool))
            }
        )
        rows = []
        for step, (x, theta) in enumerate(zip(xs, thetas)):
            for dimension, (value, optimum) in enumerate(zip(x, theta)):
                rows.append(
                    {
                        **common,
                        "step": step,
                        "dim": dimension,
                        "x_value": float(value),
                        "theta_value": float(optimum),
                        "coordinate": f"x_{dimension}",
                        "is_optimal": False,
                        "metric_name": "decision_coordinate",
                        "metric_value": None,
                    }
                )
            for name, values in self.metrics_history.items():
                if step >= len(values):
                    continue
                definition = self.metric_definitions.get(name, {})
                rows.append(
                    {
                        **common,
                        "step": step,
                        "dim": -1,
                        "x_value": values[step],
                        "theta_value": None,
                        "coordinate": "metric",
                        "is_optimal": False,
                        "metric_name": name,
                        "metric_value": values[step],
                        "metric_id": definition.get("id"),
                        "metric_status": self.metric_status.get(name, {}).get("state"),
                    }
                )
        path = Path(filepath)
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing result: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved CSV: {path}")
        return path

    def get_ground_truth(self) -> np.ndarray:
        """
        Extract ground truth optimum trajectory theta_t.

        Returns:
            Array of shape (T, d) containing theta_t for all timesteps.

        Raises:
            ValueError: If trajectory was not recorded.
        """
        if self.trajectory is None or "theta" not in self.trajectory:
            raise ValueError(
                "Ground truth not available. Use record_trajectory=True in BenchmarkRunner."
            )
        return np.array(self.trajectory["theta"])

    def get_algorithm_path(self) -> np.ndarray:
        """
        Extract algorithm trajectory x_t.

        Returns:
            Array of shape (T, d) containing x_t for all timesteps.
        """
        if self.trajectory is None or "x" not in self.trajectory:
            raise ValueError("Trajectory not recorded.")
        return np.array(self.trajectory["x"])


# ============================================================================
# BENCHMARK RUNNER (Single experiment execution)
# ============================================================================
class BenchmarkRunner(TruthGuard):
    """
    Executes a single optimization experiment in dynamic environment.


    Example:
        runner = BenchmarkRunner(env, oracle, metrics, record_trajectory=True)
        result = runner.run(optimizer, T=500, x0=np.zeros(5), seed=42)
        result.save_to_json("results/adam_5d.json")
    """

    _public_during_run = frozenset({"run", "environment", "oracle"})

    def __init__(
        self,
        environment: Any,
        oracle: Any,
        metrics: Any,
        record_trajectory: bool = False,
        tail_fraction: float = 0.2,
    ):
        """
        Initialize benchmark runner.

        Args:
            environment: DynamicEnvironment instance
            oracle: OracleProtocol instance (FirstOrder/ZeroOrder/Hybrid)
            metrics: MetricsCollection instance
            record_trajectory: If True, record full x_t and theta_t trajectories
            tail_fraction: Fraction of trajectory to use for final metrics (e.g., 0.2 = last 20%)
        """
        self.environment = environment
        self.oracle = oracle
        _validate_component(metrics)
        for metric in metrics.metrics:
            _validate_component(metric)
            if getattr(metric, "decision_stream", "committed") != "committed":
                raise ValueError("BenchmarkRunner evaluates committed iterates only")
            if (
                type(environment.landscape).__name__
                in ("StiefelLandscape", "GrassmannLandscape")
                and metric.rho is not None
            ):
                raise ValueError(
                    "Lyapunov metrics with rho are defined for vector tasks, not manifolds"
                )
        self.metrics = metrics
        self._metric_template = deepcopy(metrics)
        self.record_trajectory = record_trajectory
        self.tail_fraction = validate_tail(tail_fraction)

        # Internal state
        self.trajectory = None
        self.metrics_history = {}
        self._step = 0

    @_run
    @preserve_global_rng
    def run(
        self,
        optimizer: OptimizerProtocol,
        T: int,
        x0: np.ndarray,
        seed: Optional[int] = None,
        reseed_components: bool = True,
    ):
        """
        Execute optimization experiment for T timesteps.

        Scientific guarantees:
          ✓ Non-anticipative protocol (x_t committed before f_t evaluation)
          ✓ Temporal consistency (all queries at step t see identical theta_t)
          ✓ Ground truth recording (theta_t saved for regret calculation)
          ✓ Reproducibility (fixed seed → identical trajectories)

        Args:
            optimizer: Algorithm implementing OptimizerProtocol
            T: Number of optimization steps
            x0: Initial query point
            seed: Root seed for algorithm and dynamic observation streams. The
                configured landscape and initial state remain part of the task.
            reseed_components: False replays the components' construction-time
                streams (e.g. a restored evaluation snapshot).

        Returns:
            ExperimentResult containing:
              - Optimizer signature
              - Environment configuration
              - Final metrics (tail-processed)
              - Full metric history
              - Trajectories (x_t and theta_t if record_trajectory=True)
        """
        # ← MUST BE FIRST LINE OF METHOD

        _bind(self)
        _bind_environment(self.environment, self.oracle)
        x0 = _user_call(np.asarray, x0, dtype=float)
        if type(T) not in (int, np.int32, np.int64) or T < 1:
            raise ValueError("T must be a positive integer")
        if seed is not None and type(seed) not in (
            int,
            np.int32,
            np.int64,
            np.uint32,
            np.uint64,
        ):
            raise ValueError("seed must be an integer")
        if type(reseed_components) is not bool:
            raise ValueError("reseed_components must be a boolean")
        _validate_component(self._metric_template)
        for metric in self._metric_template.metrics:
            _validate_component(metric)
        _user_call(require_algorithm, optimizer)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            if reseed_components:
                reseed_environment(self.environment, stream_seed(seed, "environment"))
                reseed_oracle(self.oracle, stream_seed(seed, "oracle"))
        start_time = time.time()
        self.trajectory = None
        self.metrics_history = {}
        self.metrics = deepcopy(self._metric_template)
        self.metrics.reset()
        run_metrics = deepcopy(self.metrics)
        self.environment.reset()
        self.oracle.reset()
        landscape = self.environment.landscape
        kind = getattr(landscape, "geometry", None)
        if kind is None:
            class_name = type(landscape).__name__.lower()
            kind = next(
                (g for g in ("grassmann", "stiefel", "simplex") if g in class_name),
                "euclidean",
            )
        matrix_shape = None
        if kind in {"grassmann", "stiefel"}:
            matrix_shape = (landscape.d, landscape.r)
        task_spec = TaskSpec(self.environment.dim, kind, matrix_shape)
        rng_seed = stream_seed(seed, "algorithm") if seed is not None else None
        _user_call(
            lambda *args: optimizer.initialize(*args),
            x0.copy(),
            np.random.default_rng(rng_seed),
            task_spec,
        )
        optimizer_info = _user_call(OptimizerInfo.from_optimizer, optimizer)
        cost = optimizer_info.query_cost_per_step
        if cost == "2*d":
            cost = 2 * self.environment.dim
        elif cost == "2*m":
            cost = 2 * optimizer_info.hyperparameters["m"]
        if not isinstance(cost, (int, np.integer)) or cost < 1:
            raise ValueError(
                "Declare query_cost_per_step as a positive integer (or catalog 2*d/2*m)"
            )
        budget = T * int(cost)
        initial_theta = self.environment.get_current_theta(for_analysis=True)
        run_metrics.initialize(x0, initial_theta, self.environment)
        trajectory = (
            {
                "x": [x0.copy()],
                "theta": [initial_theta.copy()],
                "action": [np.zeros_like(x0)],
            }
            if self.record_trajectory
            else None
        )
        evaluation_log = [] if self.record_trajectory else None
        recommendations = []
        completed_updates = 0
        metric_index = 0
        last_point = x0.copy()

        @_internal
        def query(point):
            nonlocal metric_index, last_point
            if self.oracle.n_queries >= budget:
                raise RuntimeError("Query budget exhausted")
            response = self.oracle.query(point)
            record = self.oracle._take_evaluation()
            run_metrics.update(
                metric_index, response.x, record.theta, record, self.environment
            )
            metric_index += 1
            last_point = response.x.copy()
            if evaluation_log is not None:
                evaluation_log.append(record)
                trajectory["x"].append(response.x.copy())
                trajectory["theta"].append(record.theta.copy())
                trajectory["action"].append(response.x.copy())
            return response

        for t in range(T):
            self._step = t
            if self.oracle.n_queries >= budget:
                break
            self.oracle.start_step(t)
            client = OracleClient(query, lambda: budget - self.oracle.n_queries)
            before = self.oracle.n_queries
            try:
                update = _user_call(lambda: optimizer.advance(client))
                if type(update) is not AdvanceResult:
                    raise TypeError("advance(client) must return AdvanceResult")
                if update.queries != self.oracle.n_queries - before:
                    raise ValueError("Algorithm's query count disagrees with oracle")
                if update.queries == 0:
                    raise ValueError(
                        "advance must consume queries while budget remains"
                    )
                if (
                    not np.isfinite(update.recommendation).all()
                    or update.recommendation.shape != x0.shape
                ):
                    raise FloatingPointError("Invalid algorithm recommendation")
                completed_updates += int(update.update_complete)
                recommendations.append(
                    {
                        "query_index": self.oracle.n_queries,
                        "update_complete": update.update_complete,
                        "x": update.recommendation.tolist(),
                    }
                )
            finally:
                client._close()
                self.oracle._last_evaluation = None
                self.oracle.end_step()
            # Ordinary runner uses one environment tick per attempted update.
            # Query-clock frozen/streaming experiments use OracleSession instead.
            self.environment.step(action=last_point)

        # Final metric computation (tail-processed)
        run_metrics.finish(self.oracle.n_queries)
        final_metrics = run_metrics.get_results(tail_fraction=self.tail_fraction)
        metrics_history = run_metrics.get_histories()

        # Create optimizer and environment signatures
        optimizer_info = _user_call(OptimizerInfo.from_optimizer, optimizer)
        environment_config = EnvironmentConfig.from_environment(self.environment)

        # Metadata
        metadata = {
            "privacy_protocol": "local_series_v2",
            "metrics_protocol": "metrics_v2",
            "export_schema_version": 4,
            "query_counts": {
                "total": self.oracle.n_queries,
                "value": self.oracle.n_value_queries,
                "gradient": self.oracle.n_grad_queries,
            },
            "decision_stream": "queried-point",
            "time_protocol": "frozen-per-update",
            "query_budget": budget,
            "completed_updates": completed_updates,
            "recommendations": recommendations,
            "experiment_id": f"sthrd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": seed,
            "rng_protocol": "named_streams_v2",
            "reseed_components": reseed_components,
            "initial_x": np.asarray(x0).tolist(),
            "initialization": {"kind": "fixed", "x0": np.asarray(x0).tolist()},
            "time_grid": {
                "kind": "query_index",
                "start": 1,
                "step": 1,
                "size": self.oracle.n_queries,
            },
            "total_steps": T,
            "dimension": self.environment.dim,
            "oracle_type": optimizer_info.oracle_type,
            "optimizer_implementation_id": optimizer_info.implementation_id,
            "optimizer_update_schedule": optimizer_info.update_schedule,
            "record_trajectory": self.record_trajectory,
            "tail_fraction": self.tail_fraction,
        }

        # Runtime
        runtime = time.time() - start_time

        # Construct result
        result = ExperimentResult(
            status="SUCCESS",
            error_message=None,
            optimizer_info=optimizer_info,
            environment_config=environment_config,
            final_metrics=final_metrics,
            metrics_history=metrics_history,
            trajectory=trajectory,
            metadata=metadata,
            runtime=runtime,
            evaluation_log=evaluation_log,
            metric_definitions=run_metrics.get_definitions(),
            metric_status=run_metrics.get_status(self.tail_fraction),
        )
        _bind(environment_config)
        _bind(result)

        # Publish only after every optimizer call and result validation succeeded.
        self.metrics = run_metrics
        self.metrics_history = metrics_history
        self.trajectory = trajectory
        return result


# ============================================================================
# BATCH RUNNER (Multiple seeds/experiments)
# ============================================================================
class BatchRunner:
    """
    Executes multiple experiments with different seeds for statistical significance.

    Aggregates results across seeds and provides unified export.

    Example:
        batch = BatchRunner(env_config, oracle_config, optimizer_factories)
        results = batch.run(dim=5, rho=0.3, seeds=[42,43,44,45,46], T=500)
        batch.save_to_json("batch_results.json")
    """

    def __init__(
        self,
        environment_factory: callable,
        oracle_factory: callable,
        metric_factory: callable,
        record_trajectory: bool = False,
        tail_fraction: float = 0.2,
    ):
        """
        Initialize batch runner.

        Args:
            environment_factory: Callable returning DynamicEnvironment
            oracle_factory: Callable returning OracleProtocol
            metric_factory: Callable returning MetricsCollection
            record_trajectory: If True, record trajectories for all seeds
            tail_fraction: Fraction for tail processing
        """
        self.environment_factory = environment_factory
        self.oracle_factory = oracle_factory
        self.metric_factory = metric_factory
        self.record_trajectory = record_trajectory
        self.tail_fraction = tail_fraction

    @_series
    def run(
        self,
        optimizer_factory: callable,
        seeds: List[int],
        T: int = 500,
        initial_x: Optional[np.ndarray] = None,
        **env_kwargs,
    ) -> List[ExperimentResult]:
        """
        Run experiment across multiple seeds.

        Args:
            optimizer_factory: Callable returning OptimizerProtocol instance
            seeds: List of random seeds for reproducibility
            T: Number of steps per experiment
            **env_kwargs: Additional kwargs for environment_factory

        Returns:
            List of ExperimentResult objects (one per seed)
        """
        results = []

        for seed in seeds:
            print(f"Running seed {seed}...", end=" ")

            # Create fresh components for each seed
            env = _user_call(
                seeded_factory,
                self.environment_factory,
                stream_seed(seed, "environment"),
                **env_kwargs,
            )
            if not factory_accepts_seed(self.environment_factory):
                reseed_environment(
                    env, stream_seed(seed, "environment"), landscape=True
                )
            oracle = _user_call(
                seeded_factory, self.oracle_factory, stream_seed(seed, "oracle"), env
            )
            metrics = _user_call(self.metric_factory)
            optimizer = _user_call(
                seeded_factory, optimizer_factory, stream_seed(seed, "algorithm")
            )

            # Run experiment
            runner = BenchmarkRunner(
                env,
                oracle,
                metrics,
                record_trajectory=self.record_trajectory,
                tail_fraction=self.tail_fraction,
            )
            x0 = (
                np.random.default_rng(seed).normal(0.0, 0.1, size=env.dim)
                if initial_x is None
                else np.asarray(initial_x, dtype=float).copy()
            )
            result = runner.run(optimizer, T=T, x0=x0, seed=seed)

            result.metadata["expected_series_seeds"] = list(seeds)
            result.metadata["initialization"] = (
                {
                    "kind": "iid_normal",
                    "mean": 0.0,
                    "std": 0.1,
                }
                if initial_x is None
                else {"kind": "fixed", "x0": x0.tolist()}
            )

            results.append(result)
            print("DONE")

        return results

    @staticmethod
    def summarize(results):
        """Post-release ensemble estimates; run() continues to return a list."""
        return ensemble_metrics(results)

    @staticmethod
    def ci95(values: List[float]) -> Tuple[float, float]:
        """Return (mean, half-width of the 95% confidence interval) via normal approx."""
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        mean = float(np.mean(arr)) if n > 0 else float("nan")
        if n < 2:
            return mean, float("inf")
        std = float(np.std(arr, ddof=1))
        return mean, 1.96 * std / np.sqrt(n)

    @_series
    def run_adaptive(
        self,
        optimizer_factory: callable,
        T: int = 500,
        min_runs: int = 10,
        max_runs: int = 1000,
        tol: float = 0.05,
        metric_key: str = "error_l2",
        base_seed: int = 0,
        initial_x: Optional[np.ndarray] = None,
        **env_kwargs,
    ) -> List[ExperimentResult]:
        """
        Run additional seeds until the 95% CI relative half-width of `metric_key`
        drops below `tol` (Table 5 note), bounded by [min_runs, max_runs].

        Returns the list of ExperimentResult collected so far. Pass initial_x
        explicitly for methods with a prescribed start (for example RDA).
        """
        if min_runs < 2:
            raise ValueError("min_runs must be >= 2 for a confidence interval")
        if max_runs < min_runs:
            raise ValueError("max_runs must be >= min_runs")

        results: List[ExperimentResult] = []
        seed = base_seed
        while len(results) < max_runs:
            env = _user_call(
                seeded_factory,
                self.environment_factory,
                stream_seed(seed, "environment"),
                **env_kwargs,
            )
            if not factory_accepts_seed(self.environment_factory):
                reseed_environment(
                    env, stream_seed(seed, "environment"), landscape=True
                )
            oracle = _user_call(
                seeded_factory, self.oracle_factory, stream_seed(seed, "oracle"), env
            )
            metrics = _user_call(self.metric_factory)
            optimizer = _user_call(
                seeded_factory, optimizer_factory, stream_seed(seed, "algorithm")
            )
            runner = BenchmarkRunner(
                env,
                oracle,
                metrics,
                record_trajectory=self.record_trajectory,
                tail_fraction=self.tail_fraction,
            )
            x0 = (
                np.random.default_rng(seed).normal(0.0, 0.1, size=env.dim)
                if initial_x is None
                else np.asarray(initial_x, dtype=float).copy()
            )
            results.append(runner.run(optimizer, T=T, x0=x0, seed=seed))
            seed += 1

            if len(results) >= min_runs:
                vals = [
                    r.final_metrics[metric_key]
                    for r in results
                    if metric_key in r.final_metrics
                ]
                mean, half = self.ci95(vals)
                rel = half / (abs(mean) + 1e-12)
                if rel < tol:
                    break

        return results

    def save_batch_to_json(
        self,
        results: List[ExperimentResult],
        filepath: str,
        algorithm_name: str = "unknown",
    ) -> Path:
        """
        Save batch of experiments to single JSON file.

        Structure:
          {
            "batch_metadata": {...},
            "algorithm_name": "...",
            "results": [
              {experiment 1},
              {experiment 2},
              ...
            ],
            "aggregated_metrics": {
              "error_l2_mean": 0.45,
              "error_l2_std": 0.08,
              ...
            }
          }
        """
        # Aggregate metrics across seeds
        if not results:
            raise ValueError("Cannot export an empty batch")
        metric_names = results[0].final_metrics.keys()
        aggregated = {}
        for name in metric_names:
            values = [
                r.final_metrics[name]
                for r in results
                if name in r.final_metrics and np.isfinite(r.final_metrics[name])
            ]
            aggregated[f"{name}_n_valid"] = len(values)
            aggregated[f"{name}_n_undefined"] = len(results) - len(values)
            metric_class = results[0].metric_definitions.get(name, {}).get("class")
            aggregated[f"{name}_aggregation"] = (
                "conditional_on_reaching_threshold; primary restricted mean in ensemble_metrics"
                if metric_class == "QueryEfficiencyMetric"
                else "valid_runs_only"
            )
            for label, function in (
                ("mean", np.mean),
                ("std", np.std),
                ("min", np.min),
                ("max", np.max),
            ):
                aggregated[f"{name}_{label}"] = (
                    float(function(values)) if values else None
                )
            aggregated[f"{name}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )

        # Build batch structure
        batch_data = {
            "batch_metadata": {
                "algorithm_name": algorithm_name,
                "num_seeds": len(results),
                "seeds": [r.metadata.get("seed") for r in results],
                "dimension": results[0].environment_config.dim,
                "drift_type": results[0].environment_config.drift_type,
                "drift_parameters": results[0].environment_config.drift_parameters,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "algorithm_name": algorithm_name,
            "results": results,
            "aggregated_metrics": aggregated,
            "ensemble_metrics": ensemble_metrics(results),
            "schema_version": 4,
        }

        # Save
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_json(output_path, batch_data)

        print(f"✅ Saved batch of {len(results)} experiments to: {output_path}")
        print(f"   → Algorithm: {algorithm_name}")
        print(f"   → Seeds: {batch_data['batch_metadata']['seeds']}")
        print(
            f"   Aggregated error_l2: {aggregated.get('error_l2_mean')} +/- {aggregated.get('error_l2_std')}"
        )

        return output_path
