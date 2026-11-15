"""Versioned default algorithm evaluation, independent of the legacy CLI.

    python -m wind_benchmark.experiments.standard --profile smoke
    python -m wind_benchmark.experiments.standard --profile standard

All queries (including exploratory queries) are scored, charged, and advance the
target once. This is a streaming query-position benchmark, not a recommended-
iterate benchmark. Algorithms only receive Observation / a scoped OracleClient.
No Gymnasium behavior or existing experiment is changed by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import zlib
from contextlib import closing
from types import FunctionType
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

# Also support: python src/experiments/standard.py (with the package installed).
if __package__ in {None, ""}:
    if __name__ == "__main__":
        import runpy

        # Register the installed module as __main__ so spawn imports that same
        # package, not this potentially external checkout file.
        runpy.run_module(
            "wind_benchmark.experiments.standard", run_name="__main__", alter_sys=True
        )
        raise SystemExit(0)
    # multiprocessing re-imports a file entrypoint as __mp_main__. Do not
    # execute the CLI again during child bootstrap.
    __package__ = "wind_benchmark.experiments"

import numpy as np
import pandas as pd

from wind_benchmark.algorithms.catalog import (
    CATALOG_VERSION,
    create_optimizer,
    resolve_optimizer_spec,
    canonical_optimizer_names,
)
from wind_benchmark.algorithms.protocol import (
    TaskSpec,
    AdvanceResult,
    require_algorithm,
)
import time
from wind_benchmark.core.environment import (
    Drift,
    DynamicEnvironment,
    SimplexLandscape,
    make_drift,
    make_landscape,
    make_noise,
)
from wind_benchmark.core.geometry import principal_angle_distance, random_stiefel
from wind_benchmark.core.oracle import (
    FirstOrderOracle,
    HybridOracle,
    OracleClient,
    ZeroOrderOracle,
)
from wind_benchmark.core.resources import ResourceSampler
from wind_benchmark.experiments.parallel import (
    CatalogFactory,
    Execution,
    check_cancelled,
    describe_method,
    execution_options,
)
from threadpoolctl import threadpool_limits
from wind_benchmark.core.randomness import stream_seed as identity_seed
from wind_benchmark.core.randomness import preserve_global_rng
from wind_benchmark.reporting.common import (
    json_safe,
    safe_print as print,
    source_hashes,
)
from wind_benchmark.core.privacy import (
    _run,
    _series,
    _internal,
    _bind_environment,
    _user_call,
    _after_release,
    _trusted_file,
)

SUITE_VERSION = "wind_tracking_v1"
DEFAULT_METHODS = canonical_optimizer_names()
ORACLES = {
    "first-order": FirstOrderOracle,
    "zero-order": ZeroOrderOracle,
    "hybrid": HybridOracle,
}


@dataclass(frozen=True)
class Profile:
    name: str = "standard"
    budget: int = 4000
    seeds: tuple[int, ...] = tuple(range(10001, 10051))
    bootstrap: int = 10000
    curve_points: int = 61

    @classmethod
    def build(cls, name: str) -> "Profile":
        if name == "smoke":
            return cls("smoke", 120, (101, 202, 303), 300, 31)
        if name == "standard":
            return cls()
        raise ValueError(f"Unknown profile: {name}")

    def validate(self):
        if self.budget < 40 or self.bootstrap < 100 or self.curve_points < 2:
            raise ValueError(
                "Require budget >= 40, bootstrap >= 100, curve_points >= 2"
            )
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("At least two distinct independent seeds are required")
        if any(not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("Seeds must be nonnegative integers")


@dataclass(frozen=True)
class Task:
    name: str
    stage: str
    landscape: str = "quadratic"
    noise: str = "none"
    drift: str = "stationary"
    dim: int = 10
    kappa: float = 1.0
    geometry: str = "euclidean"
    rank: int = 2

    @property
    def size(self):
        return (
            self.dim * self.rank
            if self.geometry in {"stiefel", "grassmann"}
            else self.dim
        )

    @property
    def primary(self):
        return "tail_error" if self.drift == "stationary" else "mean_error"


def default_tasks() -> tuple[Task, ...]:
    """Representative endpoints, not a Cartesian product of all settings."""
    tasks = [
        Task("clean", "reference"),
        Task("gaussian", "reference", noise="gaussian"),
    ]
    for noise in ("none", "gaussian"):
        tasks.extend(
            [
                Task(f"conditioned_{noise}", "landscape", noise=noise, kappa=100),
                Task(f"pnorm_{noise}", "landscape", landscape="pnorm", noise=noise),
                Task(
                    f"rosenbrock_{noise}",
                    "landscape",
                    landscape="rosenbrock",
                    noise=noise,
                ),
                Task(
                    f"multimodal_{noise}",
                    "landscape",
                    landscape="multiextremal",
                    noise=noise,
                ),
            ]
        )
    for noise in (
        "correlated",
        "heavy_tailed",
        "sparse",
        "multiplicative",
        "quantized",
    ):
        tasks.append(Task(noise, "observation", noise=noise))
    for drift in ("linear", "random_walk", "cyclic", "jump"):
        tasks.append(Task(drift, "dynamics", noise="gaussian", drift=drift))
    tasks.extend(
        [
            Task(
                "linear_correlated", "interaction", noise="correlated", drift="linear"
            ),
            Task("jump_outliers", "interaction", noise="heavy_tailed", drift="jump"),
            Task("linear_d30", "scaling", noise="gaussian", drift="linear", dim=30),
            Task("linear_d100", "scaling", noise="gaussian", drift="linear", dim=100),
        ]
    )
    for geometry in ("simplex", "stiefel", "grassmann"):
        for moving, noise in (
            (False, "gaussian"),
            (True, "gaussian"),
            (True, "correlated"),
        ):
            tasks.append(
                Task(
                    f"{geometry}_{'moving' if moving else 'stationary'}_{noise}",
                    "geometry",
                    landscape=geometry,
                    geometry=geometry,
                    dim=6,
                    noise=noise,
                    drift=(
                        ("simplex_cycle" if geometry == "simplex" else "stiefel")
                        if moving
                        else "stationary"
                    ),
                )
            )
    return tuple(tasks)


@dataclass(frozen=True)
class Method:
    """Custom factory receives task geometry and an algorithm-only seed, no truth."""

    name: str
    oracle_type: str
    geometries: tuple[str, ...]
    factory: Callable[[Task, int], Any] = field(repr=False, compare=False)
    parameters: dict = field(default_factory=dict)
    implementation: str = "custom"
    rng_key: str | None = None
    source: str = "user supplied"
    variant: str = "user supplied"
    implementation_version: int = 1

    @property
    def stream_id(self):
        if self.rng_key:
            return self.rng_key
        if self.implementation != "custom":
            return self.implementation
        module = getattr(self.factory, "__module__", type(self.factory).__module__)
        name = getattr(self.factory, "__qualname__", type(self.factory).__qualname__)
        return f"{module}:{name}"


def catalog_method(name: str, parameters: dict | None = None) -> Method:
    spec = resolve_optimizer_spec(name)
    algorithm = create_optimizer(name, parameters)
    params = dict(algorithm.parameters)
    return Method(
        spec.canonical_name,
        spec.oracle_type,
        spec.geometries,
        CatalogFactory(name, params),
        params,
        spec.implementation_id,
        rng_key=spec.rng_stream_id,
        source=spec.source,
        variant=spec.variant,
        implementation_version=spec.implementation_version,
    )


def stream_seed(seed: int, channel: str) -> int:
    split = "evaluation" if 10001 <= seed <= 10050 else "development"
    return identity_seed(seed, "canonical", SUITE_VERSION, split, channel)


class SimplexCycle(Drift):
    """A positive, unit-sum cyclic target, without clipping latent coordinates."""

    def __init__(self, center, period):
        self.center = np.asarray(center).copy()
        self.period = period

    def reset(self):
        pass

    def step(self, theta, t, action=None):
        phase = 2 * np.pi * (t + 1) / self.period
        weights = self.center * np.exp(
            np.sin(phase + np.arange(len(theta))) - np.sin(np.arange(len(theta)))
        )
        return weights / weights.sum()


def task_components(task: Task, profile: Profile, seed: int):
    rng = np.random.default_rng(stream_seed(seed, "initial"))
    direction = rng.normal(size=task.size)
    direction /= np.linalg.norm(direction)
    theta = direction.copy()
    x0 = np.zeros(task.size)
    if task.geometry == "simplex":
        theta, x0 = rng.dirichlet(np.ones(task.size), size=2)
    elif task.geometry in {"stiefel", "grassmann"}:
        theta = random_stiefel(task.dim, task.rank, rng).ravel()
        x0 = random_stiefel(task.dim, task.rank, rng).ravel()
    landscape_params = {}
    if task.landscape == "quadratic":
        landscape_params = {
            "dim": task.dim,
            "condition_number": task.kappa,
            "seed": stream_seed(seed, "landscape"),
        }
    elif task.landscape == "pnorm":
        landscape_params = {"p": 1.5}
    elif task.landscape == "multiextremal":
        landscape_params = {"width": 1.0}
    elif task.geometry in {"stiefel", "grassmann"}:
        landscape_params = {"d": task.dim, "r": task.rank}
    landscape = make_landscape(task.landscape, **landscape_params)
    drift_params = {}
    drift_seed = stream_seed(seed, "drift")
    if task.drift == "linear":
        v = np.random.default_rng(drift_seed).normal(size=task.size)
        drift_params = {"velocity": 0.005 * v / np.linalg.norm(v)}
    elif task.drift == "random_walk":
        drift_params = {"sigma": 0.005 / math.sqrt(task.size), "seed": drift_seed}
    elif task.drift == "cyclic":
        drift_params = {
            "amplitude": 1 / math.sqrt(task.size),
            "period": profile.budget // 4,
            "center": theta.copy(),
        }
    elif task.drift == "jump":
        drift_params = {
            "interval": profile.budget // 5,
            "jump_magnitude": 1.0,
            "dim": task.size,
            "seed": drift_seed,
        }
    elif task.drift == "stiefel":
        drift_params = {
            "d": task.dim,
            "r": task.rank,
            "sigma": 0.005,
            "seed": drift_seed,
        }
    if task.drift == "simplex_cycle":
        drift = SimplexCycle(theta, profile.budget // 4)
        drift_params = {"period": profile.budget // 4}
    else:
        drift = make_drift(task.drift, **drift_params)
    env = DynamicEnvironment(task.size, drift, landscape, initial_theta=theta)
    resolved = {
        "landscape": landscape_params,
        "drift": drift_params,
        "initial_theta": theta,
        "x0": x0,
    }
    return env, x0, resolved


def noise_parameters(task: Task, seed: int, *, gradient: bool):
    # Additive gradient scales hold the total (not per-coordinate) scale fixed.
    scale = 0.1 / math.sqrt(task.size) if gradient else 0.1
    params = {
        "seed": stream_seed(seed, "gradient_noise" if gradient else "value_noise")
    }
    if task.noise in {"gaussian", "correlated"}:
        params["sigma"] = scale
        if task.noise == "correlated":
            params.update(phi=0.9, dim=task.size if gradient else 1)
    elif task.noise == "heavy_tailed":
        params.update(alpha=1.5, scale=scale)  # Not variance-matched to Gaussian.
    elif task.noise == "sparse":
        params.update(p=0.05, sigma=scale / math.sqrt(0.05))
    elif task.noise == "multiplicative":
        params.update(sigma_rel=0.3)
    elif task.noise == "quantized":
        params = {"delta": scale}
    else:
        params = {}
    return params


class BudgetExhausted(RuntimeError):
    pass


class NumericalFailure(RuntimeError):
    pass


class AlgorithmFormatError(ValueError):
    pass


def required_algorithm_format(task: Task, oracle_type: str) -> str:
    geometry = {
        "euclidean": "finite unconstrained vector",
        "simplex": "nonnegative coordinates, sum=1 (simplex)",
        "stiefel": f"flattened ({task.dim}, {task.rank}) orthonormal frame, X.T@X=I; frame tracking",
        "grassmann": f"flattened ({task.dim}, {task.rank}) orthonormal frame, X.T@X=I; subspace tracking",
    }[task.geometry]
    feedback = {
        "first-order": "obs.grad only; obs.value=None",
        "zero-order": "obs.value only; obs.grad=None",
        "hybrid": "obs.value and obs.grad",
    }.get(oracle_type, "declare first-order, zero-order or hybrid")
    return (
        f"initialize(x0, rng, task_spec); advance(client) -> AdvanceResult with "
        f"recommendation shape ({task.size},), update_complete and queries; {feedback}; {geometry}; "
        f"declare geometries including '{task.geometry}'; no latent-state access; "
        "each client.query consumes budget and advances time"
    )


def feasibility(task: Task, x: np.ndarray) -> float:
    if task.geometry == "simplex":
        return max(abs(float(x.sum()) - 1), max(0.0, -float(x.min())))
    if task.geometry in {"stiefel", "grassmann"}:
        X = x.reshape(task.dim, task.rank)
        return float(np.linalg.norm(X.T @ X - np.eye(task.rank)))
    return 0.0


def evaluate_records(task: Task, profile: Profile, records):
    errors, regrets, violations = [], [], []
    for record in records:
        x, theta = record.observation.x, record.theta
        violation = feasibility(task, x)
        if task.geometry == "grassmann":
            # Principal angles only define the intended metric on feasible frames.
            error = (
                principal_angle_distance(
                    x.reshape(task.dim, task.rank), theta.reshape(task.dim, task.rank)
                )
                ** 2
                if violation <= 1e-6
                else np.nan
            )
        else:
            error = float(np.sum((x - theta) ** 2))
        errors.append(error)
        regrets.append(record.clean_value - record.optimum_value)
        violations.append(violation)
    errors, regrets, violations = map(np.asarray, (errors, regrets, violations))
    if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(regrets)):
        raise NumericalFailure("Non-finite or geometrically undefined evaluation")
    metrics = {
        "mean_error": float(errors.mean()),
        "tail_error": float(errors[int(0.8 * len(errors)) :].mean()),
        "dynamic_regret": float(regrets.sum()),
        "mean_excess_loss": float(regrets.mean()),
        "max_constraint_violation": float(violations.max()),
    }
    hits = np.flatnonzero(errors <= 0.1**2)
    reached = bool(len(hits))
    metrics.update(
        queries_to_threshold=float(hits[0] + 1) if reached else None,
        threshold_reached=reached,
        threshold_status="reached" if reached else "right_censored",
        restricted_queries_to_threshold=int(hits[0] + 1) if reached else profile.budget,
        threshold_budget=profile.budget,
        threshold_n_valid=len(errors) + 1,
    )
    events = []
    if task.drift == "jump":
        interval = profile.budget // 5
        window = profile.budget // 10
        hold = max(2, profile.budget // 200)
        for start in range(interval, len(errors), interval):
            end = min(start + window, len(errors), start + interval)
            if end - start < window:
                events.append(
                    {
                        "jump_query": start,
                        "complete_window": False,
                        "recovered": None,
                        "restricted_time": None,
                    }
                )
                continue
            good = errors[start:end] <= 0.2**2
            tau = None
            for offset in range(hold - 1, len(good)):
                if good[offset - hold + 1 : offset + 1].all():
                    tau = offset + 1  # queries after change, including confirmation
                    break
            events.append(
                {
                    "jump_query": start,
                    "complete_window": True,
                    "recovered": tau is not None,
                    "restricted_time": window if tau is None else tau,
                }
            )
        for event in events:
            event.update(
                confirmation_queries=hold,
                declared_window=window,
                time_convention="queries_after_change_including_confirmation",
                censored=(not event["recovered"]) if event["complete_window"] else None,
            )
        full = [e for e in events if e["complete_window"]]
        metrics["recovery_n_events"] = len(events)
        metrics["recovery_n_complete_windows"] = len(full)
        metrics["recovery_n_censored"] = sum(not e["recovered"] for e in full)
        if full:
            metrics["recovery_fraction"] = float(
                np.mean([e["recovered"] for e in full])
            )
            metrics["restricted_recovery_time"] = float(
                np.mean([e["restricted_time"] for e in full])
            )
    return metrics, errors, events


@_run
def run_one(
    method: Method,
    task: Task,
    profile: Profile,
    seed: int,
    trace_sink=None,
    recommendation_sink=None,
):
    base = {
        "task": task.name,
        "stage": task.stage,
        "geometry": task.geometry,
        "algorithm": method.name,
        "implementation_id": method.implementation,
        "implementation_version": method.implementation_version,
        "rng_key": method.stream_id,
        "oracle": method.oracle_type,
        "seed": seed,
        "primary_metric": task.primary,
        "budget": profile.budget,
        "queries": 0,
        "value_queries": 0,
        "gradient_queries": 0,
        "required_format": required_algorithm_format(task, method.oracle_type),
    }
    if task.geometry not in method.geometries:
        return (
            {
                **base,
                "status": "SKIPPED",
                "reason": f"incompatible_geometry: method supports {method.geometries}, task requires {task.geometry}",
            },
            [],
            [],
            {},
        )
    env, x, resolved = task_components(task, profile, seed)
    value_params = noise_parameters(task, seed, gradient=False)
    grad_params = noise_parameters(task, seed, gradient=True)
    oracle = ORACLES[method.oracle_type](
        env,
        value_noise=(
            make_noise(task.noise, **value_params)
            if method.oracle_type != "first-order"
            else None
        ),
        grad_noise=(
            make_noise(task.noise, **grad_params)
            if method.oracle_type != "zero-order"
            else None
        ),
    )
    resolved.update(
        value_noise=value_params if method.oracle_type != "first-order" else None,
        gradient_noise=grad_params if method.oracle_type != "zero-order" else None,
    )
    resolved["instance_sha256"] = hashlib.sha256(
        json.dumps(
            jsonable(
                {
                    "task": asdict(task),
                    "initial_theta": resolved["initial_theta"],
                    "x0": resolved["x0"],
                    "landscape": resolved["landscape"],
                    "drift": resolved["drift"],
                }
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    base["instance_sha256"] = resolved["instance_sha256"]
    records = []
    recommendations, recommendation_queries = [x.copy()], [0]
    completed_updates = 0
    started = time.perf_counter()
    resources = ResourceSampler()
    _bind_environment(env, oracle)
    np_state, py_state = np.random.get_state(), random.getstate()
    algorithm_seed = identity_seed(
        seed, "canonical", SUITE_VERSION, "algorithm", method.stream_id
    )
    resolved["algorithm_seed"] = algorithm_seed

    def checked(point):
        point = _user_call(np.asarray, point, dtype=float)
        if point.shape != (task.size,):
            raise AlgorithmFormatError(
                f"Expected decision shape {(task.size,)}, got {point.shape}"
            )
        if not np.isfinite(point).all() or np.linalg.norm(point) > 1e12:
            raise NumericalFailure("Non-finite decision or decision norm exceeds 1e12")
        return point.copy()

    @_internal
    def query(point):
        check_cancelled()
        if oracle.n_queries >= profile.budget:
            raise BudgetExhausted()
        point = checked(point)
        oracle.start_step(oracle.n_queries)
        try:
            observation = oracle.query(point)
            record = oracle._take_evaluation()
            if not np.isfinite(record.clean_value) or not np.isfinite(
                record.optimum_value
            ):
                raise NumericalFailure("Non-finite evaluator loss")
            records.append(record)
        finally:
            oracle._last_evaluation = None
            oracle.end_step()
        env.step(action=point)
        return observation

    status, reason = "SUCCESS", ""
    resources.start()
    try:
        np.random.seed(algorithm_seed)
        random.seed(algorithm_seed)
        optimizer = _user_call(method.factory, task, algorithm_seed)
        try:
            _user_call(require_algorithm, optimizer)
        except TypeError as exc:
            raise AlgorithmFormatError(_user_call(str, exc)) from exc
        shape = (
            (task.dim, task.rank) if task.geometry in {"stiefel", "grassmann"} else None
        )
        try:
            _user_call(
                lambda *args: optimizer.initialize(*args),
                x.copy(),
                np.random.default_rng(algorithm_seed),
                TaskSpec(task.size, task.geometry, shape),
            )
        except (TypeError, ValueError) as exc:
            raise AlgorithmFormatError(_user_call(str, exc)) from exc
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            while oracle.n_queries < profile.budget:
                check_cancelled()
                client = OracleClient(query, lambda: profile.budget - oracle.n_queries)
                before = oracle.n_queries
                try:
                    update = _user_call(lambda: optimizer.advance(client))
                    if type(update) is not AdvanceResult:
                        raise AlgorithmFormatError("advance must return AdvanceResult")
                    if (
                        update.queries != oracle.n_queries - before
                        or update.queries == 0
                    ):
                        raise AlgorithmFormatError(
                            "Incorrect or zero query count in update"
                        )
                    x = checked(update.recommendation)
                    completed_updates += int(update.update_complete)
                    recommendations.append(x.copy())
                    recommendation_queries.append(oracle.n_queries)
                finally:
                    client._close()
    except AlgorithmFormatError as exc:
        status, reason = "ERROR_INTERFACE", _user_call(str, exc)
    except (
        NumericalFailure,
        FloatingPointError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as exc:
        status, reason = "NUMERICAL_FAILURE", _user_call(str, exc)
    except Exception as exc:
        status, reason = "ERROR", _user_call(lambda: f"{type(exc).__name__}: {exc}")
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        base.update(resources.finish())
    base.update(
        runtime_seconds=time.perf_counter() - started,
        completed_updates=completed_updates,
        recommendations=len(recommendations),
        queries=oracle.n_queries,
        value_queries=oracle.n_value_queries,
        gradient_queries=oracle.n_grad_queries,
    )
    curves, events = [], []
    if recommendation_sink is not None:
        # Only algorithm-owned decisions; no truth or evaluation statistics.
        _user_call(
            recommendation_sink,
            {k: base[k] for k in ("task", "algorithm", "oracle", "seed")},
            np.array(recommendation_queries),
            np.array(recommendations),
        )
    if status == "SUCCESS":
        try:
            if len(records) != profile.budget:
                raise RuntimeError(
                    "A successful run must exhaust the exact query budget"
                )
            if any(feasibility(task, r.observation.x) > 1e-6 for r in records):
                status, reason = "INFEASIBLE", "Decision violates declared geometry"
            else:
                metrics, errors, events = evaluate_records(task, profile, records)
                if task.geometry == "grassmann":
                    initial_error = principal_angle_distance(
                        recommendations[0].reshape(task.dim, task.rank),
                        np.asarray(resolved["initial_theta"]).reshape(
                            task.dim, task.rank
                        ),
                    )
                else:
                    initial_error = np.linalg.norm(
                        recommendations[0] - resolved["initial_theta"]
                    )
                if initial_error <= 0.1:
                    metrics.update(
                        queries_to_threshold=0,
                        threshold_reached=True,
                        threshold_status="reached",
                        restricted_queries_to_threshold=0,
                    )
                base.update(metrics)
                base["score"] = metrics[task.primary]
                if trace_sink is not None:
                    trace_metadata = {
                        k: base[k] for k in ("task", "algorithm", "oracle", "seed")
                    }
                    code = (
                        trace_sink.__code__
                        if type(trace_sink) is FunctionType
                        else None
                    )
                    if code is not None and _trusted_file(code.co_filename):
                        trace_sink(trace_metadata, errors.copy())
                    else:
                        _after_release(trace_sink, trace_metadata, errors.copy())
                indices = np.unique(
                    np.linspace(0, len(errors) - 1, profile.curve_points, dtype=int)
                )
                curves = [
                    {
                        **{k: base[k] for k in ("task", "algorithm", "oracle", "seed")},
                        "query": int(i + 1),
                        "squared_error": float(errors[i]),
                    }
                    for i in indices
                ]
        except NumericalFailure as exc:
            status, reason = "NUMERICAL_FAILURE", _user_call(str, exc)
        except Exception as exc:
            status, reason = "ERROR", _user_call(lambda: f"{type(exc).__name__}: {exc}")
    if status not in {"SUCCESS", "SKIPPED"}:
        base.update(
            threshold_status=(
                "incomplete_numerical_run"
                if status == "NUMERICAL_FAILURE"
                else "unavailable"
            ),
            threshold_n_valid=len(records) + 1,
        )
    return {**base, "status": status, "reason": reason}, curves, events, resolved


def mean_interval(values, repetitions: int, seed: int):
    """Percentile bootstrap of independent run scalars (never time samples)."""
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return None, None, None
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repetitions, 256):
        indices = rng.integers(
            len(values), size=(min(256, repetitions - start), len(values))
        )
        means.extend(values[indices].mean(axis=1))
    low, high = np.quantile(means, [0.025, 0.975])
    return mean, float(low), float(high)


def wilson_interval(failures: int, total: int):
    if not total:
        return None, None
    z = 1.959963984540054
    p = failures / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(runs: pd.DataFrame, profile: Profile):
    summaries, comparisons = [], []
    for (task, algorithm, oracle), group in runs.groupby(
        ["task", "algorithm", "oracle"], sort=False
    ):
        attempted = group[group.status != "SKIPPED"]
        good = group[group.status == "SUCCESS"]
        failed = len(attempted) - len(good)
        lo_fail, hi_fail = wilson_interval(failed, len(attempted))
        mean, low, high = mean_interval(
            good.get("score", []),
            profile.bootstrap,
            identity_seed(0, "scalar_bootstrap"),
        )
        reached = int(
            good.get("threshold_reached", pd.Series(dtype=bool)).fillna(False).sum()
        )
        attainment_low, attainment_high = wilson_interval(reached, len(good))
        summaries.append(
            {
                "task": task,
                "algorithm": algorithm,
                "oracle": oracle,
                "primary_metric": group.primary_metric.iloc[0],
                "n_success": len(good),
                "n_threshold_reached": reached,
                "threshold_rate_success_only": (
                    reached / len(good) if len(good) else None
                ),
                "threshold_ci_low": attainment_low,
                "threshold_ci_high": attainment_high,
                "n_failed": failed,
                "n_skipped": len(group) - len(attempted),
                "mean_success_only": mean,
                "ci_low": low,
                "ci_high": high,
                "failure_rate": failed / len(attempted) if len(attempted) else None,
                "failure_ci_low": lo_fail,
                "failure_ci_high": hi_fail,
            }
        )
    # References are explicit and independent of requested method ordering.
    for (task, oracle), group in runs[runs.status != "SKIPPED"].groupby(
        ["task", "oracle"], sort=False
    ):
        names = list(group.algorithm.unique())
        geometry = group.geometry.iloc[0]
        reference = (
            "ProjectedGradient"
            if geometry == "simplex"
            else (
                "RiemannianSGD"
                if geometry in {"stiefel", "grassmann"}
                else "SPSA" if oracle == "zero-order" else "SGD"
            )
        )
        if reference not in names:
            continue
        baseline = group[group.algorithm == reference].set_index("seed")
        for name in (n for n in names if n != reference):
            other = group[group.algorithm == name].set_index("seed")
            common = baseline.index.intersection(other.index)
            paired = [
                s
                for s in common
                if baseline.loc[s, "status"] == "SUCCESS"
                and other.loc[s, "status"] == "SUCCESS"
            ]
            diffs = [other.loc[s, "score"] - baseline.loc[s, "score"] for s in paired]
            mean, low, high = mean_interval(
                diffs, profile.bootstrap, identity_seed(1, "paired_bootstrap")
            )
            comparisons.append(
                {
                    "task": task,
                    "oracle": oracle,
                    "algorithm": name,
                    "baseline": reference,
                    "n_pairs": len(paired),
                    "n_excluded_pairs": len(common) - len(paired),
                    "mean_difference_success_pairs_only": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(
        comparisons,
        columns=[
            "task",
            "oracle",
            "algorithm",
            "baseline",
            "n_pairs",
            "n_excluded_pairs",
            "mean_difference_success_pairs_only",
            "ci_low",
            "ci_high",
        ],
    )


def jsonable(value):
    return json_safe(value)


@_series
@preserve_global_rng
def run_suite(
    output: Path,
    profile: Profile | None = None,
    methods: Sequence[Method] | None = None,
    tasks: Sequence[Task] | None = None,
    *,
    plots: bool = True,
    progress: bool = True,
    workers: int = 1,
):
    profile = profile or Profile()
    profile.validate()
    methods = (
        tuple(methods)
        if methods is not None
        else tuple(map(catalog_method, DEFAULT_METHODS))
    )
    tasks = tuple(tasks) if tasks is not None else default_tasks()
    if not methods or len({m.name for m in methods}) != len(methods):
        raise ValueError("Choose nonempty, uniquely named methods")
    if any(m.oracle_type not in ORACLES for m in methods):
        raise ValueError("Supported oracle tracks: first-order, zero-order, hybrid")
    if not tasks or len({t.name for t in tasks}) != len(tasks):
        raise ValueError("Choose nonempty, uniquely named tasks")
    compatible = sum(t.geometry in m.geometries for t in tasks for m in methods) * len(
        profile.seeds
    )
    options = execution_options(workers, compatible)
    descriptions = (
        {id(m): _user_call(describe_method, m) for m in methods} if workers > 1 else {}
    )
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Output directory must be empty; existing results are never overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    execution = Execution(output, options, compatible)
    try:
        with threadpool_limits(limits=1):
            result = _run_suite(
                output,
                profile,
                methods,
                tasks,
                plots,
                progress,
                execution,
                descriptions,
            )
        execution.computed(result["manifest"])
        _after_release(execution.export)
        return result
    except BaseException as exc:
        if execution.state not in {"complete", "export_failed"}:
            execution.failed(isinstance(exc, KeyboardInterrupt))
        raise


def _run_suite(
    output, profile, methods, tasks, plots, progress, execution, descriptions
):
    rows, curves, events, configurations = [], [], [], []
    trace_arrays, trace_index, log_lines = {}, [], []
    recommendation_directory = output / "recommendations"
    recommendation_directory.mkdir()

    def save_recommendations(metadata, queries, positions):
        # These are public algorithm decisions, not evaluator outputs.
        key = f"{metadata['task']}__{metadata['algorithm']}__{metadata['seed']}"
        key = hashlib.sha256(key.encode()).hexdigest()
        np.savez_compressed(
            recommendation_directory / (key + ".npz"),
            query=queries,
            x=positions,
            metadata=json.dumps(metadata),
        )

    def collect_trace(metadata, values):
        key = f"run_{len(trace_index):05d}"
        trace_index.append({"key": key, **metadata})
        trace_arrays[key] = values

    def log(message):
        log_lines.append(message)
        if progress:
            print(message, flush=True)

    # Evaluation payloads remain private until the entire outer series releases.
    jobs, completed = [], {}
    design = []
    for task in tasks:
        for method in methods:
            if task.geometry not in method.geometries:
                log(
                    f"SKIPPED {task.name} / {method.name}: declared {method.geometries}; REQUIRED: {required_algorithm_format(task, method.oracle_type)}"
                )
            for seed in profile.seeds:
                index = len(design)
                design.append((method, task, seed))
                if task.geometry in method.geometries:
                    jobs.append((index, method, task, seed))
                else:
                    completed[index] = (run_one(method, task, profile, seed), [])
    with closing(execution.results(jobs, descriptions, profile)) as results:
        for index, (result, traces, recommendations) in results:
            for recommendation in recommendations:
                save_recommendations(*recommendation)
            completed[index] = (result, traces)
            execution.collected()
            if execution.completed % 10 == 0:
                log(
                    f"Progress: {execution.completed}/{execution.total} compatible runs completed"
                )
    execution.state = "aggregating"
    execution.status(force=True)
    for index, (method, task, seed) in enumerate(design):
        (row, curve, recovery, resolved), traces = completed.pop(index)
        for trace in traces:
            collect_trace(*trace)
        rows.append(row)
        if row["status"] not in {"SUCCESS", "SKIPPED"}:
            # Diagnostics may depend on evaluator data; export after release only.
            message = f"{row['status']} {task.name} / {method.name} / seed={seed}: {row['reason']}; REQUIRED: {row['required_format']}"
            log_lines.append(message)
            if progress:
                execution.defer(print, message, flush=True)
        curves.extend(curve)
        events.extend(
            {"task": task.name, "algorithm": method.name, "seed": seed, **e}
            for e in recovery
        )
        if resolved:
            configurations.append(
                {"task": task.name, "algorithm": method.name, "seed": seed, **resolved}
            )
    runs = pd.DataFrame(rows)
    summary, paired = summarize(runs, profile)
    metric_rows = []
    for (task_name, algorithm, oracle), group in runs.groupby(
        ["task", "algorithm", "oracle"], sort=False
    ):
        good = group[group.status == "SUCCESS"]
        for metric in (
            "mean_error",
            "tail_error",
            "dynamic_regret",
            "mean_excess_loss",
            "max_constraint_violation",
            "recovery_fraction",
            "restricted_recovery_time",
            "restricted_queries_to_threshold",
            "runtime_seconds",
            "sampled_peak_rss_bytes",
            "sampled_rss_increase_bytes",
        ):
            values = good[metric].dropna() if metric in good else []
            if len(values):
                mean, low, high = mean_interval(
                    values,
                    profile.bootstrap,
                    identity_seed(2, "metric_bootstrap", metric),
                )
                metric_rows.append(
                    {
                        "task": task_name,
                        "algorithm": algorithm,
                        "oracle": oracle,
                        "metric": metric,
                        "n_success": len(values),
                        "n_total_success": len(good),
                        "n_valid": len(values),
                        "n_undefined": len(good) - len(values),
                        "metric_status": "conditional_on_success",
                        "mean_success_only": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            else:
                metric_rows.append(
                    {
                        "task": task_name,
                        "algorithm": algorithm,
                        "oracle": oracle,
                        "metric": metric,
                        "n_success": 0,
                        "n_total_success": len(good),
                        "n_valid": 0,
                        "n_undefined": len(good),
                        "metric_status": "not_estimable_or_not_applicable",
                        "mean_success_only": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }
                )
    curve_frame = pd.DataFrame(
        curves,
        columns=["task", "algorithm", "oracle", "seed", "query", "squared_error"],
    )
    event_frame = pd.DataFrame(
        events,
        columns=[
            "task",
            "algorithm",
            "seed",
            "jump_query",
            "complete_window",
            "recovered",
            "restricted_time",
            "confirmation_queries",
            "declared_window",
            "time_convention",
            "censored",
        ],
    )
    dependencies = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "wind-benchmark",
        "threadpoolctl",
    ):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    sources = source_hashes()
    from wind_benchmark.experiments.canonical import specification, specification_hash

    manifest = {
        "execution": execution.options,
        "suite": SUITE_VERSION,
        "catalog_version": CATALOG_VERSION,
        "reference_implementations": {
            name: resolve_optimizer_spec(name).implementation_id
            for name in ("SGD", "SPSA", "ProjectedGradient", "RiemannianSGD")
        },
        "run_kind": (
            "canonical"
            if (
                tasks == default_tasks()
                and profile.budget == 4000
                and profile.bootstrap == 10000
                and profile.seeds
                in (tuple(range(10001, 10051)), tuple(range(20001, 20021)))
            )
            else "custom_study"
        ),
        "memory_measurement": "10 ms sampled process RSS peak estimate; includes interpreter and evaluator; not isolated algorithm memory",
        "decision_stream": "queried-point",
        "specification": specification(),
        "specification_sha256": specification_hash(),
        "profile": asdict(profile),
        "tasks": [asdict(t) for t in tasks],
        "methods": [
            {
                "name": m.name,
                "oracle": m.oracle_type,
                "geometries": m.geometries,
                "parameters": m.parameters,
                "implementation": m.implementation,
                "rng_key": m.stream_id,
                "source": m.source,
                "variant": m.variant,
                "implementation_version": m.implementation_version,
            }
            for m in methods
        ],
        "resolved_runs": configurations,
        "python": platform.python_version(),
        "dependencies": dependencies,
        "source_sha256": sources,
        "protocol": "streaming; every query advances target; every queried position is scored",
        "rng_protocol": "named_streams_v2; display labels are not seed inputs",
        "information_barrier": "Observation plus scoped query-only client; evaluator output after complete suite; not a Python sandbox",
        "tuning": "none; fixed supplied or catalog parameters; not tuned algorithm rankings",
        "statistics": "run-level percentile bootstrap, 95% individual intervals; paired differences within task/oracle; Wilson failure intervals; no significance claims",
        "noise_notes": "AR(1) starts at zero; Gaussian innovations; heavy tails clipped at 1000*scale and not variance-matched; sparse noise RMS-matched to Gaussian",
        "geometry_notes": "squared Euclidean/frame distance or squared Grassmann principal-angle distance; feasibility tolerance 1e-6; no automatic projection",
        "failure_rule": "nonfinite arithmetic or decision norm > 1e12; failed runs retained, means conditional on success",
        "recovery_rule": {
            "radius": 0.2,
            "window": profile.budget // 10,
            "consecutive_queries": max(2, profile.budget // 200),
            "unrecovered": "restricted time equals window",
            "time_convention": "queries after change including the confirming observation",
        },
        "smoke_warning": profile.name == "smoke",
        "report_version": "3",
        "export_schema_version": 4,
        "metrics_protocol": "metrics_v2",
        "privacy_protocol": "local_series_v2",
        "task_generator_version": "wind_tracking_v1.haar_qr_signs.sphere_zero_start",
        "references": {
            "euclidean_first_order": "SGD",
            "euclidean_zero_order": "SPSA",
            "simplex": "ProjectedGradient",
            "manifold": "RiemannianSGD",
        },
        "curve_storage": "full successful-run squared errors in trajectories.npz with trace_index.csv; curves.csv is a sampled preview",
        "curve_intervals": "pointwise 95% percentile bootstrap of complete independent runs; up to 2000 resamples; conditional on success, not simultaneous bands",
    }
    execution.defer(runs.to_csv, output / "runs.csv", index=False)
    execution.defer(summary.to_csv, output / "summary.csv", index=False)
    execution.defer(paired.to_csv, output / "paired_comparisons.csv", index=False)
    metric_frame = pd.DataFrame(
        metric_rows,
        columns=[
            "task",
            "algorithm",
            "oracle",
            "metric",
            "n_success",
            "n_total_success",
            "n_valid",
            "n_undefined",
            "metric_status",
            "mean_success_only",
            "ci_low",
            "ci_high",
        ],
    )
    execution.defer(metric_frame.to_csv, output / "metrics_summary.csv", index=False)
    execution.defer(curve_frame.to_csv, output / "curves.csv", index=False)
    execution.defer(np.savez_compressed, output / "trajectories.npz", **trace_arrays)
    index_frame = pd.DataFrame(
        trace_index, columns=["key", "task", "algorithm", "oracle", "seed"]
    )
    execution.defer(index_frame.to_csv, output / "trace_index.csv", index=False)
    execution.defer(
        (output / "execution.log").write_text,
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )
    execution.defer(event_frame.to_csv, output / "recovery_events.csv", index=False)
    execution.defer(
        (output / "manifest.json").write_text,
        json.dumps(jsonable(manifest), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    report = [
        f"# {SUITE_VERSION}",
        "",
        f"Profile: {profile.name}; {len(profile.seeds)} independent runs per configuration; {profile.budget} calls per run.",
        "",
        "This is fixed-parameter evaluation, not hyperparameter tuning. Smoke runs are execution checks, not statistical evidence.",
        "",
        "Means and curves condition on successful runs. Read failure counts before comparing them. Paired intervals exclude unsuccessful pairs explicitly; intervals are individual, not simultaneous. No global ranking or significance declaration is produced.",
        "",
        "| Task | Method | Oracle | Metric | Success | Failed | Skipped | Mean (success only) | 95% CI |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples():
        mean = (
            f"{row.mean_success_only:.5g}" if pd.notna(row.mean_success_only) else "—"
        )
        ci = (
            f"[{row.ci_low:.5g}, {row.ci_high:.5g}]"
            if pd.notna(row.ci_low)
            else "not estimable"
        )
        report.append(
            f"| {row.task} | {row.algorithm} | {row.oracle} | {row.primary_metric} | {row.n_success} | {row.n_failed} | {row.n_skipped} | {mean} | {ci} |"
        )
    execution.defer(
        (output / "report.md").write_text, "\n".join(report) + "\n", encoding="utf-8"
    )
    if plots:
        from wind_benchmark.reporting.standard import render_report

        execution.defer(render_report, output)
    return {"runs": runs, "summary": summary, "paired": paired, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument(
        "--split", choices=("evaluation", "development"), default="evaluation"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument(
        "--optimizer-config",
        type=Path,
        help="JSON mapping requested algorithm names to parameter dictionaries",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Optional task IDs; omitted runs all compatible default tasks",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent worker processes (default: 1; no resume)",
    )
    args = parser.parse_args(argv)
    parameters = (
        json.loads(args.optimizer_config.read_text(encoding="utf-8"))
        if args.optimizer_config
        else {}
    )
    if not isinstance(parameters, dict) or set(parameters) - set(args.algorithms):
        parser.error("Optimizer config must map selected algorithm names to parameters")
    methods = [catalog_method(name, parameters.get(name)) for name in args.algorithms]
    tasks = default_tasks()
    if args.tasks:
        unknown = set(args.tasks) - {t.name for t in tasks}
        if unknown:
            parser.error(f"Unknown tasks: {sorted(unknown)}")
        tasks = tuple(t for t in tasks if t.name in args.tasks)
    profile = Profile.build(args.profile)
    if args.profile == "standard" and args.split == "development":
        profile = Profile("development", seeds=tuple(range(20001, 20021)))
    compatible = sum(t.geometry in m.geometries for t in tasks for m in methods)
    try:
        options = execution_options(args.workers, compatible * len(profile.seeds))
        if args.workers > 1:
            for method in methods:
                describe_method(method)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        compatible = sum(t.geometry in m.geometries for t in tasks for m in methods)
        print(
            json.dumps(
                {
                    "suite": SUITE_VERSION,
                    "execution": options,
                    "profile": asdict(profile),
                    "tasks": [asdict(t) for t in tasks],
                    "algorithms": [m.name for m in methods],
                    "compatible_runs": compatible * len(profile.seeds),
                    "maximum_oracle_calls": compatible
                    * len(profile.seeds)
                    * profile.budget,
                },
                indent=2,
            )
        )
        return 0
    output = args.output_dir or Path("results/default_pipeline") / profile.name
    result = run_suite(
        output, profile, methods, tasks, plots=not args.no_plots, workers=args.workers
    )
    print(f"Results: {output.resolve()}")
    return 1 if result["runs"].status.isin(["ERROR", "ERROR_INTERFACE"]).any() else 0


if __name__ == "__main__":
    sys.exit(main())
