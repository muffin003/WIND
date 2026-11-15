"""Publication-oriented experiments for the WIND benchmark.

This module complements :mod:`wind_benchmark.experiments.configurable`.  The legacy experiment
is a broad integration benchmark; this file isolates five falsifiable mechanisms:

1. exact empirical/theoretical calibration of constant-step SGD on a drifting
   quadratic;
2. the memory--adaptation trade-off for SGD, Heavy Ball, Nesterov, Adam, and
   AMSGrad;
3. the zero-order variance--freshness trade-off under frozen-round and
   streaming-query semantics, with comparisons made at equal oracle-query budgets;
4. feasibility-aware tracking on the simplex, Stiefel manifold, and Grassmann
   manifold;
5. zero-order tracking quality as ambient dimension grows, separating stationary
   estimation error from dynamic tracking error at a fixed query budget.

The default ``smoke`` profile is intentionally small.  The ``paper`` profile uses
more seeds, longer horizons, and one-factor-at-a-time parameter sweeps.  Run with::

    python -m wind_benchmark.experiments.paper --profile smoke
    python -m wind_benchmark.experiments.paper --profile paper --studies all

Every run writes run-level summaries, aggregated curves, confidence intervals,
and a machine-readable manifest below ``results/final_experiment``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import hashlib
from wind_benchmark.core.randomness import stream_seed as identity_seed
import platform
import importlib.metadata
import math
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_benchmark.algorithms.catalog import (
    create_optimizer,
    resolve_optimizer_spec,
    CATALOG_VERSION,
)
from wind_benchmark.core.runner import OptimizerInfo, OptimizerProtocol
from wind_benchmark.core.execution import OracleSession
from wind_benchmark.core.metrics import RecoveryEvents
from wind_benchmark.reporting.common import (
    json_safe,
    safe_print as print,
    source_hashes,
)
from wind_benchmark.core.privacy import (
    _series,
    _user_call,
    _publish,
    _after_release,
    TruthGuard,
    _bind,
    _internal,
    _owner,
    _current,
)
from wind_benchmark.core.oracle import FirstOrderOracle, ZeroOrderOracle, Observation
from wind_benchmark.algorithms.protocol import TaskSpec
from wind_benchmark.experiments.controls import ProjectedAmbientControl
from wind_benchmark.core.environment import (
    DynamicEnvironment,
    GaussianNoise,
    LinearDrift,
    PrescribedDrift,
    GrassmannLandscape,
    QuadraticLandscape,
    SimplexLandscape,
    StiefelLandscape,
)
from wind_benchmark.core.geometry import (
    principal_angle_distance,
    project_to_stiefel,
    random_stiefel,
)

STUDY_NAMES = ("calibration", "memory", "zeroth", "geometry", "dimension")
PAPER_PROTOCOL = "paper_tracking_v4"
# Historical generator tokens are immutable; a reporting version is not a seed.
PAPER_GENERATOR_ID = "paper_tracking_v3"
TUNING_PROTOCOL = "paper_zo_equal_budget_v1"


@dataclass(frozen=True)
class ExperimentProfile:
    """Computational budget for a named experiment profile."""

    name: str
    seeds: Tuple[int, ...]
    calibration_steps: int
    memory_steps: int
    zero_order_budget: int
    zero_order_tuning_budget: int
    geometry_steps: int
    bootstrap_repetitions: int
    curve_points: int

    @classmethod
    def build(cls, name: str) -> "ExperimentProfile":
        if name == "smoke":
            return cls(
                name=name,
                seeds=(101, 202, 303),
                calibration_steps=500,
                memory_steps=180,
                zero_order_budget=800,
                zero_order_tuning_budget=300,
                geometry_steps=180,
                bootstrap_repetitions=300,
                curve_points=61,
            )
        if name == "paper":
            return cls(
                name=name,
                seeds=tuple(range(1001, 1031)),
                calibration_steps=1500,
                memory_steps=2000,
                zero_order_budget=10_000,
                zero_order_tuning_budget=2000,
                geometry_steps=1000,
                bootstrap_repetitions=2000,
                curve_points=201,
            )
        raise ValueError(f"Unknown profile: {name}")


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    sweep: str
    x_value: float
    dim: int = 20
    condition_number: float = 5.0
    learning_rate: float = 0.05
    gradient_noise: float = 0.02
    drift_per_step: float = 0.01


@dataclass(frozen=True)
class MemoryScenario:
    name: str
    kind: str
    gradient_noise: float
    drift_per_step: float = 0.0
    amplitude: float = 0.0
    period: int = 0
    jump_interval: int = 0


@dataclass(frozen=True)
class ZeroOrderCase:
    case_id: str
    dim: int
    condition_number: float
    path_kind: str
    observation_noise: float = 0.001
    drift_per_query: float = 0.0
    amplitude: float = 0.0
    period: int = 0
    threshold_epsilon: float = 0.1

    def __post_init__(self):
        if not np.isfinite(self.threshold_epsilon) or self.threshold_epsilon < 0:
            raise ValueError("threshold_epsilon must be finite and nonnegative")


@dataclass(frozen=True)
class ZeroOrderMethodConfig:
    name: str
    family: str
    directions: int = 1

    @property
    def implementation_id(self) -> str:
        return resolve_optimizer_spec(self.catalog_name).implementation_id

    @property
    def catalog_name(self):
        return {
            "first_order": "SGD",
            "spsa": "SPSA",
            "spherical": "SphericalZOSGD",
            "coordinate": "KieferWolfowitz",
            "one_point": "OnePointSPSA",
            "zo_sign": "ZOSignSGD",
            "accelerated": "AcceleratedSPSA",
            "random_search": "RandomSearch",
        }[self.family]

    @property
    def rng_stream_id(self):
        tokens = {
            "first_order": "paper_tracking_v3:first_order:m1",
            "coordinate": "paper_tracking_v3:coordinate:m1",
            "spsa": "paper:spsa:normalized_tracking:v2",
            "one_point": "paper:one_point:v1",
            "zo_sign": "paper:zo_sign:v1",
            "accelerated": "paper:accelerated:constant:v1",
            "random_search": "paper:random_search:fresh:v1",
        }
        if self.family == "spherical":
            return f"paper_tracking_v3:spherical:m{self.directions}"
        return tokens[self.family]

    def query_cost(self, dim: int) -> int:
        if self.family == "first_order":
            return 1
        if self.family in {"spsa", "accelerated", "random_search"}:
            return 2
        if self.family == "one_point":
            return 1
        if self.family == "zo_sign":
            return self.directions + 1
        if self.family == "spherical":
            return 2 * self.directions
        if self.family == "coordinate":
            return 2 * dim
        raise ValueError(f"Unknown zero-order family: {self.family}")


@dataclass(frozen=True)
class ZeroOrderHyperparameters:
    parameters: dict
    tuning_score: float = float("nan")


@dataclass
class StudyResult:
    runs: pd.DataFrame
    summary: pd.DataFrame
    curves: pd.DataFrame
    extras: Dict[str, pd.DataFrame]
    histories: dict = field(default_factory=dict)


def _pack_histories(curves, columns, seeds):
    arrays, index = {}, []
    for group, trajectories in curves.items():
        for seed, (axis, values) in zip(seeds, trajectories, strict=True):
            key = f"trace_{len(index):05d}"
            arrays[key] = np.asarray(values)
            index.append(
                {
                    "key": key,
                    **dict(zip(columns, group)),
                    "seed": seed,
                    "start": float(axis[0]),
                    "stop": float(axis[-1]),
                    "length": len(axis),
                    "grid": "all_integer_ticks",
                }
            )
    return {"arrays": arrays, "index": index}


def _path_environment(landscape, path):
    path = np.asarray(path, dtype=float).reshape(len(path), -1)
    return DynamicEnvironment(
        dim=path.shape[1],
        drift=PrescribedDrift(path),
        landscape=landscape,
        initial_theta=path[0],
        bounds=None,
    )


@_series
def _first_order_trace(environment, optimizer, x0, steps, noise):
    """Run through the public gradient oracle; release truth only after closure."""
    session = OracleSession(
        FirstOrderOracle(environment, grad_noise=noise, blind_value=True),
        steps,
    )
    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    landscape = environment.landscape
    if isinstance(landscape, GrassmannLandscape):
        geometry, shape = "grassmann", (landscape.d, landscape.r)
    elif isinstance(landscape, StiefelLandscape):
        geometry, shape = "stiefel", (landscape.d, landscape.r)
    elif isinstance(landscape, SimplexLandscape):
        geometry, shape = "simplex", None
    else:
        geometry, shape = "euclidean", None
    # Explicit ambient raw control, not canonical geometry compatibility.
    if "euclidean" in _user_call(getattr, optimizer, "geometries", ("euclidean",)):
        geometry, shape = "euclidean", None
    _user_call(
        lambda *args: optimizer.initialize(*args),
        x,
        np.random.default_rng(0),
        TaskSpec(x.size, geometry, shape),
    )
    try:
        for _ in range(steps):
            with session.client() as client:
                update = optimizer.advance(client)
            if not update.update_complete or update.queries != 1:
                raise RuntimeError("FO study requires one completed gradient update")
    except BaseException:
        session.abort()
        raise
    records = session.finish()
    return records, session.counts


def _geometry_method(name, lr, d=None, r=None):
    if name == "PolarProjectedSGDControl":
        return ProjectedAmbientControl(lr)
    catalog_name = {
        "RawAmbientSGDControl": "SGD",
        "ProjectedGradient": "ProjectedGradient",
    }.get(name, name)
    return create_optimizer(catalog_name, {"lr": lr})


def _stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(
        str(part)
        for part in (
            PAPER_GENERATOR_ID,
            "evaluation" if base in range(1001, 1031) else "development",
            *parts,
        )
    ).encode("utf-8")
    return int((base + zlib.crc32(payload)) % (2**32 - 1))


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _tail_mean(values: np.ndarray, fraction: float = 0.2) -> float:
    start = max(0, int((1.0 - fraction) * len(values)))
    return float(np.mean(values[start:]))


def _bootstrap_mean_interval(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> Tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = np.sort(clean[np.isfinite(clean)])
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        return float("nan"), float("nan")
    indices = rng.integers(0, clean.size, size=(repetitions, clean.size))
    bootstrap_means = clean[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _summarize_runs(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metric_columns = list(metric_columns)
    if "queries_to_threshold" in metric_columns:
        metric_columns.extend(["restricted_queries_to_threshold", "threshold_reached"])
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in frame.groupby(grouper, dropna=False, sort=True):
        key_values = (key,) if len(group_columns) == 1 else tuple(key)
        row: Dict[str, object] = dict(zip(group_columns, key_values))
        row["runs"] = int(len(group))
        if "status" in group:
            row["n_tuning_failed"] = int(group.status.eq("TUNING_FAILED").sum())
            row["n_numerical_failure"] = int(group.status.eq("NUMERICAL_FAILURE").sum())
            row["n_evaluation_attempted"] = int(group.status.ne("TUNING_FAILED").sum())
        complete = group[group.failed == 0] if "failed" in group else group
        row["n_failed"] = int(len(group) - len(complete))
        if {"queries_to_threshold", "restricted_queries_to_threshold"}.intersection(
            metric_columns
        ):
            row["threshold_n_censored"] = int((complete.threshold_reached == 0).sum())
            row["threshold_budget_comparable"] = (
                complete.threshold_budget.nunique() <= 1
            )
            row["threshold_n_complete"] = len(complete)
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            if "failed" in group and metric != "failed":
                values = values.copy()
                values[group.failed.to_numpy() != 0] = np.nan
            if (
                metric == "restricted_queries_to_threshold"
                and not row["threshold_budget_comparable"]
            ):
                values = np.full_like(values, np.nan)
            finite = np.sort(values[np.isfinite(values)])
            row[f"{metric}_n_valid"] = int(finite.size)
            row[f"{metric}_n_undefined"] = int(len(values) - finite.size)
            row[f"{metric}_aggregation"] = (
                "conditional_on_reaching_threshold"
                if metric == "queries_to_threshold"
                else "mean_of_valid_runs"
            )
            if finite.size == 0:
                for suffix in (
                    "mean",
                    "std",
                    "median",
                    "q25",
                    "q75",
                    "ci_low",
                    "ci_high",
                ):
                    row[f"{metric}_{suffix}"] = float("nan")
                continue
            rng = np.random.default_rng(identity_seed(seed, "bootstrap", metric))
            low, high = _bootstrap_mean_interval(finite, repetitions, rng)
            row[f"{metric}_mean"] = float(np.mean(finite))
            row[f"{metric}_std"] = float(
                np.std(finite, ddof=1) if len(finite) > 1 else np.nan
            )
            row[f"{metric}_median"] = float(np.median(finite))
            row[f"{metric}_q25"] = float(np.quantile(finite, 0.25))
            row[f"{metric}_q75"] = float(np.quantile(finite, 0.75))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


CurveStore = Dict[Tuple[object, ...], List[Tuple[np.ndarray, np.ndarray]]]


def _aggregate_curves(
    store: CurveStore, key_columns: Sequence[str], axis_name: str
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for key, samples in sorted(store.items(), key=lambda item: str(item[0])):
        if not samples:
            continue
        axis = samples[0][0]
        if any(not np.array_equal(axis, sample_axis) for sample_axis, _ in samples):
            raise ValueError(f"Curve axis mismatch for key={key}")
        values = np.vstack([sample_values for _, sample_values in samples])
        metadata = dict(zip(key_columns, key))
        # Rendering summaries may be sparse; the raw histories retain every tick.
        indices = np.unique(
            np.linspace(0, len(axis) - 1, min(201, len(axis)), dtype=int)
        )
        for index in indices:
            axis_value = axis[index]
            point_values = values[:, index]
            finite = point_values[np.isfinite(point_values)]
            if finite.size:
                mean = float(np.mean(finite))
                std = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
                half_width = 1.96 * std / math.sqrt(finite.size)
            else:
                mean = std = half_width = float("nan")
            rows.append(
                {
                    **metadata,
                    axis_name: float(axis_value),
                    "mean": mean,
                    "std": std,
                    "ci_low": mean - half_width,
                    "ci_high": mean + half_width,
                    "runs": int(finite.size),
                    "n_total": int(len(point_values)),
                    "n_undefined": int(len(point_values) - finite.size),
                }
            )
    return pd.DataFrame(rows)


def _unit_direction(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=dim)
    return direction / np.linalg.norm(direction)


def _calibration_cases(profile: ExperimentProfile) -> List[CalibrationCase]:
    if profile.name == "smoke":
        return [
            CalibrationCase("reference", "reference", 1.0, dim=5),
            CalibrationCase(
                "eta_noise_003",
                "eta_noise",
                0.03,
                dim=5,
                learning_rate=0.03,
                drift_per_step=0.0,
            ),
            CalibrationCase(
                "eta_noise_008",
                "eta_noise",
                0.08,
                dim=5,
                learning_rate=0.08,
                drift_per_step=0.0,
            ),
            CalibrationCase(
                "eta_drift_003",
                "eta_drift",
                0.03,
                dim=5,
                learning_rate=0.03,
                gradient_noise=0.0,
            ),
            CalibrationCase(
                "eta_drift_008",
                "eta_drift",
                0.08,
                dim=5,
                learning_rate=0.08,
                gradient_noise=0.0,
            ),
        ]

    cases: List[CalibrationCase] = [CalibrationCase("reference", "reference", 1.0)]
    for learning_rate in (0.01, 0.025, 0.05, 0.09):
        token = str(learning_rate).replace(".", "p")
        cases.append(
            CalibrationCase(
                f"eta_noise_{token}",
                "eta_noise",
                learning_rate,
                learning_rate=learning_rate,
                drift_per_step=0.0,
            )
        )
        cases.append(
            CalibrationCase(
                f"eta_drift_{token}",
                "eta_drift",
                learning_rate,
                learning_rate=learning_rate,
                gradient_noise=0.0,
            )
        )
    for sigma in (0.005, 0.01, 0.02, 0.05):
        token = str(sigma).replace(".", "p")
        cases.append(
            CalibrationCase(
                f"sigma_{token}",
                "sigma",
                sigma,
                gradient_noise=sigma,
                drift_per_step=0.0,
            )
        )
    for drift in (0.001, 0.003, 0.01, 0.03):
        token = str(drift).replace(".", "p")
        cases.append(
            CalibrationCase(
                f"drift_{token}",
                "drift",
                drift,
                gradient_noise=0.0,
                drift_per_step=drift,
            )
        )
    for dim in (5, 20, 100):
        cases.append(
            CalibrationCase(f"dimension_{dim}", "dimension", float(dim), dim=dim)
        )
    for condition in (1.0, 5.0, 20.0):
        cases.append(
            CalibrationCase(
                f"condition_{condition:g}",
                "condition_number",
                condition,
                condition_number=condition,
                learning_rate=min(0.05, 0.45 / condition),
            )
        )
    return cases


@_series
def run_calibration_study(profile: ExperimentProfile) -> StudyResult:
    """Compare empirical SGD MSE with its exact linear-system prediction."""

    run_rows: List[Dict[str, object]] = []
    curves: CurveStore = {}
    cases = _calibration_cases(profile)

    for case in cases:
        landscape_seed = _stable_seed(
            17, "calibration", case.dim, case.condition_number
        )
        landscape = QuadraticLandscape(
            dim=case.dim,
            condition_number=case.condition_number,
            seed=landscape_seed,
        )
        hessian = landscape.A
        direction = _unit_direction(
            case.dim,
            _stable_seed(19, "calibration-direction", case.dim, case.condition_number),
        )
        velocity = case.drift_per_step * direction
        transition = np.eye(case.dim) - case.learning_rate * hessian
        x0 = np.ones(case.dim, dtype=float) / math.sqrt(case.dim)

        for seed in profile.seeds:
            environment = DynamicEnvironment(
                case.dim,
                LinearDrift(velocity),
                landscape,
                initial_theta=np.zeros(case.dim),
                bounds=None,
            )
            records, counts = _first_order_trace(
                environment,
                create_optimizer("SGD", {"lr": case.learning_rate}),
                x0,
                profile.calibration_steps,
                GaussianNoise(
                    case.gradient_noise, _stable_seed(seed, "calibration", case.case_id)
                ),
            )
            exact_mean = x0.copy()
            exact_covariance = np.zeros((case.dim, case.dim))
            empirical_mse = np.array(
                [
                    float(
                        (record.observation.x - record.theta)
                        @ (record.observation.x - record.theta)
                    )
                    for record in records
                ]
            )
            exact_mse = np.empty(profile.calibration_steps)

            for step in range(profile.calibration_steps):
                exact_mse[step] = float(
                    exact_mean @ exact_mean + np.trace(exact_covariance)
                )

                exact_mean = transition @ exact_mean - velocity
                exact_covariance = (
                    transition @ exact_covariance @ transition.T
                    + case.learning_rate**2 * case.gradient_noise**2 * np.eye(case.dim)
                )

            empirical_tail = _tail_mean(empirical_mse)
            exact_tail = _tail_mean(exact_mse)
            run_rows.append(
                {
                    **asdict(case),
                    "seed": seed,
                    **counts,
                    "tail_empirical_mse": empirical_tail,
                    "tail_exact_mse": exact_tail,
                    "empirical_exact_ratio": empirical_tail / max(exact_tail, 1e-15),
                    "final_empirical_mse": float(empirical_mse[-1]),
                    "final_exact_mse": float(exact_mse[-1]),
                }
            )
            axis = np.arange(profile.calibration_steps, dtype=float)
            curves.setdefault((case.case_id, case.sweep, "empirical_mse"), []).append(
                (axis, empirical_mse)
            )
            curves.setdefault((case.case_id, case.sweep, "exact_mse"), []).append(
                (axis, exact_mse)
            )

    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(
        runs,
        group_columns=("case_id", "sweep", "x_value"),
        metric_columns=(
            "tail_empirical_mse",
            "tail_exact_mse",
            "empirical_exact_ratio",
        ),
        repetitions=profile.bootstrap_repetitions,
        seed=711,
    )
    curve_frame = _aggregate_curves(curves, ("case_id", "sweep", "series"), "step")

    expected_slopes = {
        "eta_noise": 1.0,
        "eta_drift": -2.0,
        "sigma": 2.0,
        "drift": 2.0,
    }
    slope_rows: List[Dict[str, object]] = []
    grouped = runs.groupby(["case_id", "sweep", "x_value"], as_index=False)[
        "tail_empirical_mse"
    ].mean()
    for sweep, expected in expected_slopes.items():
        subset = grouped[grouped["sweep"] == sweep]
        subset = subset[(subset["x_value"] > 0) & (subset["tail_empirical_mse"] > 0)]
        if len(subset) < 2:
            continue
        slope, intercept = np.polyfit(
            np.log(subset["x_value"]), np.log(subset["tail_empirical_mse"]), 1
        )
        slope_rows.append(
            {
                "sweep": sweep,
                "estimated_log_log_slope": float(slope),
                "expected_slope": expected,
                "absolute_slope_error": float(abs(slope - expected)),
                "log_intercept": float(intercept),
                "points": int(len(subset)),
            }
        )

    return StudyResult(
        runs=runs,
        summary=summary,
        curves=curve_frame,
        extras={"slopes": pd.DataFrame(slope_rows)},
        histories=_pack_histories(
            curves, ("case_id", "sweep", "series"), profile.seeds
        ),
    )


def _memory_scenarios(profile: ExperimentProfile) -> List[MemoryScenario]:
    scenarios = [
        MemoryScenario("stationary_noisy", "stationary", gradient_noise=0.05),
        MemoryScenario(
            "linear_slow", "linear", gradient_noise=0.02, drift_per_step=0.002
        ),
        MemoryScenario(
            "linear_fast", "linear", gradient_noise=0.02, drift_per_step=0.02
        ),
        MemoryScenario(
            "cyclic_fast", "cyclic", gradient_noise=0.02, amplitude=1.0, period=50
        ),
        MemoryScenario(
            "jump_noisy",
            "jump",
            gradient_noise=0.05,
            amplitude=1.0,
            jump_interval=max(40, profile.memory_steps // 4),
        ),
    ]
    if profile.name == "paper":
        scenarios.extend(
            [
                MemoryScenario(
                    "cyclic_slow",
                    "cyclic",
                    gradient_noise=0.02,
                    amplitude=1.0,
                    period=400,
                ),
                MemoryScenario(
                    "jump_clean",
                    "jump",
                    gradient_noise=0.0,
                    amplitude=1.0,
                    jump_interval=400,
                ),
            ]
        )
    return scenarios


MEMORY_PARAMETERS = {
    "SGD": {"lr": 0.05},
    "HeavyBall": {"lr": 0.05, "beta": 0.9},
    "Nesterov": {"lr": 0.05, "beta": 0.9},
    "Adam": {"lr": 0.02},
    "AMSGrad": {"lr": 0.02},
    "AdaGrad": {"lr": 0.1, "eps": 1e-8},
}


def _memory_methods() -> Mapping[str, Callable[[], OptimizerProtocol]]:
    return {
        name: (lambda name=name, params=params: create_optimizer(name, params))
        for name, params in MEMORY_PARAMETERS.items()
    }


def _target_path(
    scenario: MemoryScenario, steps: int, direction: np.ndarray
) -> np.ndarray:
    times = np.arange(steps, dtype=float)
    if scenario.kind == "stationary":
        coefficients = np.zeros(steps)
    elif scenario.kind == "linear":
        coefficients = scenario.drift_per_step * times
    elif scenario.kind == "cyclic":
        coefficients = scenario.amplitude * np.sin(
            2.0 * np.pi * times / scenario.period
        )
    elif scenario.kind == "jump":
        segments = np.floor_divide(np.arange(steps), scenario.jump_interval)
        coefficients = np.where(
            segments == 0,
            0.0,
            np.where(segments % 2 == 1, scenario.amplitude, -scenario.amplitude),
        )
    else:
        raise ValueError(f"Unknown target path: {scenario.kind}")
    return coefficients[:, None] * direction[None, :]


def _phase_lag(estimate: np.ndarray, target: np.ndarray, period: int) -> float:
    if period <= 0:
        return float("nan")
    start = len(estimate) // 3
    estimate = estimate[start:]
    target = target[start:]
    best_lag = 0
    best_correlation = -np.inf
    for lag in range(0, max(1, period // 2) + 1):
        if lag == 0:
            left, right = estimate, target
        else:
            left, right = estimate[lag:], target[:-lag]
        if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return float(best_lag) if np.isfinite(best_correlation) else float("nan")


def _time_to_recovery(
    errors: np.ndarray, theta_path: np.ndarray, scenario: MemoryScenario, dim: int
) -> float:
    events = _recovery_events(errors, theta_path, scenario, dim)
    return (
        float(np.median([e["restricted_time"] for e in events]))
        if events
        else float("nan")
    )


def _recovery_events(errors, theta_path, scenario, dim):
    if scenario.kind != "jump":
        return []
    journal = RecoveryEvents(0, 0)
    noise_floor = 3.0 * scenario.gradient_noise * math.sqrt(dim)
    for t, error in enumerate(errors):
        jump_size = (
            float(np.linalg.norm(theta_path[t] - theta_path[t - 1])) if t else 0.0
        )
        threshold = max(0.2 * jump_size, noise_floor)
        journal.update(t, theta_path[t], error, epsilon=threshold)
    return journal.events()


@_series
def run_memory_study(profile: ExperimentProfile) -> StudyResult:
    """Measure when optimizer memory denoises and when it becomes stale."""

    dim = 20 if profile.name == "paper" else 5
    condition_number = 5.0
    landscape = QuadraticLandscape(dim, condition_number, seed=31415)
    hessian = landscape.A
    direction = _unit_direction(dim, 27182)
    scenarios = _memory_scenarios(profile)
    methods = _memory_methods()
    run_rows: List[Dict[str, object]] = []
    event_rows = []
    curves: CurveStore = {}

    for scenario in scenarios:
        theta_path = _target_path(scenario, profile.memory_steps + 1, direction)
        target_projection = theta_path[:-1] @ direction
        for seed in profile.seeds:
            for method_name, factory in methods.items():
                records, counts = _first_order_trace(
                    _path_environment(landscape, theta_path),
                    factory(),
                    np.zeros(dim),
                    profile.memory_steps,
                    GaussianNoise(
                        scenario.gradient_noise,
                        _stable_seed(seed, "memory", scenario.name),
                    ),
                )
                squared_error = np.array(
                    [
                        float(
                            (record.observation.x - record.theta)
                            @ (record.observation.x - record.theta)
                        )
                        for record in records
                    ]
                )
                regret = np.array(
                    [record.clean_value - record.optimum_value for record in records]
                )
                estimate_projection = np.array(
                    [record.observation.x @ direction for record in records]
                )

                error_norm = np.sqrt(squared_error)
                events = _recovery_events(error_norm, theta_path[:-1], scenario, dim)
                event_rows.extend(
                    {
                        "scenario": scenario.name,
                        "method": method_name,
                        "seed": seed,
                        **event,
                    }
                    for event in events
                )
                phase = _phase_lag(
                    estimate_projection, target_projection, scenario.period
                )
                run_rows.append(
                    {
                        **asdict(scenario),
                        "method": method_name,
                        "seed": seed,
                        "dim": dim,
                        "condition_number": condition_number,
                        **counts,
                        "tail_mse": _tail_mean(squared_error),
                        "tail_regret": _tail_mean(regret),
                        "tail_p90_error": float(
                            np.quantile(error_norm[int(0.8 * len(error_norm)) :], 0.9)
                        ),
                        "phase_lag_steps": phase,
                        "phase_lag_status": (
                            "estimated" if np.isfinite(phase) else "not_estimable"
                        ),
                        "time_to_recovery": (
                            float(np.median([e["restricted_time"] for e in events]))
                            if events
                            else np.nan
                        ),
                        "recovery_status": "events_observed" if events else "no_events",
                        "recovery_n_events": len(events),
                        "recovery_n_censored": sum(e["censored"] for e in events),
                        "recovery_fraction": (
                            float(np.mean([e["recovered"] for e in events]))
                            if events
                            else np.nan
                        ),
                        "recovery_aggregation": "median_capped_within_run;mean_between_runs",
                    }
                )
                axis = np.arange(profile.memory_steps, dtype=float)
                curves.setdefault(
                    (scenario.name, method_name, "squared_error"), []
                ).append((axis, squared_error))

    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(
        runs,
        group_columns=("name", "kind", "method"),
        metric_columns=(
            "tail_mse",
            "tail_regret",
            "tail_p90_error",
            "phase_lag_steps",
            "time_to_recovery",
        ),
        repetitions=profile.bootstrap_repetitions,
        seed=811,
    )
    curve_frame = _aggregate_curves(curves, ("scenario", "method", "metric"), "step")
    return StudyResult(
        runs,
        summary,
        curve_frame,
        extras={"recovery_events": pd.DataFrame(event_rows)},
        histories=_pack_histories(
            curves, ("scenario", "method", "metric"), profile.seeds
        ),
    )


def _zero_order_cases(profile: ExperimentProfile) -> List[ZeroOrderCase]:
    if profile.name == "smoke":
        return [
            ZeroOrderCase("stationary_d20", 20, 5.0, "stationary"),
            ZeroOrderCase(
                "linear_fast_d20",
                20,
                5.0,
                "linear",
                drift_per_query=0.001,
            ),
            ZeroOrderCase(
                "cyclic_fast_d20",
                20,
                5.0,
                "cyclic",
                amplitude=1.0,
                period=160,
            ),
        ]
    return [
        ZeroOrderCase("stationary_d20", 20, 5.0, "stationary"),
        ZeroOrderCase(
            "stationary_noise_high_d20",
            20,
            5.0,
            "stationary",
            observation_noise=0.005,
        ),
        ZeroOrderCase("linear_slow_d20", 20, 5.0, "linear", drift_per_query=0.00005),
        ZeroOrderCase("linear_fast_d20", 20, 5.0, "linear", drift_per_query=0.001),
        ZeroOrderCase(
            "cyclic_slow_d20",
            20,
            5.0,
            "cyclic",
            amplitude=1.0,
            period=5000,
        ),
        ZeroOrderCase(
            "cyclic_fast_d20",
            20,
            5.0,
            "cyclic",
            amplitude=1.0,
            period=200,
        ),
    ]


def _zero_order_methods() -> Tuple[ZeroOrderMethodConfig, ...]:
    return (
        ZeroOrderMethodConfig("FirstOrderReference", "first_order"),
        ZeroOrderMethodConfig("SPSA", "spsa"),
        ZeroOrderMethodConfig("SphericalZOSGD-m1", "spherical", directions=1),
        ZeroOrderMethodConfig("SphericalZOSGD-m5", "spherical", directions=5),
        ZeroOrderMethodConfig("SphericalZOSGD-m20", "spherical", directions=20),
        ZeroOrderMethodConfig("KieferWolfowitz", "coordinate"),
        ZeroOrderMethodConfig("OnePointSPSA", "one_point"),
        ZeroOrderMethodConfig("ZOSignSGD", "zo_sign"),
        ZeroOrderMethodConfig("AcceleratedSPSA", "accelerated"),
        ZeroOrderMethodConfig("RandomSearch", "random_search"),
    )


def _zero_order_theta_at(
    case: ZeroOrderCase, direction: np.ndarray, query: int
) -> np.ndarray:
    if case.path_kind == "stationary":
        coefficient = 0.0
    elif case.path_kind == "linear":
        coefficient = case.drift_per_query * query
    elif case.path_kind == "cyclic":
        coefficient = case.amplitude * math.sin(2.0 * math.pi * query / case.period)
    else:
        raise ValueError(f"Unknown zero-order target path: {case.path_kind}")
    return coefficient * direction


def _sample_query_curve(
    query_history: Sequence[int],
    state_history: Sequence[np.ndarray],
    checkpoints: np.ndarray,
    theta_at: Callable[[int], np.ndarray],
) -> np.ndarray:
    queries = np.asarray(query_history)
    values = np.empty(len(checkpoints))
    for index, checkpoint in enumerate(checkpoints):
        history_index = int(np.searchsorted(queries, checkpoint, side="right") - 1)
        history_index = max(history_index, 0)
        error = state_history[history_index] - theta_at(int(checkpoint))
        values[index] = float(error @ error)
    return values


@_series
def _run_zero_order_method(
    method: ZeroOrderMethodConfig,
    protocol: str,
    case: ZeroOrderCase,
    budget: int,
    checkpoints: np.ndarray,
    seed: int,
    hyperparameters: ZeroOrderHyperparameters,
    *,
    full_history: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if hyperparameters is None:
        # Failed development selection is not an evaluation trajectory.
        spec = resolve_optimizer_spec(method.catalog_name)
        full_error = np.full(budget + 1, np.nan)
        metrics = {
            key: float("nan")
            for key in (
                "final_mse",
                "query_averaged_mse",
                "tail_mse",
                "queries_to_threshold",
                "restricted_queries_to_threshold",
                "mean_gradient_cosine",
                "maximum_gradient_norm",
            )
        }
        metrics.update(
            status="TUNING_FAILED",
            failed=1.0,
            threshold_status="not_evaluated",
            threshold_reached=False,
            threshold_budget=0,
            threshold_n_valid=0,
            gradient_cosine_n_valid=0,
            gradient_cosine_n_undefined=0,
            gradient_diagnostic_status="not_evaluated",
            updates=0,
            queries=0,
            value_queries=0,
            gradient_queries=0,
            unused_queries=budget,
            parameters=None,
            implementation_id=method.implementation_id,
            implementation_version=spec.implementation_version,
            rng_stream_id=method.rng_stream_id,
            source=spec.source,
            variant=spec.variant,
            geometry="euclidean",
            directions=method.directions,
        )
        result = (np.full(len(checkpoints), np.nan), metrics)
        return (*result, full_error) if full_history else result
    landscape = QuadraticLandscape(
        case.dim,
        case.condition_number,
        seed=_stable_seed(41, "zero-landscape", case.dim, case.condition_number),
    )
    direction = _unit_direction(case.dim, _stable_seed(43, "zero-direction", case.dim))
    theta_at = lambda query: _zero_order_theta_at(case, direction, query)
    path = np.array([theta_at(query) for query in range(budget + 1)])
    environment = _path_environment(landscape, path)
    # Observation noise and estimator directions are independent named streams.
    # Frozen/streaming comparisons share both, without using presentation names.
    noise = GaussianNoise(
        case.observation_noise,
        identity_seed(
            seed,
            PAPER_GENERATOR_ID,
            "evaluation" if seed in range(1001, 1031) else "development",
            "zero",
            case.case_id,
            "value_noise",
        ),
    )
    first_order = method.family == "first_order"
    oracle = (
        FirstOrderOracle(environment, blind_value=True)
        if first_order
        else ZeroOrderOracle(environment, value_noise=noise)
    )
    session = OracleSession(oracle, budget, protocol=protocol)
    algorithm = create_optimizer(method.catalog_name, hyperparameters.parameters)
    x = _unit_direction(case.dim, _stable_seed(47, "zero-initial", case.dim))
    rng = np.random.default_rng(
        identity_seed(
            seed,
            "paper",
            "evaluation" if seed in range(1001, 1031) else "development",
            "zero",
            case.case_id,
            method.rng_stream_id,
            "directions",
        )
    )
    _user_call(
        lambda *args: algorithm.initialize(*args), x.copy(), rng, TaskSpec(case.dim)
    )
    query_history, state_history = [0], [x.copy()]
    updates, failed = 0, False
    failure_query = budget + 1
    maximum_gradient_norm = 0.0
    diagnostics = []
    cost = method.query_cost(case.dim)
    try:
        while session.remaining:
            previous = x.copy()
            with session.client() as client:
                try:
                    update = algorithm.advance(client)
                except (FloatingPointError, OverflowError):
                    failed = True
            if failed:
                failure_query = session.n_queries
                break
            query = session.n_queries
            if not update.update_complete:
                query_history.append(query)
                state_history.append(x.copy())
                break
            gradient = update.estimated_gradient
            gradient_norm = (
                None if gradient is None else float(np.linalg.norm(gradient))
            )
            if gradient_norm is not None:
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            x = update.recommendation.copy()
            if (
                gradient_norm is not None
                and (not np.isfinite(gradient_norm) or gradient_norm > 1e6)
            ) or np.linalg.norm(x) > 1e6:
                failed, failure_query = True, query
                query_history.append(query)
                state_history.append(x.copy())
                break
            if gradient is not None:
                point = (
                    update.estimation_point
                    if update.estimation_point is not None
                    else previous
                )
                diagnostics.append((point.copy(), gradient.copy(), query))
            updates += 1
            query_history.append(query)
            state_history.append(x.copy())
    except BaseException:
        session.abort()
        raise
    session.finish()

    # Evaluator-only diagnostics happen after the query capability is closed.
    gradient_cosines = []
    for point, gradient, query in diagnostics:
        current_gradient = landscape.grad(point, path[query])
        denominator = float(np.linalg.norm(gradient)) * float(
            np.linalg.norm(current_gradient)
        )
        if denominator > 0.0:
            gradient_cosines.append(float(gradient @ current_gradient / denominator))

    sampled_error = _sample_query_curve(
        query_history, state_history, checkpoints, theta_at
    )
    # Definition v2: recommended iterate at EVERY integer query tick. The
    # objective is the arithmetic mean at q=0,...,B-1; q=B is terminal only.
    full_axis = np.arange(budget + 1)
    full_error = _sample_query_curve(query_history, state_history, full_axis, theta_at)
    if failed:
        sampled_error[checkpoints >= failure_query] = np.nan
        full_error[full_axis >= failure_query] = np.nan
    finite_curve = bool(np.all(np.isfinite(full_error)))
    mean_error = float(np.mean(full_error[:-1])) if finite_curve else float("nan")
    threshold_hits = np.flatnonzero(
        np.isfinite(full_error) & (full_error <= case.threshold_epsilon**2)
    )
    reached = bool(threshold_hits.size)
    observation_budget = int(failure_query if failed else budget)
    metrics = {
        "final_mse": float(full_error[-1]) if finite_curve else float("nan"),
        "query_averaged_mse": mean_error,
        "tail_mse": _tail_mean(full_error[:-1]) if finite_curve else float("nan"),
        "queries_to_threshold": (
            float(full_axis[threshold_hits[0]]) if threshold_hits.size else float("nan")
        ),
        "threshold_reached": float(reached),
        "threshold_status": (
            "numerical_failure"
            if failed
            else "reached" if reached else "right_censored"
        ),
        "threshold_epsilon": case.threshold_epsilon,
        "threshold_budget": observation_budget,
        "threshold_n_valid": int(np.isfinite(full_error).sum()),
        "threshold_decision_stream": "recommended_integer_query_ticks",
        "restricted_queries_to_threshold": (
            float("nan")
            if failed
            else float(full_axis[threshold_hits[0]]) if reached else observation_budget
        ),
        "mean_gradient_cosine": (
            float(np.mean(gradient_cosines)) if gradient_cosines else float("nan")
        ),
        "gradient_cosine_n_valid": len(gradient_cosines),
        "gradient_cosine_n_undefined": len(diagnostics) - len(gradient_cosines),
        "updates": float(updates),
        "queries_per_update": float(cost),
        **session.counts,
        "unused_queries": session.remaining,
        "parameters": dict(hyperparameters.parameters),
        "implementation_id": method.implementation_id,
        "implementation_version": algorithm.implementation_version,
        "rng_stream_id": method.rng_stream_id,
        "source": algorithm.source,
        "variant": algorithm.variant,
        "geometry": "euclidean",
        "directions": float(method.directions),
        "maximum_gradient_norm": (
            maximum_gradient_norm if method.family != "random_search" else float("nan")
        ),
        "gradient_diagnostic_status": (
            "not_applicable"
            if method.family == "random_search"
            else ("available" if gradient_cosines else "undefined")
        ),
        "status": "NUMERICAL_FAILURE" if failed else "SUCCESS",
        "failed": float(failed),
    }
    return (
        (sampled_error, metrics, full_error)
        if full_history
        else (sampled_error, metrics)
    )


def _zero_order_candidate_hyperparameters(
    profile: ExperimentProfile, method: ZeroOrderMethodConfig, dim: int
) -> List[dict]:
    if method.family == "first_order":
        return [{"lr": 0.04}]
    if method.family == "random_search":
        return [
            {"sigma": s}
            for s in (0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2)
        ]
    constants = (
        (0.01, 0.02, 0.04, 0.08)
        if method.family == "coordinate"
        else (0.05, 0.1, 0.2, 0.4)
    )
    rows = []
    for c in constants:
        for r in (0.03, 0.1, 0.3):
            if method.family == "spsa":
                params = dict(alpha=c, beta=r)
            elif method.family == "accelerated":
                params = dict(h=c, beta=r, a=0.1, delta=1.05)
            elif method.family == "spherical":
                params = dict(
                    lr=c * min(1.0, method.directions / dim), h=r, m=method.directions
                )
            elif method.family == "one_point":
                params = dict(lr=c / dim, r=r)
            elif method.family == "zo_sign":
                params = dict(lr=c / np.sqrt(dim), r=r, m=method.directions)
            elif method.family == "coordinate":
                params = dict(lr=c, h=r)
            else:
                raise ValueError(f"Unknown family {method.family}")
            rows.append(params)
    return rows


class _TuningCache(TruthGuard):
    """Series-owned evidence, never a mutable public cache of intermediate truth."""

    @_internal
    def __init__(self):
        self.entries = {}
        _bind(self)


@_series
def _tune_zero_order_methods(
    profile: ExperimentProfile,
    cases: Sequence[ZeroOrderCase],
    methods: Sequence[ZeroOrderMethodConfig],
    cache=None,
) -> Tuple[Dict[Tuple[int, str], ZeroOrderHyperparameters], pd.DataFrame]:
    dimensions = sorted({case.dim for case in cases})
    tuning_seeds = (9001, 9002) if profile.name == "smoke" else tuple(range(9001, 9006))
    checkpoints = np.unique(
        np.linspace(
            0,
            profile.zero_order_tuning_budget,
            min(profile.curve_points, 41),
            dtype=int,
        )
    ).astype(float)
    selected: Dict[Tuple[int, str], ZeroOrderHyperparameters] = {}
    rows: List[Dict[str, object]] = []
    if cache is None:
        cache = _TuningCache()
    if type(cache) is not _TuningCache:
        raise TypeError("Expected a series-owned tuning cache")
    if _owner(cache) is not _current.get():
        raise ValueError("Tuning cache must belong to the current evaluation series")

    for dim in dimensions:
        tuning_case = ZeroOrderCase(f"tuning_stationary_d{dim}", dim, 5.0, "stationary")
        for method in methods:
            if method.family == "first_order":
                selected[(dim, method.name)] = ZeroOrderHyperparameters({"lr": 0.04})
                continue
            candidates = _zero_order_candidate_hyperparameters(profile, method, dim)
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "protocol": TUNING_PROTOCOL,
                        "profile": profile.name,
                        "case": asdict(tuning_case),
                        "seeds": tuning_seeds,
                        "budget": profile.zero_order_tuning_budget,
                        "candidates": candidates,
                        "implementation": method.implementation_id,
                        "rng": method.rng_stream_id,
                        "generator": PAPER_GENERATOR_ID,
                        "clock": "frozen",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if fingerprint in cache.entries:
                cached_choice, cached_rows = cache.entries[fingerprint]
                selected[(dim, method.name)] = cached_choice
                rows.extend(
                    [
                        {
                            **row,
                            "method": method.name,
                            "reused": True,
                            "charged_queries": 0,
                        }
                        for row in cached_rows
                    ]
                )
                continue
            candidate_rows: List[Dict[str, object]] = []
            for candidate in candidates:
                parameters = ZeroOrderHyperparameters(candidate)
                scores: List[float] = []
                failures = 0
                trials = []
                for seed in tuning_seeds:
                    _, metrics = _run_zero_order_method(
                        method,
                        "frozen",
                        tuning_case,
                        profile.zero_order_tuning_budget,
                        checkpoints,
                        seed,
                        parameters,
                    )
                    failures += int(metrics["failed"])
                    trials.append(
                        {
                            "seed": seed,
                            "queries": metrics["queries"],
                            "status": metrics["status"],
                            "score": metrics["query_averaged_mse"],
                        }
                    )
                    if np.isfinite(metrics["query_averaged_mse"]):
                        scores.append(float(metrics["query_averaged_mse"]))
                score = (
                    float(np.mean(scores))
                    if len(scores) == len(tuning_seeds) and failures == 0
                    else float("nan")
                )
                candidate_rows.append(
                    {
                        "dim": dim,
                        "method": method.name,
                        "family": method.family,
                        "directions": method.directions,
                        "parameters": dict(candidate),
                        "fingerprint": fingerprint,
                        "implementation_id": method.implementation_id,
                        "implementation_version": resolve_optimizer_spec(
                            method.catalog_name
                        ).implementation_version,
                        "rng_stream_id": method.rng_stream_id,
                        "trials": trials,
                        "charged_queries": sum(t["queries"] for t in trials),
                        "maximum_queries": len(tuning_seeds)
                        * profile.zero_order_tuning_budget,
                        "reused": False,
                        "status": (
                            "VALID" if np.isfinite(score) else "NUMERICAL_FAILURE"
                        ),
                        "tuning_score": score,
                        "failure_rate": failures / len(tuning_seeds),
                        "selected": False,
                    }
                )
            valid = [row for row in candidate_rows if np.isfinite(row["tuning_score"])]
            choice = None
            if valid:
                best = min(
                    valid,
                    key=lambda row: (
                        row["tuning_score"],
                        tuple(sorted(row["parameters"].items())),
                    ),
                )
                best["selected"] = True
                choice = ZeroOrderHyperparameters(
                    dict(best["parameters"]), best["tuning_score"]
                )
            for row in candidate_rows:
                row["selection_status"] = (
                    "SELECTED" if choice is not None else "TUNING_FAILED"
                )
            selected[(dim, method.name)] = choice
            cache.entries[fingerprint] = (choice, candidate_rows)
            rows.extend(candidate_rows)
    return selected, pd.DataFrame(rows)


@_series
def run_zero_order_study(
    profile: ExperimentProfile, *, _tuning_cache=None
) -> StudyResult:
    """Measure the variance--freshness trade-off at a fixed dimension."""

    cases = _zero_order_cases(profile)
    methods = _zero_order_methods()
    selected_hyperparameters, tuning = _tune_zero_order_methods(
        profile, cases, methods, _tuning_cache
    )
    checkpoints = np.unique(
        np.linspace(0, profile.zero_order_budget, profile.curve_points, dtype=int)
    ).astype(float)
    run_rows: List[Dict[str, object]] = []
    curves: CurveStore = {}

    for case in cases:
        protocols = (
            ("frozen",)
            if case.path_kind == "stationary"
            else (
                "frozen",
                "streaming",
            )
        )
        for protocol in protocols:
            for method in methods:
                for seed in profile.seeds:
                    error_curve, metrics, full_error = _run_zero_order_method(
                        method,
                        protocol,
                        case,
                        profile.zero_order_budget,
                        checkpoints,
                        seed,
                        selected_hyperparameters[(case.dim, method.name)],
                        full_history=True,
                    )
                    run_rows.append(
                        {
                            **asdict(case),
                            "protocol": protocol,
                            "method": method.name,
                            "family": method.family,
                            "seed": seed,
                            **metrics,
                        }
                    )
                    curves.setdefault(
                        (case.case_id, protocol, method.name, "squared_error"), []
                    ).append((np.arange(profile.zero_order_budget + 1), full_error))

    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(
        runs,
        group_columns=("case_id", "protocol", "method"),
        metric_columns=(
            "final_mse",
            "query_averaged_mse",
            "tail_mse",
            "restricted_queries_to_threshold",
            "threshold_reached",
            "mean_gradient_cosine",
            "updates",
            "maximum_gradient_norm",
            "failed",
        ),
        repetitions=profile.bootstrap_repetitions,
        seed=911,
    )
    curve_frame = _aggregate_curves(
        curves, ("case_id", "protocol", "method", "metric"), "queries"
    )

    dynamic = runs[runs["path_kind"] != "stationary"]
    paired = dynamic.pivot_table(
        index=["case_id", "method", "seed"],
        columns="protocol",
        values="query_averaged_mse",
        aggfunc="first",
    ).reset_index()
    if {"frozen", "streaming"}.issubset(paired.columns):
        paired = paired[
            (paired["frozen"] > 0)
            & (paired["streaming"] > 0)
            & np.isfinite(paired["frozen"])
            & np.isfinite(paired["streaming"])
        ].copy()
        paired["freshness_log_ratio"] = np.log(paired["streaming"] / paired["frozen"])
        freshness_summary = _summarize_runs(
            paired,
            group_columns=("case_id", "method"),
            metric_columns=("freshness_log_ratio",),
            repetitions=profile.bootstrap_repetitions,
            seed=919,
        )
    else:
        paired = pd.DataFrame()
        freshness_summary = pd.DataFrame()

    selected_frame = tuning[tuning["selected"]].reset_index(drop=True)
    return StudyResult(
        runs,
        summary,
        curve_frame,
        histories=_pack_histories(
            curves, ("case_id", "protocol", "method", "metric"), profile.seeds
        ),
        extras={
            "tuning": tuning,
            "selected_hyperparameters": selected_frame,
            "freshness_runs": paired,
            "freshness_summary": freshness_summary,
        },
    )


def _simplex_feasibility(x: np.ndarray) -> float:
    negative_mass = float(np.sum(np.maximum(-x, 0.0)))
    return abs(float(np.sum(x)) - 1.0) + negative_mass


def _simplex_path(steps: int, dim: int, seed: int, period: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    endpoint_a = rng.dirichlet(np.full(dim, 3.0))
    endpoint_b = rng.dirichlet(np.full(dim, 3.0))
    weights = 0.5 * (1.0 + np.sin(2.0 * np.pi * np.arange(steps, dtype=float) / period))
    return (1.0 - weights)[:, None] * endpoint_a[None, :] + weights[
        :, None
    ] * endpoint_b[None, :]


def _plane_rotation(size: int, angle: float) -> np.ndarray:
    rotation = np.eye(size)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation[0, 0] = cosine
    rotation[0, 1] = -sine
    rotation[1, 0] = sine
    rotation[1, 1] = cosine
    return rotation


def _manifold_path(
    task: str, steps: int, d: int, r: int, seed: int, period: int
) -> np.ndarray:
    base = random_stiefel(d, r, np.random.default_rng(seed))
    path = np.empty((steps, d, r))
    for step in range(steps):
        angle = 0.8 * math.sin(2.0 * math.pi * step / period)
        if task == "grassmann_basis_rotation":
            path[step] = base @ _plane_rotation(r, angle)
        else:
            path[step] = _plane_rotation(d, angle) @ base
    return path


def _run_simplex_geometry(
    profile: ExperimentProfile, seed: int, curves: CurveStore
) -> List[Dict[str, object]]:
    dim = 10
    period = 120 if profile.name == "paper" else 60
    theta_path = _simplex_path(
        profile.geometry_steps + 1,
        dim,
        _stable_seed(seed, "simplex-target"),
        period,
    )
    landscape = SimplexLandscape()
    rows: List[Dict[str, object]] = []
    methods = ("ProjectedGradient", "ExponentiatedGradient", "RawAmbientSGDControl")

    for method in methods:
        records, counts = _first_order_trace(
            _path_environment(landscape, theta_path),
            _geometry_method(method, 0.15),
            np.full(dim, 1.0 / dim),
            profile.geometry_steps,
            GaussianNoise(0.01, _stable_seed(seed, "simplex-noise")),
        )
        loss = np.array([record.clean_value for record in records])
        tracking = np.array(
            [
                float(np.linalg.norm(record.observation.x - record.theta))
                for record in records
            ]
        )
        feasibility = np.array(
            [_simplex_feasibility(record.observation.x) for record in records]
        )

        rows.append(
            {
                "task": "simplex_tracking",
                **counts,
                "method": method,
                "seed": seed,
                "tail_loss": _tail_mean(loss),
                "tail_tracking_error": _tail_mean(tracking),
                "tail_feasibility_residual": _tail_mean(feasibility),
                "tail_frame_error": float("nan"),
                "tail_subspace_error": float("nan"),
            }
        )
        axis = np.arange(profile.geometry_steps, dtype=float)
        curves.setdefault(("simplex_tracking", method, "tracking_error"), []).append(
            (axis, tracking)
        )
        curves.setdefault(
            ("simplex_tracking", method, "feasibility_residual"), []
        ).append((axis, feasibility))
    return rows


def _run_manifold_geometry(
    profile: ExperimentProfile, seed: int, curves: CurveStore
) -> List[Dict[str, object]]:
    d, r = 6, 2
    period = 160 if profile.name == "paper" else 80
    tasks = (
        "stiefel_frame_tracking",
        "grassmann_subspace_tracking",
        "grassmann_basis_rotation",
    )
    methods = ("RiemannianSGD", "PolarProjectedSGDControl", "RawAmbientSGDControl")
    rows: List[Dict[str, object]] = []

    for task in tasks:
        theta_path = _manifold_path(
            task,
            profile.geometry_steps + 1,
            d,
            r,
            _stable_seed(seed, "manifold-target", task),
            period,
        )
        if task == "stiefel_frame_tracking":
            landscape = StiefelLandscape(d, r)
        else:
            landscape = GrassmannLandscape(d, r)
        noise_sigma = 0.0 if task == "grassmann_basis_rotation" else 0.01
        for method in methods:
            records, counts = _first_order_trace(
                _path_environment(landscape, theta_path),
                _geometry_method(method, 0.12, d, r),
                theta_path[0].reshape(-1),
                profile.geometry_steps,
                GaussianNoise(noise_sigma, _stable_seed(seed, "manifold-noise", task)),
            )
            loss = np.array([record.clean_value for record in records])
            tracking = np.empty(profile.geometry_steps)
            feasibility = np.empty(profile.geometry_steps)
            frame_error = np.empty(profile.geometry_steps)
            subspace_error = np.empty(profile.geometry_steps)

            for step, record in enumerate(records):
                x = record.observation.x.reshape(d, r)
                theta = record.theta.reshape(d, r)
                projected_x = project_to_stiefel(x)
                frame_error[step] = float(
                    np.linalg.norm(projected_x - theta, ord="fro")
                )
                subspace_error[step] = principal_angle_distance(projected_x, theta)
                feasibility[step] = float(
                    np.linalg.norm(x.T @ x - np.eye(r), ord="fro")
                )
                tracking[step] = (
                    frame_error[step]
                    if task == "stiefel_frame_tracking"
                    else subspace_error[step]
                )

            rows.append(
                {
                    "task": task,
                    **counts,
                    "method": method,
                    "seed": seed,
                    "tail_loss": _tail_mean(loss),
                    "tail_tracking_error": _tail_mean(tracking),
                    "tail_feasibility_residual": _tail_mean(feasibility),
                    "tail_frame_error": _tail_mean(frame_error),
                    "tail_subspace_error": _tail_mean(subspace_error),
                }
            )
            axis = np.arange(profile.geometry_steps, dtype=float)
            curves.setdefault((task, method, "tracking_error"), []).append(
                (axis, tracking)
            )
            curves.setdefault((task, method, "feasibility_residual"), []).append(
                (axis, feasibility)
            )
            if task == "grassmann_basis_rotation":
                curves.setdefault((task, method, "frame_error"), []).append(
                    (axis, frame_error)
                )
    return rows


@_series
def run_geometry_study(profile: ExperimentProfile) -> StudyResult:
    """Compare unconstrained, projected, and intrinsic geometry-aware updates."""

    curves: CurveStore = {}
    run_rows: List[Dict[str, object]] = []
    for seed in profile.seeds:
        run_rows.extend(_run_simplex_geometry(profile, seed, curves))
        run_rows.extend(_run_manifold_geometry(profile, seed, curves))
    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(
        runs,
        group_columns=("task", "method"),
        metric_columns=(
            "tail_loss",
            "tail_tracking_error",
            "tail_feasibility_residual",
            "tail_frame_error",
            "tail_subspace_error",
        ),
        repetitions=profile.bootstrap_repetitions,
        seed=1011,
    )
    curve_frame = _aggregate_curves(curves, ("task", "method", "metric"), "step")
    return StudyResult(
        runs,
        summary,
        curve_frame,
        extras={},
        histories=_pack_histories(curves, ("task", "method", "metric"), profile.seeds),
    )


def _zero_order_dimension_cases(profile: ExperimentProfile) -> List[ZeroOrderCase]:
    dimensions = (
        (5, 20, 100)
        if profile.name == "smoke"
        else (
            5,
            10,
            20,
            50,
            100,
            200,
        )
    )
    drift_per_query = 1.0 / profile.zero_order_budget
    cases: List[ZeroOrderCase] = []
    for dim in dimensions:
        cases.append(ZeroOrderCase(f"stationary_d{dim}", dim, 5.0, "stationary"))
        cases.append(
            ZeroOrderCase(
                f"linear_d{dim}",
                dim,
                5.0,
                "linear",
                drift_per_query=drift_per_query,
            )
        )
    return cases


@_series
def run_dimension_study(
    profile: ExperimentProfile, *, _tuning_cache=None
) -> StudyResult:
    """Measure zero-order estimation and tracking as dimension grows."""

    cases = _zero_order_dimension_cases(profile)
    methods = _zero_order_methods()
    selected_hyperparameters, tuning = _tune_zero_order_methods(
        profile, cases, methods, _tuning_cache
    )
    checkpoints = np.unique(
        np.linspace(0, profile.zero_order_budget, profile.curve_points, dtype=int)
    ).astype(float)
    run_rows: List[Dict[str, object]] = []
    curves: CurveStore = {}

    for case in cases:
        for method in methods:
            parameters = selected_hyperparameters[(case.dim, method.name)]
            for seed in profile.seeds:
                error_curve, metrics, full_error = _run_zero_order_method(
                    method,
                    "frozen",
                    case,
                    profile.zero_order_budget,
                    checkpoints,
                    seed,
                    parameters,
                    full_history=True,
                )
                run_rows.append(
                    {
                        **asdict(case),
                        "protocol": "frozen",
                        "method": method.name,
                        "family": method.family,
                        "seed": seed,
                        "query_budget": profile.zero_order_budget,
                        "total_target_displacement": (
                            case.drift_per_query * profile.zero_order_budget
                        ),
                        **metrics,
                    }
                )
                curves.setdefault(
                    (case.path_kind, case.dim, method.name, "squared_error"), []
                ).append((np.arange(profile.zero_order_budget + 1), full_error))

    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(
        runs,
        group_columns=("path_kind", "dim", "method"),
        metric_columns=(
            "final_mse",
            "query_averaged_mse",
            "tail_mse",
            "restricted_queries_to_threshold",
            "threshold_reached",
            "mean_gradient_cosine",
            "updates",
            "maximum_gradient_norm",
            "failed",
        ),
        repetitions=profile.bootstrap_repetitions,
        seed=1221,
    )
    curve_frame = _aggregate_curves(
        curves, ("path_kind", "dim", "method", "metric"), "queries"
    )

    baseline = runs[runs["method"] == "FirstOrderReference"][
        ["path_kind", "dim", "seed", "query_averaged_mse"]
    ].rename(columns={"query_averaged_mse": "first_order_mse"})
    relative = runs.merge(baseline, on=["path_kind", "dim", "seed"], how="left")
    relative = relative[
        (relative["method"] != "FirstOrderReference")
        & (relative["query_averaged_mse"] > 0)
        & (relative["first_order_mse"] > 0)
    ].copy()
    relative["mse_ratio_to_first_order"] = (
        relative["query_averaged_mse"] / relative["first_order_mse"]
    )
    relative_summary = _summarize_runs(
        relative,
        group_columns=("path_kind", "dim", "method"),
        metric_columns=("mse_ratio_to_first_order",),
        repetitions=profile.bootstrap_repetitions,
        seed=1223,
    )

    slope_rows: List[Dict[str, object]] = []
    grouped = runs.groupby(["path_kind", "dim", "method"], as_index=False)[
        ["query_averaged_mse", "tail_mse", "mean_gradient_cosine"]
    ].mean()
    for path_kind in grouped["path_kind"].unique():
        for method in grouped["method"].unique():
            subset = grouped[
                (grouped["path_kind"] == path_kind) & (grouped["method"] == method)
            ]
            for metric in ("query_averaged_mse", "tail_mse"):
                valid = subset[(subset["dim"] > 0) & (subset[metric] > 0)]
                if len(valid) < 2:
                    continue
                slope, intercept = np.polyfit(
                    np.log(valid["dim"]), np.log(valid[metric]), 1
                )
                slope_rows.append(
                    {
                        "path_kind": path_kind,
                        "method": method,
                        "metric": metric,
                        "estimated_log_log_slope": float(slope),
                        "log_intercept": float(intercept),
                        "points": int(len(valid)),
                    }
                )

    return StudyResult(
        runs,
        summary,
        curve_frame,
        histories=_pack_histories(
            curves, ("path_kind", "dim", "method", "metric"), profile.seeds
        ),
        extras={
            "tuning": tuning,
            "selected_hyperparameters": tuning[tuning["selected"]].reset_index(
                drop=True
            ),
            "relative_runs": relative,
            "relative_summary": relative_summary,
            "dimension_slopes": pd.DataFrame(slope_rows),
        },
    )


def _plot_calibration(result: StudyResult, path: Path) -> None:
    grouped = result.runs.groupby("case_id", as_index=False)[
        ["tail_empirical_mse", "tail_exact_mse"]
    ].mean()
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.scatter(
        grouped["tail_exact_mse"], grouped["tail_empirical_mse"], color="#1f77b4"
    )
    positive = grouped[
        (grouped["tail_exact_mse"] > 0) & (grouped["tail_empirical_mse"] > 0)
    ]
    if not positive.empty:
        low = min(
            positive["tail_exact_mse"].min(), positive["tail_empirical_mse"].min()
        )
        high = max(
            positive["tail_exact_mse"].max(), positive["tail_empirical_mse"].max()
        )
        axis.plot([low, high], [low, high], "--", color="black", label="exact match")
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlabel("Exact tail MSE")
    axis.set_ylabel("Empirical tail MSE")
    axis.set_title("SGD calibration on drifting quadratics")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_panel_curves(
    result: StudyResult,
    panel_column: str,
    line_column: str,
    axis_column: str,
    metric_filter: str,
    title: str,
    ylabel: str,
    path: Path,
    maximum_panels: int = 8,
) -> None:
    data = result.curves[result.curves["metric"] == metric_filter]
    panels = list(data[panel_column].drop_duplicates())[:maximum_panels]
    if not panels:
        return
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 3.6 * rows), squeeze=False
    )
    for panel, axis in zip(panels, axes.flat):
        subset = data[data[panel_column] == panel]
        for line_name, line in subset.groupby(line_column):
            line = line.sort_values(axis_column)
            axis.plot(line[axis_column], line["mean"], label=str(line_name))
            axis.fill_between(
                line[axis_column], line["ci_low"], line["ci_high"], alpha=0.15
            )
        axis.set_title(str(panel))
        axis.set_xlabel(axis_column.title())
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_geometry(result: StudyResult, path: Path) -> None:
    """Show task accuracy beside the constraint or invariance diagnostic."""

    tasks = list(result.curves["task"].drop_duplicates())
    figure, axes = plt.subplots(
        len(tasks), 2, figsize=(13.0, 3.4 * len(tasks)), squeeze=False
    )
    for row, task in enumerate(tasks):
        diagnostics = (
            ("tracking_error", "Task-aligned tracking error"),
            (
                ("frame_error", "Frame error (same Grassmann subspace)")
                if task == "grassmann_basis_rotation"
                else ("feasibility_residual", "Feasibility residual")
            ),
        )
        for column, (metric, ylabel) in enumerate(diagnostics):
            axis = axes[row, column]
            data = result.curves[
                (result.curves["task"] == task) & (result.curves["metric"] == metric)
            ]
            for method, line in data.groupby("method"):
                line = line.sort_values("step")
                mean = np.maximum(line["mean"].to_numpy(dtype=float), 1e-16)
                low = np.maximum(line["ci_low"].to_numpy(dtype=float), 1e-16)
                high = np.maximum(line["ci_high"].to_numpy(dtype=float), 1e-16)
                axis.plot(line["step"], mean, label=str(method))
                axis.fill_between(line["step"], low, high, alpha=0.15)
            axis.set_title(f"{task}: {metric}")
            axis.set_xlabel("Step")
            axis.set_ylabel(ylabel)
            axis.set_yscale("log")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.suptitle("Geometry-aware tracking and feasibility")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_zeroth_freshness(result: StudyResult, path: Path) -> None:
    summary = result.extras.get("freshness_summary", pd.DataFrame())
    if summary.empty:
        return
    cases = list(summary["case_id"].drop_duplicates())
    methods = list(summary["method"].drop_duplicates())
    figure, axis = plt.subplots(figsize=(max(8.0, 1.7 * len(cases)), 5.4))
    centers = np.arange(len(cases), dtype=float)
    width = 0.8 / max(len(methods), 1)
    for index, method in enumerate(methods):
        line = summary[summary["method"] == method].set_index("case_id")
        positions = centers + (index - 0.5 * (len(methods) - 1)) * width
        means = np.array(
            [
                (
                    line.loc[case, "freshness_log_ratio_mean"]
                    if case in line.index
                    else np.nan
                )
                for case in cases
            ],
            dtype=float,
        )
        lows = np.array(
            [
                (
                    line.loc[case, "freshness_log_ratio_ci_low"]
                    if case in line.index
                    else np.nan
                )
                for case in cases
            ],
            dtype=float,
        )
        highs = np.array(
            [
                (
                    line.loc[case, "freshness_log_ratio_ci_high"]
                    if case in line.index
                    else np.nan
                )
                for case in cases
            ],
            dtype=float,
        )
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            capsize=2,
            label=str(method),
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(centers, cases, rotation=20, ha="right")
    axis.set_ylabel(r"$\log(\mathrm{MSE}_{streaming}/\mathrm{MSE}_{frozen})$")
    axis.set_title("Paired information-freshness effect")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_dimension(result: StudyResult, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    for column, path_kind in enumerate(("stationary", "linear")):
        data = result.summary[result.summary["path_kind"] == path_kind]
        for method, line in data.groupby("method"):
            line = line.sort_values("dim")
            mean = line["query_averaged_mse_mean"].to_numpy(dtype=float)
            low = line["query_averaged_mse_ci_low"].to_numpy(dtype=float)
            high = line["query_averaged_mse_ci_high"].to_numpy(dtype=float)
            axes[0, column].plot(
                line["dim"],
                mean,
                marker="o",
                label=str(method),
            )
            axes[0, column].fill_between(
                line["dim"],
                np.maximum(low, 1e-16),
                np.maximum(high, 1e-16),
                alpha=0.12,
            )
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("Query-averaged MSE")
        axes[0, column].set_title(f"{path_kind.title()} target")

    stationary = result.summary[result.summary["path_kind"] == "stationary"]
    for method, line in stationary.groupby("method"):
        line = line.sort_values("dim")
        axes[1, 0].plot(
            line["dim"],
            line["mean_gradient_cosine_mean"],
            marker="o",
            label=str(method),
        )
        axes[1, 1].plot(
            line["dim"],
            line["updates_mean"],
            marker="o",
            label=str(method),
        )
    axes[1, 0].set_ylabel("Gradient-estimate cosine")
    axes[1, 0].set_title("Estimator alignment")
    axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Updates within fixed query budget")
    axes[1, 1].set_title("Update opportunities at equal query budget")

    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("Ambient dimension")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Zero-order optimization as dimension grows")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_study(
    name: str, result: StudyResult, output_directory: Path, make_plots: bool
) -> Dict[str, object]:
    study_directory = output_directory / name
    study_directory.mkdir(parents=True, exist_ok=False)
    result.runs.to_csv(study_directory / "runs.csv", index=False)
    result.summary.to_csv(study_directory / "summary.csv", index=False)
    result.curves.to_csv(study_directory / "curves.csv", index=False)
    np.savez_compressed(
        study_directory / "histories.npz", **result.histories.get("arrays", {})
    )
    pd.DataFrame(result.histories.get("index", [])).to_csv(
        study_directory / "history_index.csv", index=False
    )
    for extra_name, frame in result.extras.items():
        frame.to_csv(study_directory / f"{extra_name}.csv", index=False)
        if extra_name in {"tuning", "selected_hyperparameters"}:
            (study_directory / f"{extra_name}.json").write_text(
                json.dumps(
                    json_safe(frame.to_dict(orient="records")),
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

    if make_plots:
        if name == "calibration":
            _plot_calibration(
                result, study_directory / "calibration_exact_vs_empirical.png"
            )
        elif name == "memory":
            _plot_panel_curves(
                result,
                panel_column="scenario",
                line_column="method",
                axis_column="step",
                metric_filter="squared_error",
                title="Memory--adaptation trade-off",
                ylabel="Squared tracking error",
                path=study_directory / "memory_tracking.png",
            )
        elif name == "zeroth":
            dynamic_cases = result.runs[result.runs["path_kind"] != "stationary"][
                "case_id"
            ]
            reference_case = (
                "cyclic_fast_d20"
                if "cyclic_fast_d20" in set(dynamic_cases)
                else str(dynamic_cases.iloc[0])
            )
            filtered = StudyResult(
                result.runs,
                result.summary,
                result.curves[result.curves["case_id"] == reference_case],
                result.extras,
            )
            _plot_panel_curves(
                filtered,
                panel_column="protocol",
                line_column="method",
                axis_column="queries",
                metric_filter="squared_error",
                title=f"Oracle freshness at equal query budget ({reference_case})",
                ylabel="Squared tracking error",
                path=study_directory / "zeroth_query_efficiency.png",
            )
            _plot_zeroth_freshness(result, study_directory / "zeroth_freshness_gap.png")
        elif name == "geometry":
            _plot_geometry(result, study_directory / "geometry_tracking.png")
        elif name == "dimension":
            _plot_dimension(result, study_directory / "dimension_scaling.png")

    return {
        "run_rows": int(len(result.runs)),
        "summary_rows": int(len(result.summary)),
        "curve_rows": int(len(result.curves)),
        "extras": {key: int(len(value)) for key, value in result.extras.items()},
        "directory": str(study_directory),
    }


def _parse_studies(value: str) -> Tuple[str, ...]:
    if value.strip().lower() == "all":
        return STUDY_NAMES
    studies = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = sorted(set(studies) - set(STUDY_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown studies {unknown}; available: {list(STUDY_NAMES)}"
        )
    if not studies:
        raise argparse.ArgumentTypeError("At least one study must be selected")
    return studies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the publication-oriented WIND experiment suite."
    )
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument(
        "--studies",
        type=_parse_studies,
        default=STUDY_NAMES,
        help=(
            "Comma-separated subset of calibration,memory,zeroth,geometry,dimension "
            "or 'all'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_experiment"),
        help="Parent directory for timestamped experiment outputs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional deterministic result directory name; must not already exist.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation.")
    return parser


@_series
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    profile = ExperimentProfile.build(args.profile)
    studies = (
        args.studies
        if isinstance(args.studies, tuple)
        else _parse_studies(args.studies)
    )
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{profile.name}_{timestamp}"
    output_directory = args.output_dir / run_name
    output_directory.mkdir(parents=True, exist_ok=False)

    runners: Mapping[str, Callable[[ExperimentProfile], StudyResult]] = {
        "calibration": run_calibration_study,
        "memory": run_memory_study,
        "zeroth": run_zero_order_study,
        "geometry": run_geometry_study,
        "dimension": run_dimension_study,
    }
    manifest: Dict[str, object] = {
        "experiment": "WIND final systematic experiment",
        "execution_backend": "library_oracle_session_v1",
        "experiment_protocol": PAPER_PROTOCOL,
        "package_version": importlib.metadata.version("wind-benchmark"),
        "catalog_version": CATALOG_VERSION,
        "metrics_protocol": "metrics_v2",
        "privacy_protocol": "local_series_v2",
        "export_schema_version": 4,
        "rng_protocol": "named_streams_v2",
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "matplotlib")
        },
        "source_sha256": source_hashes(),
        "execution_protocol": {
            "feedback": "library FirstOrderOracle / ZeroOrderOracle; immutable Observation or scoped query-only OracleClient",
            "clock": "one environment transition per consumed oracle query",
            "frozen": "target locked throughout an update, then advances by its query cost",
            "streaming": "target advances after each query",
            "partial_batches": "consume actual remaining queries without a fictitious completed update; retain last recommendation",
            "zero_order_score": "mean recommended-iterate squared error at every integer q=0,...,B-1; final error at q=B; tail is final 20% of q=0,...,B-1; rendering checkpoints do not enter metrics or tuning",
            "zero_order_rng": "independent value-noise and direction streams; stable estimator identity, never display label; frozen/streaming tapes paired",
            "truth": "query capability closes after each run; evaluator records and all study exports remain private until the entire series releases",
            "isolation": "API boundary, not a Python security sandbox",
            "analytic_reference": "calibration mean/covariance recurrence remains an independent evaluator calculation",
        },
        "memory_optimizers": {
            name: asdict(OptimizerInfo.from_optimizer(factory()))
            for name, factory in _memory_methods().items()
        },
        "zero_order_optimizers": [
            {
                **asdict(method),
                "catalog_name": method.catalog_name,
                "implementation_id": spec.implementation_id,
                "implementation_version": spec.implementation_version,
                "rng_stream_id": method.rng_stream_id,
                "source": spec.source,
                "variant": spec.variant,
                "parameters": "per-study selected_hyperparameters.json; FO reference fixed lr=0.04",
            }
            for method in _zero_order_methods()
            for spec in (resolve_optimizer_spec(method.catalog_name),)
        ],
        "algorithm_protocol": "initialize/advance; common source-specified catalog",
        "protocol_version": PAPER_PROTOCOL,
        "generator_rng_id": PAPER_GENERATOR_ID,
        "tuning_protocol": TUNING_PROTOCOL,
        "gradient_diagnostics": "estimation point versus clean gradient at post-update query tick; RandomSearch not applicable",
        "additional_implementations": {
            "geometry_control": "catalog SGD plus explicitly declared polar projection; not a canonical baseline"
        },
        "created_utc": timestamp,
        "profile": asdict(profile),
        "selected_studies": list(studies),
        "scientific_scope": {
            "calibration": "exact finite-time MSE of constant-step SGD",
            "memory": "noise reduction versus stale optimizer state",
            "zeroth": (
                "zero-order variance reduction versus information freshness at "
                "equal oracle-query budgets"
            ),
            "geometry": "feasibility and task-aligned frame/subspace tracking",
            "dimension": (
                "zero-order estimation and dynamic tracking as ambient dimension "
                "grows at a fixed oracle-query budget"
            ),
        },
        "studies": {},
    }

    print(f"WIND final experiment: profile={profile.name}, studies={list(studies)}")
    print(f"Output: {output_directory}")
    total_start = time.perf_counter()
    completed = {}
    tuning_cache = _TuningCache()
    for name in studies:
        print(f"[{name}] running...")
        start = time.perf_counter()
        result = (
            runners[name](profile, _tuning_cache=tuning_cache)
            if name in {"zeroth", "dimension"}
            else runners[name](profile)
        )
        completed[name] = (result, time.perf_counter() - start)
        print(f"[{name}] computations complete in {completed[name][1]:.2f}s")

    manifest["runtime_seconds"] = time.perf_counter() - total_start
    _export_completed(completed, output_directory, manifest, not args.no_plots)
    print(f"Complete in {manifest['runtime_seconds']:.2f}s")
    return 0


@_publish
def _export_completed(completed, output_directory, manifest, make_plots):
    for name, (result, duration) in completed.items():
        metadata = _write_study(name, result, output_directory, make_plots=make_plots)
        metadata["runtime_seconds"] = duration
        manifest["studies"][name] = metadata
        print(f"[{name}] complete in {duration:.2f}s")
    (output_directory / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
