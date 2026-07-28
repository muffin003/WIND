"""Publication-oriented experiments for the WIND benchmark.

This module complements :mod:`wind_benchmark.experiment`.  The legacy experiment
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

    python -m wind_benchmark.expFinal --profile smoke
    python -m wind_benchmark.expFinal --profile paper --studies all

Every run writes run-level summaries, aggregated curves, confidence intervals,
and a machine-readable manifest below ``results/final_experiment``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    GrassmannLandscape,
    QuadraticLandscape,
    SimplexLandscape,
    StiefelLandscape,
)
from .manifold import (
    principal_angle_distance,
    project_to_stiefel,
    random_stiefel,
    retract,
    tangent_project,
)

STUDY_NAMES = ("calibration", "memory", "zeroth", "geometry", "dimension")


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


@dataclass(frozen=True)
class ZeroOrderMethodConfig:
    name: str
    family: str
    directions: int = 1

    def query_cost(self, dim: int) -> int:
        if self.family == "first_order":
            return 1
        if self.family == "spsa":
            return 2
        if self.family == "gaussian":
            return 2 * self.directions
        if self.family == "coordinate":
            return 2 * dim
        raise ValueError(f"Unknown zero-order family: {self.family}")


@dataclass(frozen=True)
class ZeroOrderHyperparameters:
    learning_rate: float
    smoothing: float
    tuning_score: float = float("nan")


@dataclass
class StudyResult:
    runs: pd.DataFrame
    summary: pd.DataFrame
    curves: pd.DataFrame
    extras: Dict[str, pd.DataFrame]


class FirstOrderMethod(Protocol):
    name: str

    def reset(self, x0: np.ndarray) -> None: ...

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray: ...


class SGDMethod:
    name = "SGD"

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def reset(self, x0: np.ndarray) -> None:
        del x0

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return x - self.learning_rate * gradient


class HeavyBallMethod:
    name = "HeavyBall"

    def __init__(self, learning_rate: float, beta: float):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity: Optional[np.ndarray] = None

    def reset(self, x0: np.ndarray) -> None:
        self.velocity = np.zeros_like(x0)

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        assert self.velocity is not None
        self.velocity = self.beta * self.velocity - self.learning_rate * gradient
        return x + self.velocity


class NesterovMethod:
    """Fixed-momentum Nesterov method with gradients queried at the look-ahead point.

    The public iterate ``x`` is the look-ahead point ``y_t``.  After observing
    ``grad f(y_t)``, the method forms ``z_{t+1}`` and returns ``y_{t+1}``.
    """

    name = "Nesterov"

    def __init__(self, learning_rate: float, beta: float):
        self.learning_rate = learning_rate
        self.beta = beta
        self.previous_core: Optional[np.ndarray] = None

    def reset(self, x0: np.ndarray) -> None:
        del x0
        self.previous_core = None

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        core = x - self.learning_rate * gradient
        if self.previous_core is None:
            next_x = core
        else:
            next_x = core + self.beta * (core - self.previous_core)
        self.previous_core = core.copy()
        return next_x


class AdamMethod:
    name = "Adam"

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first: Optional[np.ndarray] = None
        self.second: Optional[np.ndarray] = None
        self.step_number = 0

    def reset(self, x0: np.ndarray) -> None:
        self.first = np.zeros_like(x0)
        self.second = np.zeros_like(x0)
        self.step_number = 0

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        assert self.first is not None and self.second is not None
        self.step_number += 1
        self.first = self.beta1 * self.first + (1.0 - self.beta1) * gradient
        self.second = self.beta2 * self.second + (1.0 - self.beta2) * gradient**2
        first_hat = self.first / (1.0 - self.beta1**self.step_number)
        second_hat = self.second / (1.0 - self.beta2**self.step_number)
        return x - self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)


class AMSGradMethod(AdamMethod):
    name = "AMSGrad"

    def __init__(self, learning_rate: float, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.maximum_second: Optional[np.ndarray] = None

    def reset(self, x0: np.ndarray) -> None:
        super().reset(x0)
        self.maximum_second = np.zeros_like(x0)

    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        assert self.first is not None
        assert self.second is not None
        assert self.maximum_second is not None
        self.step_number += 1
        self.first = self.beta1 * self.first + (1.0 - self.beta1) * gradient
        self.second = self.beta2 * self.second + (1.0 - self.beta2) * gradient**2
        self.maximum_second = np.maximum(self.maximum_second, self.second)
        first_hat = self.first / (1.0 - self.beta1**self.step_number)
        return x - self.learning_rate * first_hat / (
            np.sqrt(self.maximum_second) + self.epsilon
        )


def _stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
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
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if clean.size == 1:
        value = float(clean[0])
        return value, value
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
    rng = np.random.default_rng(seed)
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in frame.groupby(grouper, dropna=False, sort=True):
        key_values = (key,) if len(group_columns) == 1 else tuple(key)
        row: Dict[str, object] = dict(zip(group_columns, key_values))
        row["runs"] = int(len(group))
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
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
            low, high = _bootstrap_mean_interval(finite, repetitions, rng)
            row[f"{metric}_mean"] = float(np.mean(finite))
            row[f"{metric}_std"] = float(
                np.std(finite, ddof=1 if len(finite) > 1 else 0)
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
        for index, axis_value in enumerate(axis):
            point_values = values[:, index]
            finite = point_values[np.isfinite(point_values)]
            if finite.size:
                mean = float(np.mean(finite))
                std = float(np.std(finite, ddof=1 if finite.size > 1 else 0))
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
            rng = np.random.default_rng(_stable_seed(seed, "calibration", case.case_id))
            x = x0.copy()
            theta = np.zeros(case.dim)
            exact_mean = x0.copy()
            exact_covariance = np.zeros((case.dim, case.dim))
            empirical_mse = np.empty(profile.calibration_steps)
            exact_mse = np.empty(profile.calibration_steps)

            for step in range(profile.calibration_steps):
                error = x - theta
                empirical_mse[step] = float(error @ error)
                exact_mse[step] = float(
                    exact_mean @ exact_mean + np.trace(exact_covariance)
                )

                noise = rng.normal(0.0, case.gradient_noise, size=case.dim)
                gradient = hessian @ error + noise
                x = x - case.learning_rate * gradient
                theta = theta + velocity

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


def _memory_methods() -> Mapping[str, Callable[[], FirstOrderMethod]]:
    return {
        "SGD": lambda: SGDMethod(learning_rate=0.05),
        "HeavyBall": lambda: HeavyBallMethod(learning_rate=0.05, beta=0.9),
        "Nesterov": lambda: NesterovMethod(learning_rate=0.05, beta=0.9),
        "Adam": lambda: AdamMethod(learning_rate=0.02),
        "AMSGrad": lambda: AMSGradMethod(learning_rate=0.02),
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
    return float(best_lag)


def _time_to_recovery(
    errors: np.ndarray, theta_path: np.ndarray, scenario: MemoryScenario, dim: int
) -> float:
    if scenario.kind != "jump":
        return float("nan")
    jumps = np.flatnonzero(np.linalg.norm(np.diff(theta_path, axis=0), axis=1) > 0) + 1
    if jumps.size == 0:
        return float("nan")
    recoveries: List[int] = []
    noise_floor = 3.0 * scenario.gradient_noise * math.sqrt(dim)
    for index, jump in enumerate(jumps):
        next_jump = int(jumps[index + 1]) if index + 1 < len(jumps) else len(errors)
        jump_size = float(np.linalg.norm(theta_path[jump] - theta_path[jump - 1]))
        threshold = max(0.2 * jump_size, noise_floor)
        candidates = np.flatnonzero(errors[jump:next_jump] <= threshold)
        recoveries.append(int(candidates[0]) if candidates.size else next_jump - jump)
    return float(np.median(recoveries))


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
    curves: CurveStore = {}

    for scenario in scenarios:
        theta_path = _target_path(scenario, profile.memory_steps, direction)
        target_projection = theta_path @ direction
        for seed in profile.seeds:
            noise_rng = np.random.default_rng(
                _stable_seed(seed, "memory", scenario.name)
            )
            common_noise = noise_rng.normal(
                0.0,
                scenario.gradient_noise,
                size=(profile.memory_steps, dim),
            )
            for method_name, factory in methods.items():
                method = factory()
                x = np.zeros(dim)
                method.reset(x)
                squared_error = np.empty(profile.memory_steps)
                regret = np.empty(profile.memory_steps)
                estimate_projection = np.empty(profile.memory_steps)

                for step in range(profile.memory_steps):
                    theta = theta_path[step]
                    error = x - theta
                    squared_error[step] = float(error @ error)
                    regret[step] = 0.5 * float(error @ hessian @ error)
                    estimate_projection[step] = float(x @ direction)
                    gradient = hessian @ error + common_noise[step]
                    x = method.step(x, gradient)

                error_norm = np.sqrt(squared_error)
                run_rows.append(
                    {
                        **asdict(scenario),
                        "method": method_name,
                        "seed": seed,
                        "dim": dim,
                        "condition_number": condition_number,
                        "tail_mse": _tail_mean(squared_error),
                        "tail_regret": _tail_mean(regret),
                        "tail_p90_error": float(
                            np.quantile(error_norm[int(0.8 * len(error_norm)) :], 0.9)
                        ),
                        "phase_lag_steps": _phase_lag(
                            estimate_projection, target_projection, scenario.period
                        ),
                        "time_to_recovery": _time_to_recovery(
                            error_norm, theta_path, scenario, dim
                        ),
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
    return StudyResult(runs, summary, curve_frame, extras={})


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
        ZeroOrderMethodConfig("SPSA-m1", "spsa"),
        ZeroOrderMethodConfig("Gaussian-m1", "gaussian", directions=1),
        ZeroOrderMethodConfig("Gaussian-m5", "gaussian", directions=5),
        ZeroOrderMethodConfig("Gaussian-m20", "gaussian", directions=20),
        ZeroOrderMethodConfig("CoordinateCentral", "coordinate"),
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


def _run_zero_order_method(
    method: ZeroOrderMethodConfig,
    protocol: str,
    case: ZeroOrderCase,
    budget: int,
    checkpoints: np.ndarray,
    seed: int,
    hyperparameters: ZeroOrderHyperparameters,
) -> Tuple[np.ndarray, Dict[str, float]]:
    landscape = QuadraticLandscape(
        case.dim,
        case.condition_number,
        seed=_stable_seed(41, "zero-landscape", case.dim, case.condition_number),
    )
    hessian = landscape.A
    direction = _unit_direction(case.dim, _stable_seed(43, "zero-direction", case.dim))
    theta_at = lambda query: _zero_order_theta_at(case, direction, query)
    # Excluding protocol deliberately pairs frozen and streaming runs: they receive
    # the same perturbations and observation-noise tape.
    rng = np.random.default_rng(_stable_seed(seed, "zero", case.case_id, method.name))
    x = _unit_direction(case.dim, _stable_seed(47, "zero-initial", case.dim))
    query = 0
    query_history = [0]
    state_history = [x.copy()]
    updates = 0
    failed = False
    failure_query = budget + 1
    maximum_gradient_norm = 0.0
    gradient_cosines: List[float] = []
    learning_rate = hyperparameters.learning_rate
    smoothing = hyperparameters.smoothing
    cost = method.query_cost(case.dim)

    def value(point: np.ndarray, theta: np.ndarray) -> float:
        return landscape.loss(point, theta) + float(
            rng.normal(0.0, case.observation_noise)
        )

    while query + cost <= budget:
        snapshot = theta_at(query)

        def query_theta(current_query: int) -> np.ndarray:
            return snapshot if protocol == "frozen" else theta_at(current_query)

        if method.family == "first_order":
            gradient = hessian @ (x - query_theta(query))
            query += 1
        elif method.family == "spsa":
            perturbation = rng.choice([-1.0, 1.0], size=case.dim)
            plus = value(x + smoothing * perturbation, query_theta(query))
            query += 1
            minus = value(x - smoothing * perturbation, query_theta(query))
            query += 1
            gradient = (plus - minus) * perturbation / (2.0 * smoothing)
        elif method.family == "gaussian":
            gradient = np.zeros(case.dim)
            for _ in range(method.directions):
                perturbation = rng.normal(size=case.dim)
                perturbation /= np.linalg.norm(perturbation)
                plus = value(x + smoothing * perturbation, query_theta(query))
                query += 1
                minus = value(x - smoothing * perturbation, query_theta(query))
                query += 1
                gradient += case.dim * (plus - minus) * perturbation / (2.0 * smoothing)
            gradient /= method.directions
        elif method.family == "coordinate":
            gradient = np.zeros(case.dim)
            for coordinate in range(case.dim):
                basis = np.zeros(case.dim)
                basis[coordinate] = smoothing
                plus = value(x + basis, query_theta(query))
                query += 1
                minus = value(x - basis, query_theta(query))
                query += 1
                gradient[coordinate] = (plus - minus) / (2.0 * smoothing)
        else:
            raise ValueError(method.family)

        gradient_norm = float(np.linalg.norm(gradient))
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        if not np.isfinite(gradient_norm) or gradient_norm > 1e6:
            failed = True
            failure_query = query
            query_history.append(query)
            state_history.append(x.copy())
            break

        current_gradient = hessian @ (x - theta_at(query))
        denominator = gradient_norm * float(np.linalg.norm(current_gradient))
        if denominator > 0.0:
            gradient_cosines.append(float(gradient @ current_gradient / denominator))
        x = x - learning_rate * gradient
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
            failed = True
            failure_query = query
            query_history.append(query)
            state_history.append(x.copy())
            break
        updates += 1
        query_history.append(query)
        state_history.append(x.copy())

    sampled_error = _sample_query_curve(
        query_history, state_history, checkpoints, theta_at
    )
    if failed:
        sampled_error[checkpoints >= failure_query] = np.nan
    finite_curve = bool(np.all(np.isfinite(sampled_error)))
    mean_error = (
        float(np.trapezoid(sampled_error, checkpoints) / max(budget, 1))
        if finite_curve
        else float("nan")
    )
    threshold_hits = np.flatnonzero(
        (checkpoints > 0) & np.isfinite(sampled_error) & (sampled_error <= 1e-2)
    )
    metrics = {
        "final_mse": float(sampled_error[-1]) if finite_curve else float("nan"),
        "query_averaged_mse": mean_error,
        "tail_mse": _tail_mean(sampled_error) if finite_curve else float("nan"),
        "queries_to_threshold": (
            float(checkpoints[threshold_hits[0]])
            if threshold_hits.size
            else float("nan")
        ),
        "mean_gradient_cosine": (
            float(np.mean(gradient_cosines)) if gradient_cosines else float("nan")
        ),
        "updates": float(updates),
        "queries_per_update": float(cost),
        "learning_rate": learning_rate,
        "smoothing": smoothing,
        "directions": float(method.directions),
        "maximum_gradient_norm": maximum_gradient_norm,
        "failed": float(failed),
    }
    return sampled_error, metrics


def _zero_order_candidate_hyperparameters(
    profile: ExperimentProfile, method: ZeroOrderMethodConfig, dim: int
) -> List[Tuple[float, float, float]]:
    if method.family == "first_order":
        return [(0.04, 0.0, float("nan"))]
    if method.family == "coordinate":
        return [(0.04, 0.1, float("nan"))]
    constants = (0.1, 0.2) if profile.name == "smoke" else (0.05, 0.1, 0.2, 0.4)
    smoothings = (0.1,) if profile.name == "smoke" else (0.03, 0.1, 0.3)
    effective_directions = 1 if method.family == "spsa" else method.directions
    dimension_scale = min(1.0, effective_directions / dim)
    return [
        (constant * dimension_scale, smoothing, constant)
        for constant in constants
        for smoothing in smoothings
    ]


def _tune_zero_order_methods(
    profile: ExperimentProfile,
    cases: Sequence[ZeroOrderCase],
    methods: Sequence[ZeroOrderMethodConfig],
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

    for dim in dimensions:
        tuning_case = ZeroOrderCase(f"tuning_stationary_d{dim}", dim, 5.0, "stationary")
        for method in methods:
            candidate_rows: List[Dict[str, object]] = []
            for (
                learning_rate,
                smoothing,
                constant,
            ) in _zero_order_candidate_hyperparameters(profile, method, dim):
                parameters = ZeroOrderHyperparameters(learning_rate, smoothing)
                scores: List[float] = []
                failures = 0
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
                    if np.isfinite(metrics["query_averaged_mse"]):
                        scores.append(float(metrics["query_averaged_mse"]))
                score = (
                    float(np.mean(scores)) if scores and failures == 0 else float("inf")
                )
                candidate_rows.append(
                    {
                        "dim": dim,
                        "method": method.name,
                        "family": method.family,
                        "directions": method.directions,
                        "learning_rate": learning_rate,
                        "smoothing": smoothing,
                        "dimension_constant": constant,
                        "tuning_score": score,
                        "failure_rate": failures / len(tuning_seeds),
                        "selected": False,
                    }
                )
            best_index = min(
                range(len(candidate_rows)),
                key=lambda index: candidate_rows[index]["tuning_score"],
            )
            candidate_rows[best_index]["selected"] = True
            best = candidate_rows[best_index]
            selected[(dim, method.name)] = ZeroOrderHyperparameters(
                float(best["learning_rate"]),
                float(best["smoothing"]),
                float(best["tuning_score"]),
            )
            rows.extend(candidate_rows)
    return selected, pd.DataFrame(rows)


def run_zero_order_study(profile: ExperimentProfile) -> StudyResult:
    """Measure the variance--freshness trade-off at a fixed dimension."""

    cases = _zero_order_cases(profile)
    methods = _zero_order_methods()
    selected_hyperparameters, tuning = _tune_zero_order_methods(profile, cases, methods)
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
                    error_curve, metrics = _run_zero_order_method(
                        method,
                        protocol,
                        case,
                        profile.zero_order_budget,
                        checkpoints,
                        seed,
                        selected_hyperparameters[(case.dim, method.name)],
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
                    ).append((checkpoints, error_curve))

    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(
        runs,
        group_columns=("case_id", "protocol", "method"),
        metric_columns=(
            "final_mse",
            "query_averaged_mse",
            "tail_mse",
            "queries_to_threshold",
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


def _update_simplex(
    method: str, x: np.ndarray, gradient: np.ndarray, learning_rate: float
) -> np.ndarray:
    if method == "ProjectedSGD":
        return SimplexLandscape.project(x - learning_rate * gradient)
    if method == "ExponentiatedGradient":
        logits = np.log(np.maximum(x, 1e-15)) - learning_rate * gradient
        logits -= np.max(logits)
        weights = np.exp(logits)
        return weights / np.sum(weights)
    if method == "EuclideanSGD":
        return x - learning_rate * gradient
    raise ValueError(method)


def _update_manifold(
    method: str, x: np.ndarray, gradient: np.ndarray, learning_rate: float
) -> np.ndarray:
    if method == "RiemannianSGD":
        tangent = tangent_project(x, gradient)
        return retract(x, -learning_rate * tangent)
    if method == "ProjectedEuclideanSGD":
        return project_to_stiefel(x - learning_rate * gradient)
    if method == "EuclideanSGD":
        return x - learning_rate * gradient
    raise ValueError(method)


def _run_simplex_geometry(
    profile: ExperimentProfile, seed: int, curves: CurveStore
) -> List[Dict[str, object]]:
    dim = 10
    period = 120 if profile.name == "paper" else 60
    theta_path = _simplex_path(
        profile.geometry_steps,
        dim,
        _stable_seed(seed, "simplex-target"),
        period,
    )
    landscape = SimplexLandscape()
    noise_rng = np.random.default_rng(_stable_seed(seed, "simplex-noise"))
    common_noise = noise_rng.normal(0.0, 0.01, size=(profile.geometry_steps, dim))
    rows: List[Dict[str, object]] = []
    methods = ("ProjectedSGD", "ExponentiatedGradient", "EuclideanSGD")

    for method in methods:
        x = np.full(dim, 1.0 / dim)
        loss = np.empty(profile.geometry_steps)
        tracking = np.empty(profile.geometry_steps)
        feasibility = np.empty(profile.geometry_steps)
        for step in range(profile.geometry_steps):
            theta = theta_path[step]
            difference = x - theta
            loss[step] = landscape.loss(x, theta)
            tracking[step] = float(np.linalg.norm(difference))
            feasibility[step] = _simplex_feasibility(x)
            gradient = landscape.grad(x, theta) + common_noise[step]
            x = _update_simplex(method, x, gradient, learning_rate=0.15)

        rows.append(
            {
                "task": "simplex_tracking",
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
    methods = ("RiemannianSGD", "ProjectedEuclideanSGD", "EuclideanSGD")
    rows: List[Dict[str, object]] = []

    for task in tasks:
        theta_path = _manifold_path(
            task,
            profile.geometry_steps,
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
        noise_rng = np.random.default_rng(_stable_seed(seed, "manifold-noise", task))
        common_noise = noise_rng.normal(
            0.0, noise_sigma, size=(profile.geometry_steps, d, r)
        )

        for method in methods:
            x = theta_path[0].copy()
            loss = np.empty(profile.geometry_steps)
            tracking = np.empty(profile.geometry_steps)
            feasibility = np.empty(profile.geometry_steps)
            frame_error = np.empty(profile.geometry_steps)
            subspace_error = np.empty(profile.geometry_steps)

            for step in range(profile.geometry_steps):
                theta = theta_path[step]
                projected_x = project_to_stiefel(x)
                flattened_x = x.reshape(-1)
                flattened_theta = theta.reshape(-1)
                loss[step] = landscape.loss(flattened_x, flattened_theta)
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
                gradient = landscape.grad(flattened_x, flattened_theta).reshape(d, r)
                gradient = gradient + common_noise[step]
                x = _update_manifold(method, x, gradient, learning_rate=0.12)

            rows.append(
                {
                    "task": task,
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
    return StudyResult(runs, summary, curve_frame, extras={})


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


def run_dimension_study(profile: ExperimentProfile) -> StudyResult:
    """Measure zero-order estimation and tracking as dimension grows."""

    cases = _zero_order_dimension_cases(profile)
    methods = _zero_order_methods()
    selected_hyperparameters, tuning = _tune_zero_order_methods(profile, cases, methods)
    checkpoints = np.unique(
        np.linspace(0, profile.zero_order_budget, profile.curve_points, dtype=int)
    ).astype(float)
    run_rows: List[Dict[str, object]] = []
    curves: CurveStore = {}

    for case in cases:
        for method in methods:
            parameters = selected_hyperparameters[(case.dim, method.name)]
            for seed in profile.seeds:
                error_curve, metrics = _run_zero_order_method(
                    method,
                    "frozen",
                    case,
                    profile.zero_order_budget,
                    checkpoints,
                    seed,
                    parameters,
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
                ).append((checkpoints, error_curve))

    runs = pd.DataFrame(run_rows)
    summary = _summarize_runs(
        runs,
        group_columns=("path_kind", "dim", "method"),
        metric_columns=(
            "final_mse",
            "query_averaged_mse",
            "tail_mse",
            "queries_to_threshold",
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
    for extra_name, frame in result.extras.items():
        frame.to_csv(study_directory / f"{extra_name}.csv", index=False)

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
    for name in studies:
        print(f"[{name}] running...")
        start = time.perf_counter()
        result = runners[name](profile)
        metadata = _write_study(
            name, result, output_directory, make_plots=not args.no_plots
        )
        metadata["runtime_seconds"] = time.perf_counter() - start
        manifest["studies"][name] = metadata
        print(f"[{name}] complete in {metadata['runtime_seconds']:.2f}s")

    manifest["runtime_seconds"] = time.perf_counter() - total_start
    with (output_directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=_json_default)
    print(f"Complete in {manifest['runtime_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
