"""Execute a general WIND workbench configuration in a child process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .benchmark import BenchmarkRunner
from .core import (
    GrassmannLandscape,
    SimplexLandscape,
    StiefelLandscape,
    make_environment,
    make_noise,
)
from .experiment import (
    AMSGrad,
    Adam,
    AdamW,
    AcceleratedSPSA,
    AdaptiveLR,
    CMAES,
    FDSA,
    FiniteDiffCentral,
    GPUCB,
    HeavyBall,
    KieferWolfowitz,
    NedicSubgradient,
    Nesterov,
    OnePointSPSA,
    ProxSGD,
    QuadraticInterpolation,
    RDA,
    RandomSearch,
    SGD,
    SGDPolyak,
    SMD,
    SPSA,
    SignSGD,
    ZOSGD,
    ZOSignSGD,
)
from .metrics import (
    AdaptivityMetric,
    AsymptoticBoundMetric,
    DriftAdaptationMetric,
    DynamicRegretMetric,
    GrassmannPrincipalAngleMetric,
    InstantaneousLossMetric,
    LyapunovMetric,
    MaxCoordinateErrorMetric,
    MetricsCollection,
    QueryEfficiencyMetric,
    StiefelFrameMetric,
    TimeToRecoveryMetric,
    TrackingErrorMetric,
)
from .oracle import (
    FirstOrderOracle,
    HybridOracle,
    OfflineOracle,
    ScheduledOracle,
    ZeroOrderOracle,
)

LANDSCAPES = {
    "quadratic",
    "pnorm",
    "rosenbrock",
    "multiextremal",
    "robust",
    "simplex",
    "stiefel",
    "grassmann",
}
DRIFTS = {
    "stationary",
    "linear",
    "random_walk",
    "cyclic",
    "jump",
    "adaptive",
    "sparse",
    "stiefel",
}
NOISES = {
    "none",
    "gaussian",
    "heavy_tailed",
    "correlated",
    "quantized",
    "multiplicative",
    "sparse",
}
ORACLES = {"first-order", "zero-order", "hybrid", "scheduled", "offline"}

OPTIMIZERS = {
    "SGD": (SGD, "first-order"),
    "SGD_Polyak": (SGDPolyak, "first-order"),
    "HeavyBall": (HeavyBall, "first-order"),
    "Nesterov": (Nesterov, "first-order"),
    "Adam": (Adam, "first-order"),
    "AdamW": (AdamW, "first-order"),
    "AMSGrad": (AMSGrad, "first-order"),
    "SMD": (SMD, "first-order"),
    "RDA": (RDA, "first-order"),
    "ProxSGD": (ProxSGD, "first-order"),
    "AdaptiveLR": (AdaptiveLR, "first-order"),
    "SignSGD": (SignSGD, "first-order"),
    "RandomSearch": (RandomSearch, "zero-order"),
    "OnePointSPSA": (OnePointSPSA, "zero-order"),
    "FiniteDiffCentral": (FiniteDiffCentral, "zero-order"),
    "FDSA": (FDSA, "zero-order"),
    "SPSA": (SPSA, "zero-order"),
    "ZOSGD": (ZOSGD, "zero-order"),
    "ZOSignSGD": (ZOSignSGD, "zero-order"),
    "QuadraticInterpolation": (QuadraticInterpolation, "zero-order"),
    "KieferWolfowitz": (KieferWolfowitz, "zero-order"),
    "NedicSubgradient": (NedicSubgradient, "zero-order"),
    "AcceleratedSPSA": (AcceleratedSPSA, "zero-order"),
    "CMAES": (CMAES, "zero-order"),
    "GPUCB": (GPUCB, "zero-order"),
}

METRICS = {
    "tracking_error",
    "max_coordinate_error",
    "instant_loss",
    "dynamic_regret",
    "time_to_recovery",
    "drift_adaptation",
    "adaptivity",
    "query_efficiency",
    "lyapunov",
    "asymptotic_bound",
}


class WorkbenchConfigurationError(ValueError):
    """Raised when a workbench configuration cannot be executed."""


def optimizer_configs_from_payload(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return the multi-select schema while accepting the legacy single item."""
    if "optimizers" in payload:
        configs = payload["optimizers"]
        if not isinstance(configs, list):
            raise WorkbenchConfigurationError("optimizers must be a list")
        return configs
    optimizer = payload.get("optimizer")
    if optimizer is None:
        return []
    if not isinstance(optimizer, dict):
        raise WorkbenchConfigurationError("optimizer must be an object")
    return [optimizer]


def validate_workbench_config(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise WorkbenchConfigurationError("Workbench configuration must be an object")

    for section in ("environment", "oracle", "runner"):
        if not isinstance(payload.get(section), dict):
            raise WorkbenchConfigurationError(
                f"Missing configuration section: {section}"
            )

    environment = payload["environment"]
    dimension = environment.get("dim")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise WorkbenchConfigurationError("environment.dim must be a positive integer")

    landscape = environment.get("landscape", {})
    if landscape.get("type") not in LANDSCAPES:
        raise WorkbenchConfigurationError("Unknown landscape type")
    drift = environment.get("drift", {})
    if drift.get("type") not in DRIFTS:
        raise WorkbenchConfigurationError("Unknown drift type")
    initial_theta = environment.get("initial_theta")
    if initial_theta is not None and (
        not isinstance(initial_theta, list)
        or len(initial_theta) != dimension
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in initial_theta
        )
    ):
        raise WorkbenchConfigurationError(
            "environment.initial_theta must contain environment.dim numbers"
        )

    oracle = payload["oracle"]
    if oracle.get("type") not in ORACLES:
        raise WorkbenchConfigurationError("Unknown oracle type")
    for noise_key in ("value_noise", "grad_noise"):
        noise = oracle.get(noise_key, {"type": "none"})
        if not isinstance(noise, dict) or noise.get("type", "none") not in NOISES:
            raise WorkbenchConfigurationError(f"Unknown {noise_key} type")

    optimizer_configs = optimizer_configs_from_payload(payload)
    if not all(isinstance(optimizer, dict) for optimizer in optimizer_configs):
        raise WorkbenchConfigurationError("Each optimizer must be an object")
    names = [optimizer.get("name") for optimizer in optimizer_configs]
    if any(name not in OPTIMIZERS for name in names):
        raise WorkbenchConfigurationError("Unknown optimizer")
    if len(names) != len(set(names)):
        raise WorkbenchConfigurationError("Optimizer selection contains duplicates")
    if any(
        not isinstance(optimizer.get("params", {}), dict)
        for optimizer in optimizer_configs
    ):
        raise WorkbenchConfigurationError("optimizer.params must be an object")

    optimizer_orders = [OPTIMIZERS[name][1] for name in names]
    oracle_type = oracle["type"]
    if "first-order" in optimizer_orders and oracle_type == "zero-order":
        raise WorkbenchConfigurationError(
            "First-order optimizer requires gradient observations"
        )
    if (
        "zero-order" in optimizer_orders
        and oracle_type == "first-order"
        and bool(oracle.get("blind_value", True))
    ):
        raise WorkbenchConfigurationError(
            "Zero-order optimizer requires function-value observations"
        )
    if oracle_type == "scheduled":
        schedule = oracle.get("schedule")
        if not isinstance(schedule, list) or not schedule:
            raise WorkbenchConfigurationError(
                "Scheduled oracle requires a non-empty schedule"
            )
        valid_schedule = all(
            isinstance(segment, (list, tuple))
            and len(segment) == 2
            and segment[0] in {"first-order", "zero-order"}
            and isinstance(segment[1], int)
            and not isinstance(segment[1], bool)
            and segment[1] > 0
            for segment in schedule
        )
        if not valid_schedule:
            raise WorkbenchConfigurationError(
                "Schedule entries must be [mode, positive duration] pairs"
            )
        if "first-order" in optimizer_orders and any(
            mode == "zero-order" for mode, _ in schedule
        ):
            raise WorkbenchConfigurationError(
                "First-order optimizer cannot run during zero-order schedule phases"
            )

    runner = payload["runner"]
    steps = runner.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise WorkbenchConfigurationError("runner.steps must be a positive integer")
    seeds = runner.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise WorkbenchConfigurationError("runner.seeds must be a non-empty list")
    if not all(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
        for seed in seeds
    ):
        raise WorkbenchConfigurationError(
            "runner.seeds must contain non-negative integers"
        )
    tail_fraction = runner.get("tail_fraction", 0.2)
    if not isinstance(tail_fraction, (int, float)) or not 0 < tail_fraction <= 1:
        raise WorkbenchConfigurationError("runner.tail_fraction must be in (0, 1]")
    if runner.get("tracking_norm", "l2") not in {"l1", "l2", "linf", "mahalanobis"}:
        raise WorkbenchConfigurationError("runner.tracking_norm is invalid")
    for field in ("jump_threshold", "recovery_epsilon", "oracle_ttr"):
        value = runner.get(field, 1.0 if field != "recovery_epsilon" else 0.1)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise WorkbenchConfigurationError(f"runner.{field} must be positive")
    rho = runner.get("rho", 1.0)
    if not isinstance(rho, (int, float)) or isinstance(rho, bool) or not 0 < rho <= 1:
        raise WorkbenchConfigurationError("runner.rho must be in (0, 1]")
    if (
        not isinstance(runner.get("output_dir"), str)
        or not runner["output_dir"].strip()
    ):
        raise WorkbenchConfigurationError(
            "runner.output_dir must be a non-empty string"
        )
    if runner.get("export_csv") and not runner.get("record_trajectory", True):
        raise WorkbenchConfigurationError("CSV export requires trajectory recording")
    initial_x = runner.get("initial_x")
    if initial_x is not None and (
        not isinstance(initial_x, list)
        or len(initial_x) != dimension
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in initial_x
        )
    ):
        raise WorkbenchConfigurationError(
            "runner.initial_x must contain environment.dim numbers"
        )

    metrics = payload.get("metrics", ["tracking_error"])
    if not isinstance(metrics, list) or not metrics or set(metrics) - METRICS:
        raise WorkbenchConfigurationError("Unknown or empty metric selection")

    return payload


def _make_noise(config: Dict[str, Any], seed: int, dim: int):
    noise_type = config.get("type", "none")
    if noise_type == "none":
        return None
    params = {key: value for key, value in config.items() if key != "type"}
    if noise_type in {
        "gaussian",
        "heavy_tailed",
        "correlated",
        "multiplicative",
        "sparse",
    }:
        params.setdefault("seed", seed)
    if noise_type == "correlated":
        params.setdefault("dim", dim)
    return make_noise(noise_type, **params)


def _make_oracle(config: Dict[str, Any], environment, seed: int, project_root: Path):
    oracle_type = config["type"]
    value_noise = _make_noise(
        config.get("value_noise", {"type": "none"}), seed + 11, environment.dim
    )
    grad_noise = _make_noise(
        config.get("grad_noise", {"type": "none"}), seed + 23, environment.dim
    )
    common = {
        "environment": environment,
        "value_noise": value_noise,
        "grad_noise": grad_noise,
        "seed": seed,
    }
    if oracle_type == "first-order":
        return FirstOrderOracle(
            **common, blind_value=bool(config.get("blind_value", True))
        )
    if oracle_type == "zero-order":
        return ZeroOrderOracle(**common)
    if oracle_type == "hybrid":
        return HybridOracle(**common)
    if oracle_type == "scheduled":
        raw_schedule = config.get(
            "schedule", [["first-order", 100], ["zero-order", 50]]
        )
        schedule = [(str(mode), int(duration)) for mode, duration in raw_schedule]
        return ScheduledOracle(**common, schedule=schedule)

    recorded_path = (project_root / config.get("recorded_path", "")).resolve()
    try:
        recorded_path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise WorkbenchConfigurationError(
            "Offline replay path must stay inside the project"
        ) from exc
    data = json.loads(recorded_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("trajectory", {}).get("theta", data.get("recorded_thetas"))
    if not isinstance(data, list) or not data:
        raise WorkbenchConfigurationError(
            "Offline replay file does not contain theta trajectory"
        )
    thetas = [np.asarray(theta, dtype=float) for theta in data]
    return OfflineOracle(
        **common, recorded_thetas=thetas, landscape=environment.landscape
    )


def _make_optimizer(config: Dict[str, Any]):
    name = config["name"]
    optimizer_class, oracle_type = OPTIMIZERS[name]
    params = dict(config.get("params", {}))
    if name == "CMAES" and not params.get("population_size"):
        params.pop("population_size", None)
    optimizer = optimizer_class(**params)
    optimizer.oracle_type = oracle_type
    return optimizer


def _make_metrics(names: list[str], environment, runner: Dict[str, Any]):
    metrics = []
    rho = float(runner.get("rho", 1.0))
    for name in names:
        if name == "tracking_error":
            if isinstance(environment.landscape, StiefelLandscape):
                metrics.append(
                    StiefelFrameMetric(environment.landscape.d, environment.landscape.r)
                )
            elif isinstance(environment.landscape, GrassmannLandscape):
                metrics.append(
                    GrassmannPrincipalAngleMetric(
                        environment.landscape.d, environment.landscape.r
                    )
                )
            else:
                metrics.append(
                    TrackingErrorMetric(
                        norm=runner.get("tracking_norm", "l2"),
                        normalize_by_dim=bool(runner.get("normalize_tracking", False)),
                    )
                )
        elif name == "max_coordinate_error":
            metrics.append(MaxCoordinateErrorMetric())
        elif name == "instant_loss":
            metrics.append(InstantaneousLossMetric())
        elif name == "dynamic_regret":
            metrics.append(
                DynamicRegretMetric(
                    normalize_by_path=bool(runner.get("normalize_regret"))
                )
            )
        elif name == "time_to_recovery":
            metrics.append(
                TimeToRecoveryMetric(
                    jump_threshold=float(runner.get("jump_threshold", 1.0)),
                    epsilon=float(runner.get("recovery_epsilon", 0.1)),
                )
            )
        elif name == "drift_adaptation":
            metrics.append(DriftAdaptationMetric())
        elif name == "adaptivity":
            metrics.append(
                AdaptivityMetric(
                    jump_threshold=float(runner.get("jump_threshold", 1.0)),
                    epsilon=float(runner.get("recovery_epsilon", 0.1)),
                    oracle_ttr=float(runner.get("oracle_ttr", 1.0)),
                )
            )
        elif name == "query_efficiency":
            metrics.append(QueryEfficiencyMetric())
        elif name == "lyapunov":
            metrics.append(LyapunovMetric(rho=rho))
        elif name == "asymptotic_bound":
            metrics.append(AsymptoticBoundMetric(rho=rho))
    return MetricsCollection(metrics)


def _initial_point(
    environment, seed: int, configured: Optional[Sequence[float]] = None
) -> np.ndarray:
    if configured is not None:
        point = np.asarray(configured, dtype=float)
        if isinstance(environment.landscape, SimplexLandscape):
            return environment.landscape.project(point)
        return point
    rng = np.random.default_rng(seed + 101)
    if isinstance(environment.landscape, SimplexLandscape):
        return rng.dirichlet(np.ones(environment.dim))
    if isinstance(environment.landscape, (StiefelLandscape, GrassmannLandscape)):
        return environment.landscape.random_point(
            environment.landscape.d, environment.landscape.r, seed=seed + 101
        ).reshape(-1)
    return rng.normal(0.0, 0.1, size=environment.dim)


def _write_result(result, json_path: Path, csv_path: Optional[Path]) -> None:
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if csv_path is not None:
        result.save_to_csv(str(csv_path))


def run_workbench(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    validate_workbench_config(config)
    optimizer_configs = optimizer_configs_from_payload(config)
    if not optimizer_configs:
        raise WorkbenchConfigurationError("Select at least one optimizer")
    runner_config = config["runner"]
    output_dir = (project_root / runner_config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(dict.fromkeys(runner_config["seeds"]))
    results = []

    total_runs = len(seeds) * len(optimizer_configs)
    index = 0
    for optimizer_config in optimizer_configs:
        for seed in seeds:
            index += 1
            environment = make_environment(config["environment"], seed=seed)
            oracle = _make_oracle(config["oracle"], environment, seed, project_root)
            optimizer = _make_optimizer(optimizer_config)
            metrics = _make_metrics(config["metrics"], environment, runner_config)
            benchmark = BenchmarkRunner(
                environment=environment,
                oracle=oracle,
                metrics=metrics,
                record_trajectory=bool(runner_config.get("record_trajectory", True)),
                tail_fraction=float(runner_config.get("tail_fraction", 0.2)),
            )
            result = benchmark.run(
                optimizer=optimizer,
                T=int(runner_config["steps"]),
                x0=_initial_point(
                    environment, seed, configured=runner_config.get("initial_x")
                ),
                seed=seed,
            )
            optimizer_name = optimizer_config["name"]
            stem = f"{optimizer_name}_seed{seed}"
            json_path = output_dir / f"{stem}.json"
            csv_path = (
                output_dir / f"{stem}.csv" if runner_config.get("export_csv") else None
            )
            _write_result(result, json_path, csv_path)
            results.append(
                {
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "status": result.status,
                    "runtime": result.runtime,
                    "final_metrics": result.final_metrics,
                    "json": str(json_path.relative_to(project_root)).replace("\\", "/"),
                    "csv": (
                        str(csv_path.relative_to(project_root)).replace("\\", "/")
                        if csv_path
                        else None
                    ),
                }
            )
            print(
                f"[{index}/{total_runs}] seed={seed} optimizer={optimizer_name} "
                f"status={result.status} runtime={result.runtime:.3f}s",
                flush=True,
            )

    optimizer_names = [optimizer["name"] for optimizer in optimizer_configs]
    summary = {
        "status": "SUCCESS",
        "optimizer": (
            optimizer_names[0] if len(optimizer_names) == 1 else optimizer_names
        ),
        "optimizers": optimizer_names,
        "environment": config["environment"],
        "oracle": config["oracle"]["type"],
        "metrics": config["metrics"],
        "runs": results,
    }
    (output_dir / "workbench_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a WIND workbench configuration")
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parent.parent
    run_workbench(config, project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
