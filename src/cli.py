"""Command-line interface for reproducible WIND experiment suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .experiment import run_full_experiment_suite

DEFAULT_CONFIG: Dict[str, Any] = {
    "output_dir": "results_lyapunov",
    "seeds": [42, 43, 44, 45, 46],
    "steps": 500,
    "rho_values": [1.0, 0.5, 0.2],
    "drift_values": [0.001, 0.01, 0.1, 0.3, 0.6, 1.0],
    "dimensions": [5],
    "optimizers": None,
}


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is not None:
        with path.open("r", encoding="utf-8") as stream:
            supplied = json.load(stream)
        unknown = set(supplied) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {sorted(unknown)}")
        config.update(supplied)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wind-benchmark",
        description="Run a reproducible WIND optimization benchmark suite.",
    )
    parser.add_argument("--config", type=Path, help="JSON experiment configuration")
    parser.add_argument("--output-dir", help="Override the output directory")
    parser.add_argument("--steps", type=int, help="Override steps per run")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.steps is not None:
        config["steps"] = args.steps

    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    run_full_experiment_suite(
        output_dir=config["output_dir"],
        seeds=config["seeds"],
        T=config["steps"],
        rho_values=config["rho_values"],
        A_values=config["drift_values"],
        dimensions=config["dimensions"],
        optimizer_names=config["optimizers"],
    )
    return 0
