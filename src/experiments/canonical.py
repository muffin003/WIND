"""Versioned protocol specification and independent artifact validation.

The validator accepts third-party algorithms. It checks tasks, splits, costs,
decision streams and completeness, not membership in WIND's reference catalog.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import numpy as np
from wind_benchmark.experiments.standard import default_tasks, SUITE_VERSION


def specification():
    return {
        "suite": SUITE_VERSION,
        "schema_version": 3,
        "generator": "wind_tracking_v1.haar_qr_signs.sphere_zero_start",
        "task_parameters": {
            "quadratic_eigenvalues": "linear spacing from 1 to kappa; random orthogonal basis",
            "pnorm": {"p": 1.5},
            "multiextremal": {"width": 1.0},
            "gaussian": {
                "value_sigma": 0.1,
                "gradient_sigma": ".1/sqrt(flattened_dimension)",
            },
            "correlated": {
                "phi": 0.9,
                "sigma": "Gaussian baseline scale",
                "initial": 0,
                "innovations": "Gaussian",
            },
            "heavy_tailed": {
                "alpha": 1.5,
                "scale": "Gaussian baseline scale",
                "clip": "1000*scale",
            },
            "sparse": {"p": 0.05, "sigma": "Gaussian baseline scale/sqrt(.05)"},
            "multiplicative": {"sigma_rel": 0.3},
            "quantized": {"delta": "Gaussian baseline scale"},
            "linear": {"speed": 0.005},
            "random_walk": {"coordinate_sigma": ".005/sqrt(d)"},
            "cyclic": {
                "amplitude_per_coordinate": "1/sqrt(d)",
                "period": 1000,
                "center": "theta0",
            },
            "jump": {"interval": 800, "magnitude": 1.0},
            "simplex_cycle": {
                "period": 1000,
                "formula": "theta0_i*exp(sin(2*pi*t/P+i)-sin(i)), normalized",
            },
            "stiefel": {"sigma": 0.005, "map": "Cayley skew Gaussian left rotation"},
        },
        "tasks": [asdict(task) for task in default_tasks()],
        "development_seeds": list(range(20001, 20021)),
        "evaluation_seeds": list(range(10001, 10051)),
        "query_budget": 4000,
        "decision_stream": "queried-point",
        "clock": "streaming: one transition per oracle query",
        "initialization": {
            "euclidean": "x0=0; theta0 uniform on unit sphere",
            "simplex": "independent Dirichlet(ones) feasible states",
            "manifold": "Gaussian QR with positive R diagonal",
        },
        "primary": {
            "stationary": "mean squared error over final 20% of queries",
            "dynamic": "mean squared error over all queries",
        },
        "distance": {
            "euclidean": "L2",
            "simplex": "L2",
            "stiefel": "frame Frobenius",
            "grassmann": "L2 principal angles, feasible representatives only",
        },
        "query_efficiency": {
            "epsilon": 0.1,
            "include_initial_point": True,
            "failure": "not a full-budget censored observation",
        },
        "recovery": {
            "epsilon": 0.2,
            "window": 400,
            "consecutive": 20,
            "time": "queries after change including confirmation",
        },
        "tuning": "fixed configuration / no in-suite tuning",
        "bootstrap_repetitions": 10000,
        "confidence": 0.95,
        "failure_interval": "Wilson",
        "curve_intervals": "pointwise",
        "references": {
            "euclidean_first_order": "SGD",
            "euclidean_zero_order": "SPSA",
            "simplex": "ProjectedGradient",
            "manifold": "RiemannianSGD",
        },
        "failure_criterion": "nonfinite arithmetic or decision norm > 1e12",
        "feasibility_tolerance": 1e-6,
        "metrics_protocol": "metrics_v2",
        "privacy_protocol": "local_series_v2",
    }


def specification_hash():
    return hashlib.sha256(
        json.dumps(specification(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_result(directory, *, split="evaluation"):
    try:
        return _validate_result(directory, split=split)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {
            "valid": False,
            "errors": [f"Incomplete or damaged result: {type(exc).__name__}: {exc}"],
        }


def _validate_result(directory, *, split="evaluation"):
    directory = Path(directory)
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Nonstandard JSON {value}")
        ),
    )
    runs = pd.read_csv(directory / "runs.csv")
    spec, errors = specification(), []
    indexed = {
        key: row
        for key, row in runs.set_index(["task", "algorithm", "seed"]).iterrows()
    }

    def require(condition, message):
        if not condition:
            errors.append(message)

    if manifest.get("execution", {}).get("version") == "canonical_process_v1":
        status = json.loads(
            (directory / "execution_status.json").read_text(encoding="utf-8")
        )
        require(status.get("status") == "complete", "Execution/export is not complete")
        hashes = json.loads(
            (directory / "artifact_hashes.json").read_text(encoding="utf-8")
        )
        required = {
            "runs.csv",
            "summary.csv",
            "paired_comparisons.csv",
            "metrics_summary.csv",
            "curves.csv",
            "trajectories.npz",
            "trace_index.csv",
            "recovery_events.csv",
            "manifest.json",
        }
        require(required <= set(hashes), "Incomplete export inventory")
        for name, digest in hashes.items():
            path = directory / name
            require(
                path.resolve().is_relative_to(directory.resolve()), "Unsafe export path"
            )
            if not path.resolve().is_relative_to(directory.resolve()):
                continue
            require(
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                f"Damaged export: {name}",
            )

    require(
        manifest.get("run_kind") == "canonical",
        "Custom study is not a canonical result",
    )
    require(manifest.get("suite") == SUITE_VERSION, "Wrong suite identifier")
    require(
        manifest.get("specification_sha256") == specification_hash(),
        "Specification fingerprint differs",
    )
    require(
        manifest.get("export_schema_version") in (3, 4), "Expected export schema 3 or 4"
    )
    if manifest.get("export_schema_version") == 4:
        require(bool(manifest.get("catalog_version")), "Missing catalog version")
        require(
            bool(manifest.get("reference_implementations")),
            "Missing reference implementation IDs",
        )
    require(manifest.get("decision_stream") == "queried-point", "Wrong decision stream")
    expected_tasks = {t["name"]: t for t in spec["tasks"]}
    actual_tasks = {t["name"]: t for t in manifest.get("tasks", [])}
    require(
        actual_tasks == expected_tasks, "Task instances differ: this is a custom study"
    )
    seeds = spec[split + "_seeds"]
    profile = manifest.get("profile", {})
    require(profile.get("seeds") == seeds, "Wrong split/seeds")
    require(profile.get("budget") == 4000, "Wrong budget")
    require(profile.get("bootstrap") == 10000, "Wrong scalar bootstrap budget")
    require(not runs.duplicated(["task", "algorithm", "seed"]).any(), "Duplicate runs")
    expected_rows = set()
    compatible = 0
    methods = manifest.get("methods", [])
    require(
        len({m["name"] for m in methods}) == len(methods) > 0,
        "Empty or duplicate methods",
    )
    for method in methods:
        require(
            all(
                k in method
                for k in ("implementation", "rng_key", "parameters", "geometries")
            ),
            "Incomplete algorithm provenance",
        )
        for task_name, task in expected_tasks.items():
            allowed = task["geometry"] in method["geometries"]
            compatible += len(seeds) * allowed
            for seed in seeds:
                key = (task_name, method["name"], seed)
                expected_rows.add(key)
                row = indexed.get(key)
                if row is None:
                    errors.append(f"Missing/duplicate run: {key}")
                    continue
                require(
                    row.status
                    in (
                        {"SUCCESS", "NUMERICAL_FAILURE", "INFEASIBLE"}
                        if allowed
                        else {"SKIPPED"}
                    ),
                    f"Invalid outcome: {key}: {row.status}",
                )
                require(
                    0 <= row.queries <= 4000 and row.budget == 4000,
                    f"Invalid cost: {key}",
                )
                require(
                    row.value_queries <= row.queries
                    and row.gradient_queries <= row.queries,
                    f"Channel counts exceed total: {key}",
                )
                if row.status == "SUCCESS":
                    require(row.queries == 4000, f"Incomplete successful run: {key}")
                if row.status == "NUMERICAL_FAILURE":
                    require(
                        pd.isna(row.get("score")),
                        f"Failure reported as a complete score: {key}",
                    )
    require(
        set(runs[["task", "algorithm", "seed"]].itertuples(index=False, name=None))
        == expected_rows,
        "Run keys differ from declared design",
    )
    index_path = directory / "trace_index.csv"
    if index_path.exists() and (directory / "trajectories.npz").exists():
        trace_index = pd.read_csv(index_path)
        require(
            len(trace_index) == int((runs.status == "SUCCESS").sum()),
            "Missing successful-run traces",
        )
        with np.load(directory / "trajectories.npz", allow_pickle=False) as traces:
            for row in trace_index.itertuples(index=False):
                key = (row.task, row.algorithm, row.seed)
                if key not in indexed or row.key not in traces:
                    errors.append(f"Missing trace {key}")
                    continue
                values = traces[row.key]
                require(
                    values.shape == (4000,) and np.isfinite(values).all(),
                    f"Invalid trace {key}",
                )
                expected = float(
                    np.mean(
                        values[-800:]
                        if expected_tasks[row.task]["drift"] == "stationary"
                        else values
                    )
                )
                require(
                    np.isclose(expected, indexed[key].score, rtol=1e-10, atol=1e-12),
                    f"Score/trace mismatch {key}",
                )
    else:
        errors.append("Full queried-point histories are missing")
    return {
        "valid": not errors,
        "errors": errors,
        "compatible_runs": compatible,
        "actual_queries": int(runs.queries.sum()),
        "maximum_queries": compatible * 4000,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--split", choices=("evaluation", "development"), default="evaluation"
    )
    args = parser.parse_args(argv)
    result = validate_result(args.directory, split=args.split)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
