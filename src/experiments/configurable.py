"""
Legacy WIND experiment orchestration across (rho, A) regimes.
Finite-horizon diagnostics do not establish theoretical stabilization.

Scientific Guarantees:
  ✓ Lyapunov function V_n(x) = ‖x - θ_n‖_{ρ+1}^{ρ+1} for stabilization analysis
  ✓ Optimal ℓ_{ρ+1} geometry for ρ-Hölder smooth gradients
  ✓ Drift normalization V_n / A^{ρ+1} for fair comparison across speeds
  ✓ Ground truth θ_t inclusion for exact regret computation
  ✓ Single-file export with full optimizer signature + environment config
"""

import json
import hashlib
import numpy as np
from wind_benchmark.core.privacy import _series, _after_release, _user_call
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import warnings
from wind_benchmark.reporting.common import safe_print as print, write_json
from wind_benchmark.core.metrics import ensemble_metrics

from wind_benchmark.core.environment import (
    DynamicEnvironment,
    LinearDrift,
    QuadraticLandscape,
    PNormLandscape,
    RobustLandscape,
    make_environment,
    GaussianNoise,
    HeavyTailedNoise,
    make_noise,
)
from wind_benchmark.core.oracle import FirstOrderOracle, ZeroOrderOracle, Observation
from wind_benchmark.core.metrics import (
    MetricsCollection,
    TrackingErrorMetric,
    LyapunovMetric,
    NormalizedLyapunovMetric,
    AsymptoticBoundMetric,
    DynamicRegretMetric,
    DriftAdaptationMetric,
)
from wind_benchmark.core.runner import (
    BenchmarkRunner,
    OptimizerProtocol,
    ExperimentResult,
)
from wind_benchmark.core.randomness import stream_seed

# ============================================================================
# 2. ENVIRONMENT AND METRIC FACTORIES
# ============================================================================


def create_environment(
    rho: float, A: float, dim: int, noise_type: str = "gaussian", seed: int = 42
) -> DynamicEnvironment:
    """
    Create environment with controlled Hölder exponent ρ and drift magnitude A.

    Args:
        rho: Hölder exponent ∈ (0, 1] for gradient smoothness
        A: Drift magnitude bound ‖θ_n - θ_{n-1}‖_1 ≤ A
        dim: Problem dimensionality
        noise_type: 'gaussian' or 'heavy_tailed'
        seed: Random seed

    Returns:
        Configured DynamicEnvironment instance
    """
    # Select landscape based on Hölder exponent ρ
    if rho == 1.0:
        landscape = QuadraticLandscape(dim=dim, condition_number=5.0, seed=seed)
    elif rho == 0.5:
        landscape = PNormLandscape(p=1.5)  # p = ρ+1 = 1.5
    elif rho == 0.2:
        landscape = RobustLandscape(delta=0.1)  # Approximate ρ=0.2 smoothness
    else:
        raise ValueError(f"Unsupported rho value: {rho}")

    # Linear drift with velocity scaled to achieve ‖Δθ‖_1 ≈ A
    # For uniform direction: ‖v‖_1 = d * |v_i|, so set v_i = A / d
    velocity = np.ones(dim) * (A / dim)

    env = DynamicEnvironment(
        dim=dim,
        drift=LinearDrift(velocity=velocity.tolist()),
        landscape=landscape,
        bounds=[-10, 10],
    )
    return env


def create_metrics(rho: float, A: float) -> MetricsCollection:
    """
    Create metrics collection with theoretically grounded Lyapunov analysis.

    Args:
        rho: Hölder exponent ∈ (0, 1]
        A: Drift magnitude bound

    Returns:
        MetricsCollection with Lyapunov metrics in ℓ_{ρ+1} geometry
    """
    metrics = [
        # Stabilization metrics (Definition 1)
        LyapunovMetric(rho=rho),
        NormalizedLyapunovMetric(rho=rho, drift_magnitude_A=A),
        AsymptoticBoundMetric(rho=rho),
        # Tracking performance in optimal ℓ_{ρ+1} norm
        TrackingErrorMetric(norm=rho + 1.0, normalize_by_dim=True),
        # Regret analysis
        DynamicRegretMetric(normalize_by_path=True),
        # Adaptation behavior
        DriftAdaptationMetric(),
    ]
    return MetricsCollection(metrics)


# ============================================================================
# 3. MAIN EXPERIMENT
# ============================================================================


@_series
def run_single_experiment(
    optimizer: OptimizerProtocol,
    rho: float,
    A: float,
    dim: int,
    T: int = 500,
    seed: int = 42,
    noise_type: str = "gaussian",
) -> ExperimentResult:
    """
    Execute single experiment with full Lyapunov analysis.

    Returns:
        ExperimentResult with ground truth theta_t and Lyapunov metrics
    """
    # Create environment
    env = create_environment(rho=rho, A=A, dim=dim, noise_type=noise_type, seed=seed)

    # Resolve the declared catalog identity instead of inferring capabilities from
    # substrings in the class name. Third-party optimizers may still provide their
    # own explicit ``oracle_type`` metadata.
    oracle_type = _user_call(getattr, optimizer, "oracle_type", None)
    if oracle_type not in {"first-order", "zero-order"}:
        raise ValueError(
            "Declare oracle_type='first-order' or 'zero-order'; a display name is not a capability"
        )
    is_zero_order = oracle_type == "zero-order"

    def observation_noise(scale, channel):
        if noise_type not in {"none", "gaussian", "heavy_tailed"}:
            raise ValueError(f"Unsupported noise_type: {noise_type}")
        parameters = (
            {"alpha": 2.2, "scale": scale}
            if noise_type == "heavy_tailed"
            else {"sigma": scale}
        )
        if noise_type == "none":
            parameters = {}
        else:
            parameters["seed"] = stream_seed(seed, channel)
        return make_noise(noise_type, **parameters)

    if not is_zero_order:
        oracle = FirstOrderOracle(
            env,
            value_noise=observation_noise(0.01, "value_noise"),
            grad_noise=observation_noise(0.01, "gradient_noise"),
            seed=seed,
            blind_value=False,  # legacy experiment relies on value being available
        )
    else:
        oracle = ZeroOrderOracle(
            env, value_noise=observation_noise(0.05, "value_noise"), seed=seed
        )

    # Create metrics
    metrics = create_metrics(rho=rho, A=A)

    # Create runner
    runner = BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        record_trajectory=True,
        tail_fraction=0.2,
    )

    # Generate x0 from the experiment seed before the runner starts. Keeping this
    # generator local makes a saved seed sufficient to reproduce a single run.
    x0 = np.zeros(dim)

    # Run experiment
    result = runner.run(optimizer=optimizer, T=T, x0=x0, seed=seed)

    return result


@_series
def run_full_experiment_suite(
    output_dir: str = "results/lyapunov",
    seeds: Optional[List[int]] = None,
    T: int = 500,
    rho_values: Optional[List[float]] = None,
    A_values: Optional[List[float]] = None,
    dimensions: Optional[List[int]] = None,
    optimizer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run complete experimental suite across (ρ, A) regimes.

    Returns:
        Dictionary with aggregated results and metadata
    """
    seeds = [42, 43, 44, 45, 46] if seeds is None else seeds
    rho_values = [1.0, 0.5, 0.2] if rho_values is None else rho_values
    A_values = [0.001, 0.01, 0.1, 0.3, 0.6, 1.0] if A_values is None else A_values
    dimensions = [5] if dimensions is None else dimensions

    from wind_benchmark.algorithms.catalog import (
        OPTIMIZER_SPECS,
        resolve_optimizer_spec,
    )

    selected_specs = []
    if optimizer_names is None:
        selected_specs = [(spec, spec.canonical_name) for spec in OPTIMIZER_SPECS]
    else:
        seen = set()
        unknown = []
        for requested_name in optimizer_names:
            try:
                spec = resolve_optimizer_spec(requested_name)
            except KeyError:
                unknown.append(requested_name)
                continue
            if spec.canonical_name in seen:
                raise ValueError(
                    "Optimizer selection resolves to duplicate canonical name: "
                    f"{spec.canonical_name}"
                )
            seen.add(spec.canonical_name)
            selected_specs.append((spec, requested_name))
        if unknown:
            raise ValueError(f"Unknown optimizer(s): {sorted(unknown)}")

    optimizers_config = [
        (
            spec.canonical_name,
            lambda spec=spec: spec.instantiate(),
            requested_name,
            spec.implementation_id,
        )
        for spec, requested_name in selected_specs
    ]

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Total number of runs
    total_runs = (
        len(rho_values)
        * len(A_values)
        * len(dimensions)
        * len(optimizers_config)
        * len(seeds)
    )
    print(f"🚀 Starting WIND experiment suite")
    print(f"   Configurations: ρ ∈ {rho_values}, A ∈ {A_values}, dim ∈ {dimensions}")
    print(f"   Algorithms: {len(optimizers_config)} explicitly identified optimizers")
    print(f"   Seeds: {seeds}")
    print(f"   Total runs: {total_runs}")
    print(f"   Steps per run: {T}")
    print()

    # Progress tracking
    results = []
    pending_exports = []
    run_counter = 0
    start_time = time.time()

    # Main experiment loop
    for rho in rho_values:
        for A in A_values:
            for dim in dimensions:
                for (
                    opt_name,
                    opt_factory,
                    requested_name,
                    implementation_id,
                ) in optimizers_config:
                    for seed in seeds:
                        run_counter += 1

                        # Create optimizer
                        try:
                            optimizer = opt_factory()
                            if "euclidean" not in optimizer.geometries:
                                print(
                                    f"SKIPPED {opt_name}: this CLI creates Euclidean tasks; REQUIRED geometry: {optimizer.geometries}, initialize(x0, rng, task_spec), advance(client)"
                                )
                                continue
                            if not hasattr(optimizer, "name"):
                                optimizer.name = opt_name
                        except Exception as e:
                            print(f"⚠️  Failed to create {opt_name}: {e}")
                            continue

                        # Run experiment
                        try:
                            result = run_single_experiment(
                                optimizer=optimizer,
                                rho=rho,
                                A=A,
                                dim=dim,
                                T=T,
                                seed=seed,
                                noise_type="gaussian",
                            )

                            def metric_value(class_name):
                                keys = [
                                    definition["id"]
                                    for name, definition in result.metric_definitions.items()
                                    if definition["class"] == class_name
                                ]
                                return (
                                    result.metric_value(keys[0])
                                    if len(keys) == 1
                                    else np.nan
                                )

                            lyapunov_val = metric_value("LyapunovMetric")
                            normalized_lyapunov = metric_value(
                                "NormalizedLyapunovMetric"
                            )
                            asymptotic_bound = metric_value("AsymptoticBoundMetric")
                            tracking_error = metric_value("TrackingErrorMetric")

                            # Descriptive cutoff, not a test of an asymptotic theorem.
                            is_stabilized = (
                                not np.isnan(lyapunov_val) and lyapunov_val < 100.0
                            )

                            # Store result
                            results.append(
                                {
                                    "algorithm": opt_name,
                                    "implementation_id": implementation_id,
                                    "rho": rho,
                                    "A": A,
                                    "dim": dim,
                                    "seed": seed,
                                    "lyapunov": lyapunov_val,
                                    "normalized_lyapunov": normalized_lyapunov,
                                    "asymptotic_bound": asymptotic_bound,
                                    "tracking_error": tracking_error,
                                    "tail_lyapunov_below_100": is_stabilized,
                                    "asymptotic_bound_status": "requires_ensemble",
                                    "runtime": result.runtime,
                                    "status": result.status,
                                }
                            )

                            # Save individual result to JSON
                            result_filename = (
                                f"rho{rho:.1f}_A{A:.3f}_dim{dim}_"
                                f"{opt_name}_seed{seed}_"
                                + hashlib.sha256(
                                    json.dumps([rho, A, dim, opt_name, seed]).encode()
                                ).hexdigest()[:16]
                                + ".json"
                            )
                            pending_exports.append(
                                (result, output_path / result_filename)
                            )

                            # Progress update
                            elapsed = time.time() - start_time
                            eta = elapsed / run_counter * (total_runs - run_counter)
                            print(
                                f"[{run_counter}/{total_runs}] ρ={rho:.1f}, A={A:.3f}, "
                                f"dim={dim}, algo={opt_name:25s}, seed={seed} "
                                f"runtime={result.runtime:.2f}s "
                                f"(ETA: {eta/60:.1f} min)"
                            )

                        except Exception as e:
                            print(
                                f"❌ Failed run {run_counter}/{total_runs} "
                                f"(ρ={rho}, A={A}, dim={dim}, {opt_name}, seed={seed}): {e}"
                            )
                            continue

    # Final aggregation
    for result, result_path in pending_exports:
        _after_release(result.save_to_json, str(result_path))
    results_df = pd.DataFrame(results)

    # Save full results
    _after_release(results_df.to_csv, output_path / "full_results.csv", index=False)

    # Save aggregated statistics
    agg_stats = results_df.groupby(
        ["algorithm", "implementation_id", "rho", "A", "dim"]
    ).agg(
        {
            "lyapunov": ["mean", "std", "min", "max"],
            "normalized_lyapunov": ["mean", "std"],
            "tracking_error": ["mean", "std"],
            "tail_lyapunov_below_100": "mean",
            "runtime": ["mean", "std"],
        }
    )
    _after_release(agg_stats.to_csv, output_path / "aggregated_statistics.csv")

    ensembles = {}
    for result, _ in pending_exports:
        key = (
            result.optimizer_info.implementation_id,
            json.dumps(
                result.optimizer_info.hyperparameters, sort_keys=True, default=str
            ),
            result.environment_config.dim,
            json.dumps(result.environment_config.drift_parameters, sort_keys=True),
            json.dumps(result.metric_definitions, sort_keys=True),
        )
        ensembles.setdefault(key, []).append(result)
    _after_release(
        write_json,
        output_path / "ensemble_metrics.json",
        [
            {"stratum": list(key), "metrics": ensemble_metrics(group)}
            for key, group in ensembles.items()
        ],
    )

    # Save experiment metadata
    metadata = {
        "metrics_protocol": "metrics_v2",
        "schema_version": 4,
        "experiment_id": f"sthrd_lyapunov_{time.strftime('%Y%m%d_%H%M%S')}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runs": run_counter,
        "rho_values": rho_values,
        "A_values": A_values,
        "dimensions": dimensions,
        "algorithms": [name for name, _, _, _ in optimizers_config],
        "implementation_ids": {
            name: implementation_id
            for name, _, _, implementation_id in optimizers_config
        },
        "seeds": seeds,
        "T": T,
        "elapsed_time_seconds": time.time() - start_time,
        "output_directory": str(output_path),
        "scaled_temporal_secant_lr": 0.005,
        "heavy_ball_lr": 0.1,
        "adam_lr": 0.001,
    }
    _after_release(
        (output_path / "experiment_metadata.json").write_text,
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n✅ Experiment suite completed!")
    print(f"   Results saved to: {output_path}")
    print(f"   Total runtime: {metadata['elapsed_time_seconds']/60:.1f} minutes")
    print(f"   Successful runs: {len(results)} / {run_counter}")

    return {
        "results_df": results_df,
        "aggregated_stats": agg_stats,
        "metadata": metadata,
        "output_path": output_path,
    }


# ============================================================================
# 4. COMMAND-LINE EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("WIND BENCHMARK: Theoretically Grounded Stabilization Analysis")
    print("=" * 80)
    print()
    print("Finite-horizon diagnostics motivated by Definitions 1-2:")
    print("  Definition 1 (Stabilization): ∃ C > 0 such that 𝔼[V_n(x̂_n)] ≤ C ∀ n")
    print("  Definition 2 (Asymptotic Bound): L = lim sup_{n→∞} 𝔼[V_n(x̂_n)] < ∞")
    print()
    print("Where V_n(x) = ‖x - θ_n‖_{ρ+1}^{ρ+1} is the Lyapunov function")
    print("for ρ-Hölder smooth gradients with drift magnitude A.")
    print()
    print("⚠️  Temporal-secant baselines use lr=0.005 for stability")
    print()

    # Run experiment
    results = run_full_experiment_suite(
        output_dir="results/lyapunov", seeds=[42, 43, 44, 45, 46], T=500
    )

    # Summary statistics
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    df = results["results_df"]

    # Overall stabilization rate
    total_stabilized = df["tail_lyapunov_below_100"].sum()
    total_runs = len(df)
    stabilization_rate = total_stabilized / total_runs * 100

    print(
        f"\nTail Lyapunov below 100 (descriptive, not a proof): {stabilization_rate:.1f}% "
        f"({total_stabilized}/{total_runs} runs)"
    )

    # Best algorithms by ρ regime
    print("\nTop 3 algorithms by Hölder exponent ρ:")
    for rho in [1.0, 0.5, 0.2]:
        rho_df = df[df["rho"] == rho]
        algo_stats = (
            rho_df.groupby("algorithm")["normalized_lyapunov"]
            .mean()
            .sort_values()
            .head(3)
        )
        print(f"\n  ρ = {rho:.1f}:")
        for rank, (algo, val) in enumerate(algo_stats.items(), 1):
            print(f"    {rank}. {algo:25s} → normalized Lyapunov = {val:.3f}")

    # Drift robustness
    print("\nDrift robustness (normalized Lyapunov across A ∈ [0.001, 1.0]):")
    algo_drift_robustness = (
        df.groupby("algorithm")["normalized_lyapunov"].std().sort_values().head(5)
    )
    for rank, (algo, std_val) in enumerate(algo_drift_robustness.items(), 1):
        print(f"  {rank}. {algo:25s} → std = {std_val:.3f} (lower = more robust)")

    print("\n" + "=" * 80)
    print("✅ All results saved to ./results/lyapunov/")
    print("   Key files:")
    print("     • full_results.csv          - Raw results for all runs")
    print("     • aggregated_statistics.csv - Mean/std per (algo, ρ, A)")
    print("     • experiment_metadata.json  - Complete experiment configuration")
    print("     • *.json                    - Individual run results with ground truth")
    print("=" * 80)
