"""Rebuild standard-run figures from saved evaluator outputs, without rerunning algorithms.

python -m wind_benchmark.reporting.standard results/default_pipeline/run_001
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def curve_interval(values, repetitions=2000, seed=0):
    """Resample WHOLE runs; shared bootstrap weights at every query coordinate."""
    values = np.asarray(values, dtype=float)
    n, width = values.shape
    mean = values.mean(axis=0)
    if n < 2:
        return mean, np.full(width, np.nan), np.full(width, np.nan)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=repetitions).astype(float) / n
    low, high = np.empty(width), np.empty(width)
    # Bound memory without changing the resampled run identities across time.
    for start in range(0, width, 256):
        sample = weights @ values[:, start : start + 256]
        low[start : start + 256], high[start : start + 256] = np.quantile(
            sample, [0.025, 0.975], axis=0
        )
    return mean, low, high


def tracking_scale(axes, *, linear_when_narrow=False):
    """Shared nonnegative limits, including interval collections in dataLim.

    Set limits explicitly: switching to symlog after drawing vlines can retain
    linear autoscale padding below zero and compress all positive scores.
    """
    axes = list(axes)
    bounds = np.array([ax.dataLim.intervaly for ax in axes])
    bounds = bounds[np.all(np.isfinite(bounds), axis=1)]
    lower = max(0.0, float(bounds[:, 0].min()) * 0.8) if len(bounds) else 0.0
    upper = max(1e-10, float(bounds[:, 1].max()) * 1.25) if len(bounds) else 1.0
    narrow = linear_when_narrow and lower > 0 and upper / lower < 10
    for ax in axes:
        if narrow:
            ax.set_yscale("linear")
        else:
            ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_ylim(lower, upper)


def render_report(output: Path):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output / "summary.csv")
    paired = pd.read_csv(output / "paired_comparisons.csv")
    metrics = pd.read_csv(output / "metrics_summary.csv")
    index = pd.read_csv(output / "trace_index.csv")
    tasks = {t["name"]: t for t in manifest["tasks"]}
    methods = [m["name"] for m in manifest["methods"]]
    budget = manifest["profile"]["budget"]
    colors = {name: f"C{i % 10}" for i, name in enumerate(methods)}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    assets = []

    def save(fig, name, title):
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0.025, 1, 0.94))
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(figures / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        assets.append((name, title))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    # One availability matrix; SKIP is neither failure nor zero quality.
    matrix = np.zeros((len(tasks), len(methods)))
    labels = np.empty_like(matrix, dtype=object)
    for i, task in enumerate(tasks):
        for j, method in enumerate(methods):
            rows = summary[(summary.task == task) & (summary.algorithm == method)]
            if rows.empty:
                labels[i, j], matrix[i, j] = "N/A", 0
                continue
            row = rows.iloc[0]
            total = int(row.n_success + row.n_failed)
            matrix[i, j] = (
                0
                if not total
                else (1 if row.n_failed == 0 else (2 if row.n_success else 3))
            )
            labels[i, j] = "SKIP" if not total else f"{int(row.n_success)}/{total}"
    fig, ax = plt.subplots(
        figsize=(max(8, len(methods) * 1.7), max(3, len(tasks) * 0.34))
    )
    ax.imshow(
        matrix,
        cmap=ListedColormap(["#ececec", "#d7efdf", "#ffe3ab", "#efb6b6"]),
        vmin=0,
        vmax=3,
        aspect="auto",
    )
    ax.set_xticks(range(len(methods)), methods, rotation=25, ha="right")
    ax.set_yticks(range(len(tasks)), list(tasks))
    for (i, j), label in np.ndenumerate(labels):
        ax.text(j, i, label, ha="center", va="center", fontsize=9, color="#202020")
    save(fig, "coverage", "Coverage: successful / attempted runs; SKIP = incompatible")

    # Persist curve aggregates with confidence bounds at EVERY recorded query.
    aggregates, band_tables = {}, []
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        for (task, method, oracle), group in index.groupby(
            ["task", "algorithm", "oracle"], sort=False
        ):
            group = group.sort_values("seed")
            values = np.stack([archive[key] for key in group.key])
            mean, lo, hi = curve_interval(
                values,
                min(2000, manifest["profile"]["bootstrap"]),
                0,  # Identical resampling weights, independent of display labels.
            )
            aggregates[(task, method, oracle)] = (mean, lo, hi, len(values))
            band_tables.append(
                pd.DataFrame(
                    {
                        "task": task,
                        "algorithm": method,
                        "oracle": oracle,
                        "query": np.arange(1, len(mean) + 1),
                        "n_success": len(values),
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
            )
    (
        pd.concat(band_tables, ignore_index=True).to_csv(
            output / "curve_summary.csv", index=False
        )
        if band_tables
        else pd.DataFrame(
            columns=[
                "task",
                "algorithm",
                "oracle",
                "query",
                "n_success",
                "mean",
                "ci_low",
                "ci_high",
            ]
        ).to_csv(output / "curve_summary.csv", index=False)
    )

    groups = [
        ("reference", ["clean", "gaussian"]),
        *[
            (land, [f"{land}_none", f"{land}_gaussian"])
            for land in ("conditioned", "pnorm", "rosenbrock", "multimodal")
        ],
        (
            "noise_persistence_outliers",
            ["gaussian", "correlated", "heavy_tailed", "sparse"],
        ),
        ("noise_signal_precision", ["gaussian", "multiplicative", "quantized"]),
        ("continuous_motion", ["gaussian", "linear", "random_walk", "cyclic"]),
        ("jump_interaction", ["gaussian", "jump", "jump_outliers"]),
        ("linear_interaction", ["gaussian", "linear", "linear_correlated"]),
        ("dimension", ["linear", "linear_d30", "linear_d100"]),
        *[
            (
                geo,
                [
                    f"{geo}_stationary_gaussian",
                    f"{geo}_moving_gaussian",
                    f"{geo}_moving_correlated",
                ],
            )
            for geo in ("simplex", "stiefel", "grassmann")
        ],
    ]
    covered = {name for _, names in groups for name in names}
    groups.extend((name, [name]) for name in tasks if name not in covered)

    def draw(ax, task_name, oracle, window=None):
        subset = summary[
            (summary.task == task_name)
            & (summary.oracle == oracle)
            & ((summary.n_success + summary.n_failed) > 0)
        ]
        has_curve = False
        for row in subset.itertuples():
            data = aggregates.get((task_name, row.algorithm, oracle))
            label = f"{row.algorithm}: {row.n_success} ok, {row.n_failed} failed"
            if data is None:
                ax.plot([], [], color=colors[row.algorithm], label=label)
                continue
            has_curve = True
            mean, low, high, count = data
            q = np.arange(1, len(mean) + 1)
            mask = (
                np.ones(len(q), dtype=bool)
                if window is None
                else ((q >= window[0]) & (q <= window[1]))
            )
            ax.plot(
                q[mask], mean[mask], color=colors[row.algorithm], label=label, lw=1.2
            )
            if count >= 2:
                ax.fill_between(
                    q[mask],
                    low[mask],
                    high[mask],
                    color=colors[row.algorithm],
                    alpha=0.18,
                    linewidth=0,
                )
        if tasks[task_name]["drift"] == "jump":
            for position in range(budget // 5, budget, budget // 5):
                # Query number is 1-based: the first changed observation is start+1.
                if window is None or window[0] <= position + 1 <= window[1]:
                    ax.axvline(position + 1, color=".45", ls="--", lw=0.8)
        if not has_curve:
            ax.text(0.5, 0.5, "No successful runs", ha="center", transform=ax.transAxes)
        ylabel = {
            "stiefel": "Squared frame Frobenius distance",
            "grassmann": "Squared principal-angle distance",
        }.get(tasks[task_name]["geometry"], "Squared tracking distance")
        ax.set(title=task_name, xlabel="Oracle calls", ylabel=ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    for oracle in summary.oracle.unique():
        applicable = set(
            summary[
                (summary.oracle == oracle)
                & ((summary.n_success + summary.n_failed) > 0)
            ].task
        )
        for group_name, names in groups:
            names = [name for name in names if name in applicable]
            if not names:
                continue
            cols = min(2, len(names))
            fig, axes = plt.subplots(
                (len(names) + cols - 1) // cols,
                cols,
                figsize=(6.3 * cols, 4 * ((len(names) + cols - 1) // cols)),
                squeeze=False,
                sharey=True,
            )
            for ax, name in zip(axes.flat, names):
                draw(ax, name, oracle)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            tracking_scale(list(axes.flat)[: len(names)])
            save(
                fig,
                f"curves_{oracle}_{group_name}",
                f"{group_name} | {oracle} | shared y scale\nSuccessful-run means, pointwise 95% bootstrap CI (not simultaneous)",
            )
        for task in [t for t in applicable if tasks[t]["drift"] == "jump"]:
            fig, axes = plt.subplots(2, 2, figsize=(12.6, 8), sharey=True)
            for ax, start in zip(axes.flat, range(budget // 5, budget, budget // 5)):
                draw(
                    ax,
                    task,
                    oracle,
                    (max(1, start - budget // 100), min(budget, start + budget // 10)),
                )
                ax.set_title(f"{task}: change at query {start+1}")
            tracking_scale(axes.flat)
            save(
                fig,
                f"jump_windows_{oracle}_{task}",
                f"Post-change detail | {oracle}\nFull query resolution; dashed lines mark changes; pointwise 95% CI",
            )

        # Failure probability, independent of conditional quality.
        failures = summary[
            (summary.oracle == oracle) & ((summary.n_success + summary.n_failed) > 0)
        ]
        for page, start in enumerate(range(0, len(failures), 16), 1):
            block = failures.iloc[start : start + 16]
            fig, ax = plt.subplots(figsize=(10, max(3, len(block) * 0.38)))
            for y, row in enumerate(block.itertuples()):
                ax.hlines(
                    y,
                    row.failure_ci_low,
                    row.failure_ci_high,
                    color=colors[row.algorithm],
                )
                ax.plot(row.failure_rate, y, "o", color=colors[row.algorithm])
            ax.set_yticks(
                range(len(block)),
                [f"{r.task} / {r.algorithm}" for r in block.itertuples()],
            )
            ax.invert_yaxis()
            ax.set(
                xlim=(-0.02, 1.02), xlabel="Failure probability (95% Wilson interval)"
            )
            ax.grid(axis="x", alpha=0.2)
            save(
                fig,
                f"failures_{oracle}_{page}",
                f"Failures | {oracle} | incompatible tasks excluded",
            )

        # Raw paired effects stay on task-local axes (loss scales may differ).
        comparisons = paired[paired.oracle == oracle]
        for start in range(0, len(comparisons), 4):
            block = comparisons.iloc[start : start + 4]
            fig, axes = plt.subplots(
                len(block), 1, figsize=(10, 2.2 * len(block)), squeeze=False
            )
            for ax, row in zip(axes.flat, block.itertuples()):
                ax.axvline(0, color=".5", ls="--", lw=1)
                if pd.notna(row.mean_difference_success_pairs_only):
                    ax.plot(
                        row.mean_difference_success_pairs_only,
                        0,
                        "o",
                        color=colors[row.algorithm],
                    )
                    if pd.notna(row.ci_low):
                        ax.hlines(
                            0,
                            row.ci_low,
                            row.ci_high,
                            color=colors[row.algorithm],
                            lw=2,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Not estimable: no successful pairs",
                        ha="center",
                        transform=ax.transAxes,
                    )
                ax.set_title(
                    f"{row.task}: {row.algorithm} minus {row.baseline}; pairs={row.n_pairs}, excluded={row.n_excluded_pairs}"
                )
                ax.set(
                    yticks=[],
                    xlabel="Difference in primary score (negative favors named algorithm)",
                )
            save(
                fig,
                f"paired_{oracle}_{start//4+1}",
                f"Paired effects | {oracle}\nIndividual 95% CI; conditional on both methods succeeding; task-local axes",
            )

        for kind in ("recovery_fraction", "restricted_recovery_time"):
            block = metrics[(metrics.oracle == oracle) & (metrics.metric == kind)]
            if block.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(block))))
            for y, row in enumerate(block.itertuples()):
                ax.plot(row.mean_success_only, y, "o", color=colors[row.algorithm])
                if pd.notna(row.ci_low):
                    ax.hlines(y, row.ci_low, row.ci_high, color=colors[row.algorithm])
            ax.set_yticks(
                range(len(block)),
                [
                    f"{r.task} / {r.algorithm} (n={r.n_success})"
                    for r in block.itertuples()
                ],
            )
            ax.invert_yaxis()
            ax.set_xlabel(
                "Recovered-event fraction"
                if kind == "recovery_fraction"
                else "Restricted recovery time (oracle calls)"
            )
            if kind == "recovery_fraction":
                ax.set_xlim(-0.02, 1.02)
            save(
                fig,
                f"{kind}_{oracle}",
                f"{kind.replace('_', ' ')} | {oracle}\n95% bootstrap CI across runs, not jumps; successful runs only",
            )

        dims = summary[
            (summary.oracle == oracle)
            & summary.task.isin(["linear", "linear_d30", "linear_d100"])
        ]
        if (dims.n_success > 0).any():
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for method, block in dims.groupby("algorithm", sort=False):
                block = block[block.n_success > 0].copy()
                if block.empty:
                    continue
                block["dimension"] = block.task.map(lambda t: tasks[t]["dim"])
                block = block.sort_values("dimension")
                ax.plot(
                    block.dimension,
                    block.mean_success_only,
                    "o-",
                    color=colors[method],
                    label=method,
                )
                ax.vlines(
                    block.dimension, block.ci_low, block.ci_high, color=colors[method]
                )
            ax.set(
                xlabel="Dimension",
                ylabel="Mean squared tracking distance",
                xticks=[10, 30, 100],
            )
            tracking_scale([ax], linear_when_narrow=True)
            ax.legend()
            ax.grid(alpha=0.2)
            save(
                fig,
                f"dimension_{oracle}",
                f"Scaling at fixed query budget | {oracle}\n95% run-level CI; successful runs only; inspect failure panels",
            )

    # Replace only this generated section on a re-render, not the run-level report.
    report_path = output / "report.md"
    report = report_path.read_text(encoding="utf-8").split("\n## Figures\n")[0]
    report += "\n## Figures\n\nFull-resolution curves and pointwise run-bootstrap intervals. All quality plots condition on successful runs; use coverage and failure panels alongside them.\n\n"
    for name, title in assets:
        report += f"- {title.replace(chr(10), ' ')}: [PNG](figures/{name}.png) · [SVG](figures/{name}.svg) · [PDF](figures/{name}.pdf)\n"
    report_path.write_text(report, encoding="utf-8")
    (output / "figure_manifest.json").write_text(
        json.dumps(
            {
                "renderer_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "curve_bootstrap": min(2000, manifest["profile"]["bootstrap"]),
                "coverage": "pointwise, not simultaneous",
                "quality_population": "successful runs only",
                "files": [name for name, _ in assets],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return assets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(f"Rendered {len(render_report(parser.parse_args().output))} figure groups")
