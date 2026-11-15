# WIND standard run

The standard run is a predefined suite for evaluating how an optimization
algorithm behaves as task conditions become more demanding. It begins with
stationary quadratic problems, without noise and with Gaussian noise as a
reference. Further tasks vary landscape shape, observation corruption, target
motion and dimension. Selected combinations investigate whether weaknesses
observed separately persist together. Compatible methods also enter simplex,
Stiefel-frame and Grassmann-subspace tracking tasks.

The output is a task-by-task performance profile, not a universal ranking or a
proof of convergence. The suite follows several branches of difficulty; it does
not assume every algorithm finds the tasks harder in the same order.

## Separate experiment file

Implementation: `src/experiments/standard.py`, suite version `wind_tracking_v1`.
The other CLI scenarios remain separate and use the same algorithm catalog.
Install the package from the repository first (`python -m pip install -e .`).

```powershell
# Inspect tasks and execution cost without running algorithms or writing files.
python -m wind_benchmark.experiments.standard --dry-run

# Execution check, not publication evidence: 3 seeds, 120 calls per run.
python -m wind_benchmark.experiments.standard --profile smoke --output-dir local/internal_checks/default_pipeline/check_001

# Standard evaluation: 50 seeds, 4000 calls per run.
python -m wind_benchmark.experiments.standard --profile standard --output-dir results/default_pipeline/run_001

# The file can also be launched directly, with the package installed.
python src/experiments/standard.py --profile smoke --output-dir local/internal_checks/default_pipeline/check_002

# Evaluate selected algorithms or task subsets.
python -m wind_benchmark.experiments.standard --algorithms SGD HeavyBall Adam --tasks clean gaussian linear linear_correlated --output-dir results/default_pipeline/comparison_001
```

Output directories must be empty: previous results are never silently overwritten.
There is no automatic full run when importing the module.

## Parallel execution

`--workers N` controls independent task/method/seed runs, not queries within a
run. The default is `1`, executing in the caller's process. Larger values use
persistent spawned processes on Windows and Linux. The effective count is the
smaller of N and the number of compatible runs; zero compatible runs starts no
workers. Requests above the logical CPU count produce a warning, not a silent
reduction. `--dry-run --workers 4` reports both counts without creating files.

```powershell
wind-canonical --profile standard --workers 4 --output-dir results/wind_tracking_v1/parallel_001 --no-plots
```

Python callers use `run_suite(..., workers=4)`. Put the call under
`if __name__ == "__main__":` in an importable script. For custom methods, parallel
execution requires an importable top-level `factory(task, seed)` in a separate
module. Lambdas, closures and factories defined in `__main__` remain supported
only with `workers=1`; invalid parallel factories are rejected before computation.
Factories and their module imports receive no evaluator authority.

Every process limits supported BLAS/OpenMP pools to one thread using
`threadpoolctl`. The sequential call restores the caller's native thread limits
and NumPy/Python RNG states on exit. Seeds, budgets, task instances and statistical
rules do not depend on worker count or completion order. Resource measurements
do: elapsed run times reflect concurrent load, and per-worker RSS includes its
interpreter, evaluator and allocator history. Series memory is the peak of
contemporaneously sampled coordinator/worker RSS sums, not the sum of separate
process peaks. Sampled RSS may miss shorter peaks and double-count shared pages.

`execution_status.json` reports operational progress only. Evaluator exports still
wait for the complete outer evaluation series; no worker releases its retained
objects between assignments. `complete` is written only after export succeeds.
An abort is `stopped` or `failed`; a post-release storage failure is
`export_failed`. A forcibly killed coordinator may leave stale `running` metadata:
the file alone does not prove a process is alive. Check its PID and process start
time before treating it as active.

Ctrl+C stops assignment, requests cancellation and allows up to five seconds for
workers to finish, then terminates only this executor's remaining workers. Workers
also monitor coordinator death. Numerical failures are recorded without retries;
an infrastructure failure aborts the series. Partial folders are retained.
There is **no resume**: restart in a new output directory, never combine partial
results from earlier attempts. For unattended runs, launch the command in an
independent OS process with stdout/stderr directed to files, not unread pipes or
a tool session whose lifetime ends with a chat turn.

New manifests record executor version `canonical_process_v1`, requested/effective
workers, spawn mode, native thread limits and resource measurements. The suite
specification remains `wind_tracking_v1`. `artifact_hashes.json` verifies the
completed exports; the validator rejects unfinished or damaged new results.
Keep the original artifact set unchanged when producing derivative reports or
figures, since rewriting an inventoried file invalidates its stored checksum.

## Task registry

The suite contains **32 tasks**, using representative conditions rather than the
Cartesian product of all available settings. Default Euclidean dimension is 10.

| Block | Tasks | Conditions |
|---|---:|---|
| References | 2 | Isotropic quadratic, stationary; clean and Gaussian observations |
| Landscape | 8 | Quadratic with condition number 100, p-norm with p=1.5, Rosenbrock, Rastrigin-type; each clean and Gaussian |
| Observation | 5 | Reference quadratic with correlated, truncated heavy-tailed, sparse, multiplicative or quantized observations |
| Dynamics | 4 | Reference quadratic with Gaussian observations; linear, random-walk, cyclic and jump motion |
| Interactions | 2 | Linear motion with correlated errors; jumps with heavy-tailed errors |
| Scaling | 2 | Gaussian/linear task repeated at dimensions 30 and 100, alongside its dimension-10 reference |
| Geometry | 9 | Simplex, Stiefel and Grassmann; each stationary/Gaussian, moving/Gaussian and moving/correlated |

Resolved parameters, initial conditions, named-stream seeds and source hashes
are included in each output manifest. In particular:

- Euclidean initial distance is 1. x0 is zero and theta0 is uniform on the unit sphere. Direction and quadratic
  orientation vary between seeds, but match across algorithms.
- Quadratic eigenvalues span 1 to the requested condition number; spectral norm
  is not held fixed while conditioning increases. Results therefore concern
  the declared spectrum, not an isolated condition-number effect at fixed smoothness.
- Additive Gaussian value noise has standard deviation 0.1. Gradient coordinate
  noise has standard deviation `0.1 / sqrt(d)`, so total expected squared gradient
  noise is 0.01. Here `d` is the flattened decision size.
- Correlated noise has coefficient 0.9, Gaussian innovations, and zero initial
  state. Its limiting marginal scale matches the Gaussian reference; its initial
  variance is smaller. These are not non-Gaussian innovations.
- Heavy-tailed noise uses tail index 1.5 and the same scale convention, with the
  core implementation's clipping at `1000 * scale`. It is **not variance-matched**
  to Gaussian noise.
- Sparse noise activates each observed coordinate with probability 0.05; its
  amplitude is increased by `1 / sqrt(0.05)` to match Gaussian RMS noise.
  For value-only feedback this is occasional scalar corruption, not spatial
  coordinate corruption.
- Multiplicative noise has relative scale 0.3. Quantization resolution is 0.1 for
  values and `0.1 / sqrt(d)` for gradient coordinates. Quantization is deterministic.
- Linear motion has norm 0.005 per query. Random-walk increments have RMS norm
  0.005. Cyclic motion has radius 1 and period `budget // 4`; it follows the
  existing all-coordinates sinusoidal path. Jumps have magnitude 1 and interval
  `budget // 5`.
- Simplex targets move through a positive, normalized cyclic path. Matrix tasks
  use ambient dimension 6, rank 2 and the existing Cayley drift with scale 0.005.
  Initial matrix frames are independently sampled and feasible. They are not
  normalized to the Euclidean tasks' initial distance.

Smoke and standard profiles have different horizons and cycle/jump periods;
smoke results must not be pooled with standard results.

## Algorithms and compatibility

The catalog has 19 Euclidean methods, two simplex methods
(ExponentiatedGradient and ProjectedGradient), and RiemannianSGD for Stiefel
and Grassmann. Their formulas, defaults and sources are specified in
[the algorithm reference](algorithms_070.md). Incompatible pairings are marked
SKIPPED and spend no queries.

There are 449 compatible task-method configurations: **22450 evaluation runs**,
with an upper budget of **89.8 million queries**. The smoke profile attempts
1347 compatible runs. The evaluation split is seeds 10001–10050; development
uses 20001–20020. Development is diagnostic, not an automatically tuned track.
Any tuning must be declared separately; evaluation results cannot be used to
select parameters while retaining an untouched-evaluation claim.

Parameters remain at catalog defaults unless explicitly overridden. No tuning
is performed. In particular, an aggressive catalog learning rate can diverge on
the conditioned or Rosenbrock cases; the suite records that result instead of
silently adjusting the rate. This evaluates the supplied configuration, not the
best achievable performance of an algorithm family.

`--optimizer-config` accepts a JSON object mapping the selected algorithm names
to parameter dictionaries. The complete effective parameters enter the manifest.
Use disjoint development seeds when selecting parameters.
Old names are rejected rather than automatically migrated. Display labels
are not RNG inputs: catalog methods use explicit implementation IDs, while
custom methods supply a stable `rng_key`.

Custom algorithms can be supplied through the Python `Method` API:

```python
from pathlib import Path
from wind_benchmark.experiments.standard import Method, Profile, run_suite

method = Method(
    name="MyOptimizer",
    oracle_type="first-order",  # or zero-order / hybrid
    geometries=("euclidean",),
    factory=lambda task, seed: MyOptimizer(seed=seed),
    parameters={"description": "fixed parameters selected before evaluation"},
    implementation="my_optimizer_version_1",
    rng_key="my_optimizer_stream_v1",
)
run_suite(Path("results/default_pipeline/custom_001"), Profile(), [method])
```

The factory receives task metadata and an algorithm-only seed, never the hidden
trajectory. A fresh optimizer is constructed and initialized for every repetition.
Custom generators must use the supplied RNG in initialize(x0, rng, task_spec). Record custom implementation
versions yourself: the automatic source hashes cover WIND's source files.

## Time, information and resource accounting

The default suite uses **streaming semantics**. Every oracle call consumes one
unit of budget and advances the target once, including every call made by
`advance(client)`. The algorithm gets only an immutable `Observation` and,
when needed, a scoped query-only `OracleClient`. The client expires after the
update callback. Raw environments and evaluator records are not handed to algorithms.

Each queried position is scored, including exploratory ZO perturbations and
lookahead points. This is not a recommended-iterate or incumbent-only metric.
No extra final evaluation is free. A final incomplete multi-query update is
allowed: all observations up to the budget are scored, and no further query is
allowed. Frozen-round execution remains a separate experiment, not an implicit
alternative inside this suite.

FO is gradient-only; ZO is value-only. Hybrid calls provide both and increment
one total-call counter plus one value and one gradient counter. Different oracle
tracks are reported separately, without asserting equal computational cost of
a gradient and a function evaluation. Scheduled access and RL training are not
part of this version.

Evaluator outputs are published only after algorithm execution for the complete suite
finishes. Public algorithm-owned recommendations may be saved earlier; they contain no truth. This is API-level information isolation, not a security sandbox
against malicious Python introspection or filesystem access.

## Statistics and interpretation

- The independent unit is a **complete run**, not a time point, coordinate, or
  individual jump. Run seeds are independent; external random streams are paired
  across methods. Algorithm randomness has a separate named stream.
- Stationary tasks use tail mean squared tracking distance (last 20%) as the
  primary score. Moving tasks use mean squared tracking distance over the full
  budget. Matrix distances are squared frame Frobenius or squared Grassmann
  principal-angle distances, respectively.
- Clean excess loss, cumulative dynamic regret, feasibility and recovery
  measures are also exported. No noisy loss is substituted for evaluator truth.
- Scalar means have individual 95% percentile-bootstrap intervals: 10,000
  resamples for standard, 300 for smoke. Resampling acts on run scalars. It does
  not treat serially correlated time samples as independent.
- Comparisons use paired differences against explicitly fixed references:
  SGD for Euclidean FO, SPSA for Euclidean ZO, ProjectedGradient for simplex,
  and RiemannianSGD for manifolds. No comparison is generated for a lone method
  or an absent reference. Negative differences favor the named
  method. Bootstrap preserves matching by seed. Cross-oracle comparisons are
  deliberately not generated.
- Failed and infeasible runs remain in `runs.csv`. Accuracy summaries are
  explicitly conditional on success, and paired comparisons state how many
  unsuccessful pairs were excluded. These conditional comparisons can be
  selection-biased and must be read alongside the failure counts.
- Failure probabilities have Wilson intervals, including nonzero upper bounds
  when no failures are observed. Counts distinguish numerical failures,
  infeasibility and execution errors. No finite penalty is invented for failures.
- Numerical failure includes nonfinite arithmetic and decision norm above
  `1e12`. No hidden-state tracking-error cutoff is used during execution.
- Recovery requires distance at most 0.2 after a unit jump for
  `max(2, budget // 200)` consecutive observations. Time is measured through the
  confirming observation. The window is `budget // 10`. Unrecovered events
  receive that restricted time **and** a false recovery indicator; they are not
  described as actually recovering. Incomplete windows are flagged. Events are
  averaged within runs before bootstrap across runs.
- Curve plots show successful-run means and pointwise 95% percentile-bootstrap
  intervals, with up to 2000 resamples of complete runs. The same resampled run
  identities are used at every query. These are not simultaneous confidence bands.
  With a single successful run, no interval is drawn. Scalar summaries retain
  their 10,000 resamples in the standard profile.
- All successful-run query errors are retained, not only the sampled preview.
  Matched tasks share y limits within a figure. Figures contain at most four
  task panels, with dedicated full-resolution windows after jumps.

Fifty runs are a planning default, not proof of sufficient power. A separate
pilot can motivate a larger preregistered sample size. Changing sample size,
tasks or budgets creates a different resolved experiment, recorded in the
manifest. Do not stop when a desirable result becomes significant. The report
does not generate p-values, global rankings or claims of superiority across all
tasks; any later confirmatory multiple testing needs its own declared correction.

## Output files

| File | Contents |
|---|---|
| `report.md` | Human-readable primary results and failure/skip counts |
| `runs.csv` | Every attempted or skipped repetition, counts, status and scalar metrics |
| `summary.csv` | Primary means, bootstrap intervals and failure-rate intervals |
| `metrics_summary.csv` | Run-level uncertainty for secondary metrics, including recovery |
| `paired_comparisons.csv` | Paired differences within task/oracle, with excluded-pair counts |
| `curves.csv` | Sampled per-run error curves; these are not full latent trajectories |
| `trajectories.npz`, `trace_index.csv` | Full-resolution successful-run squared errors and their task/method/seed index |
| `curve_summary.csv` | Mean and pointwise confidence bounds at every query |
| `recovery_events.csv` | Per-jump outcomes and completeness flags |
| `manifest.json` | Resolved configuration, seeds, versions, source hashes and conventions |
| `recommendations/*.npz` | Separate public recommended positions and exact query indices; not used in queried-point scores |
| `execution.log` | Progress, failure reasons, and the required algorithm format for incompatible cases |
| `figures/` | Coverage, grouped curves, jump windows, failure probabilities, paired effects, recovery and scaling; PNG at 300 dpi, SVG and PDF |
| `figure_manifest.json` | Figure index, interval conventions and the renderer source hash (which may differ from the original run after a plotting update) |

The script does not save full hidden-state journals. Per-query records remain
inside the evaluator until summarized; error arrays and run summaries are
released after the suite. Re-running the recorded configuration reconstructs
the workload in the recorded software environment.

## Algorithm-format diagnostics

Each incompatible pairing is logged with the algorithm's declared geometries
and the required contract: `initialize(x0, rng, task_spec)` and `advance(client) -> AdvanceResult`,
output shape, observable fields and feasibility constraints. A missing interface
or wrong output shape produces `ERROR_INTERFACE`. Merely declaring compatibility
does not project decisions or repair an algorithm. The `required_format` column
in `runs.csv` preserves the contract for each task.

## Canonical command and validation

```powershell
wind-canonical --dry-run
wind-canonical --profile standard --output-dir results/wind_tracking_v1/run_001 --no-plots
python -m wind_benchmark.experiments.canonical results/wind_tracking_v1/run_001
wind-canonical --split development --output-dir local/internal_checks/development_001 --no-plots
```

The validator accepts new algorithms, checks the specification fingerprint,
task/split/budget completeness, statuses and query costs, and reconstructs
primary scores from full histories. Altered task sets, horizons, seeds or
bootstrap counts are explicitly labelled custom studies. Source versions and
resolved instance fingerprints accompany each result.

Threshold summaries report attainment fractions with Wilson intervals and
restricted query counts conditional on completed runs. Numerical failures are
not treated as full-budget censored observations. Runtime is elapsed execution
time. Memory columns report a 10 ms sampled process RSS peak estimate and
increase above the run's starting RSS; they include the interpreter and evaluator,
may miss shorter peaks, and are not isolated algorithm allocation measurements.

## Rebuild figures without rerunning algorithms

```powershell
python -m wind_benchmark.reporting.standard results/default_pipeline/run_001
```

This reads saved summaries and full error arrays, regenerates only the figures,
curve aggregates and figure index, and replaces the generated figure-links
section of the report. It does not execute algorithms or change run metrics.
Earlier reports without `trajectories.npz` cannot reconstruct lost time detail;
they require a new run. Coverage cells show successful / attempted runs, not
accuracy scores. Failure intervals are Wilson intervals. Raw paired differences
have task-local axes and must not be pooled across different metric scales.
Dimension panels use a linear y axis when the full displayed range spans less
than a decade; wider ranges use the nonnegative symmetric-log scale. Curve
panels retain the symmetric-log scale and explicitly shared limits.
