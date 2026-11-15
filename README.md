![CI: Linter](https://github.com/muffin003/WIND/actions/workflows/lint.yml/badge.svg)
![CI: Tests](https://github.com/muffin003/WIND/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# WIND Benchmark

WIND is a modular benchmark for stochastic optimization in non-stationary
environments. It compares how well optimization algorithms track a hidden,
moving optimum under different landscapes, drift processes, noise models and
information constraints.

At time `t`, an optimizer commits to `x_t`, queries an oracle, and only then the
environment advances from `theta_t` to `theta_(t+1)`. The runner records the
privileged ground truth for analysis without exposing it to the optimizer.

Direct optimizer observations contain no ground-truth fields. Evaluation records
and metric histories are released after a successful run (after the complete
batch when using `BatchRunner`). See [ground-truth access and release](docs/evaluation_privacy.md)
for the API changes, multi-query client, and deferred Gymnasium evaluation.
Live environment/oracle attributes and serialization are also locked; the
`for_analysis=True` flag cannot bypass a running series. Use `EvaluationSession`
to group manually organized runs into one release boundary. This is a local API
guarantee, not protection against modifying library internals or using a debugger.

## Requirements and installation

WIND supports **Python 3.11 or newer**. Do not copy or commit a virtual
environment; recreate it from the project metadata:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gym,dev]"
```

On Linux or macOS, activate it with `source .venv/bin/activate`.

The `gym` extra installs the optional Gymnasium adapter. The `dev` extra installs
pytest and Black. `requirements.txt` remains available for tools that cannot
install a `pyproject.toml` project.

Wheel installations include the web interface. When using a wheel, start
`wind-benchmark-ui` from a writable working directory: experiment outputs are
stored there, not inside `site-packages`. Source and editable installations keep
their existing repository-relative output paths.

See the [0.7.0 release notes](docs/releases/0.7.0.md) and
[metric definitions and migration](docs/metrics_v2.md) before comparing new
scores with historical results.

## Standard algorithm evaluation

`src/experiments/standard.py` provides a separate, versioned default task pipeline. It
tests reference problems, landscapes, observation corruption, target dynamics,
selected interactions, dimension scaling and compatible constrained geometries.
It does not replace the legacy CLI or the existing publication experiments.

```powershell
python -m wind_benchmark.experiments.standard --profile smoke --output-dir local/internal_checks/default_pipeline/check_001
wind-canonical --profile standard --output-dir results/wind_tracking_v1/run_001 --no-plots
```

The standard profile uses 50 independent paired repetitions and 4000 oracle calls
per compatible run. Parameters are fixed, not automatically tuned. Reports retain
failures and incompatible-task skips, and separate FO/ZO/hybrid comparisons.
Reports include full-resolution curves with run-level bootstrap intervals,
coverage and failure panels, paired effects, and post-jump detail in PNG/SVG/PDF.
Incompatible cases state the required algorithm interface and geometry in the log.
Saved figures can be rebuilt without rerunning algorithms:

```powershell
python -m wind_benchmark.reporting.standard results/default_pipeline/run_001
```

See [standard-run tasks, statistics and outputs](docs/default_run.md).

## Architecture

### Publication experiments

`src/experiments/paper.py` specifies the paper's workloads and statistical analyses. Its
empirical runs use `DynamicEnvironment`, `FirstOrderOracle` / `ZeroOrderOracle`,
and the reusable `OracleSession` in `src/core/execution.py`. FO optimizers come from the
canonical catalog with the paper's explicit parameters, not catalog defaults.
All 22 updates, including paired ZO and geometry-aware algorithms, come from
`src/algorithms/optimizers.py` through the same catalog. Explicit raw/projected geometry
controls compose these algorithms without entering the canonical competition. The analytic calibration recurrence stays separate.

```powershell
python -m wind_benchmark.experiments.paper --profile paper --studies all --output-dir results/paper_tracking_v4 --run-name run_001 --no-plots
```

The paper's frozen protocol holds the target during an update and subsequently
advances it by the number of consumed queries. Streaming advances after every
query. Both use the oracle's counters. Incomplete batches consume actual queries but do not complete an update;
actual and unused calls are exported. Protocol `paper_tracking_v4` measures the
recommended iterate at every integer query tick: the mean uses `q=0,...,B-1`,
the terminal error uses `q=B`, and the tail uses the last 20% of those B samples.
Rendering checkpoints do not affect metrics or hyperparameter selection. Value
noise and estimator directions use independent named streams, never display
labels. This score still differs from the standard run's score at every queried
perturbation point. Historical numbers use the frozen pre-audit implementation,
not this v4 protocol; do not silently replace archived results.

Only public observations or a scoped query-only client reach the algorithms.
Evaluator records are released after each run closes, and result files are
published after the complete selected suite finishes. Source hashes, effective
memory-optimizer parameters and execution conventions are recorded in the
manifest. This is API-level isolation, not a security sandbox.

Regression tests include fixed pre-migration numerical fixtures. Private
before/after validation runs and reports belong under the Git-ignored `local/`
directory, not among the published experimental results.

The installable package is `wind_benchmark` and its source is in `src/`.

| Package directory | Responsibility |
| --- | --- |
| `src/core/` | Environment, oracles, runners, metrics, geometry, isolation and runtime support |
| `src/algorithms/` | Public initialize/advance protocol, 22 implementations and catalog |
| `src/experiments/` | Standard suite, specification/validator, parallel executor, paper/RL studies and configurable runs |
| `src/integrations/` | Optional Gymnasium environment, deferred evaluation and RL task wrapper |
| `src/interfaces/` | CLI and local web application |
| `src/reporting/` | Export helpers, canonical plots and interactive visualizations |

Only `__init__.py` and `__main__.py` remain at the source package root.
The root-level public API (`from wind_benchmark import BenchmarkRunner`, etc.)
and console commands `wind-benchmark`, `wind-canonical`, and `wind-benchmark-ui`
are unchanged. Module-level imports now follow the directories, for example
`wind_benchmark.algorithms.catalog` and `wind_benchmark.core.environment`.
Use `python -m wind_benchmark.experiments.standard`, `.paper`, or `.rl` for
the separate experiment entry points. No duplicate flat-module shims are kept.
Source fingerprints in new manifests cover the entire nested package using
relative paths; historical manifests/results are not rewritten.

The catalog contains 22 algorithms: 11 Euclidean first-order, eight Euclidean
zero-order, and three geometry-aware first-order methods.

## Benchmark workbench

### Start locally after downloading the repository

Open PowerShell in the downloaded or cloned repository, then create the local
environment, install WIND and start the workbench:

```powershell
cd "path\to\WIND"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gym,dev]"
python -m wind_benchmark.interfaces.web
```

The browser opens at `http://127.0.0.1:8765`. Keep the PowerShell window open
while using the benchmark and press `Ctrl+C` to stop it. On later starts, only
activate the existing environment and run the final module command.

After reinstalling the project entry points, the shorter equivalent command is
`wind-benchmark-ui`.

The workbench exposes environments, landscapes and drifts, oracle and noise
models, the full optimizer catalog, runner settings, metrics, result history,
analysis tools and Gymnasium integration as separate sections. The current
configuration remains visible in a fixed JSON inspector while you work. English
is the default language; Russian and Chinese are available from the sidebar.
The local interface can launch runs through the WIND engine.

The same `web/` directory is also a self-contained interactive handbook. On a
static host such as GitHub Pages it provides the complete reference, formulas,
configuration editor, JSON download and local JSON/CSV analysis. Starting new
benchmark runs and browsing server-side result history remain available only
when the Python engine is connected.

The legacy parameter-grid command remains available for reproducible full-suite
experiments. Copy `experiment.example.json`, edit the grid, then run:

```powershell
wind-benchmark --config experiment.example.json
```

The web server uses only the Python standard library and does not add runtime
dependencies to the package. Opening [`web/index.html`](web/index.html) directly
still shows the interface, but launching benchmarks and reading result history
requires `wind-benchmark-ui` so the page can reach the local WIND engine.

The equivalent module command is:

```powershell
python -m wind_benchmark --config experiment.example.json
```

Resolve and inspect a configuration without starting calculations:

```powershell
wind-benchmark --config experiment.example.json --dry-run
```

Configuration fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Must be 3; historical configurations are not migrated |
| `output_dir` | Result directory |
| `seeds` | Independent reproducibility seeds |
| `steps` | Steps in each run |
| `rho_values` | Hölder exponents |
| `drift_values` | Drift magnitudes `A` |
| `dimensions` | Search-space dimensions |
| `optimizers` | Selected optimizer names, or `null` for all 22 |

For a quick smoke test, use one value in every grid and a small optimizer list:

```json
{
  "schema_version": 3,
  "output_dir": "local/internal_checks/smoke",
  "seeds": [42],
  "steps": 20,
  "rho_values": [1.0],
  "drift_values": [0.01],
  "dimensions": [5],
  "optimizers": ["SGD", "SPSA"]
}
```

### Source-specified algorithms

The active catalog has no historical algorithm aliases or optimizer re-exports.
Nesterov queries a lookahead point; AdamW uses decoupled decay; SPSA uses fresh
symmetric Rademacher pairs. Retired implementations remain in frozen archives,
not under names of different algorithms. See [formulas, sources and tests](docs/algorithms_070.md).

## Python API

Create components from configuration dictionaries:

```python
from wind_benchmark import BenchmarkRunner, FirstOrderOracle, make_environment

config = {
    "dim": 5,
    "drift": {"type": "random_walk", "sigma": 0.02},
    "landscape": {"type": "quadratic", "condition_number": 10},
    "x_bounds": [-10, 10],
}

environment = make_environment(config, seed=42)
oracle = FirstOrderOracle(environment, seed=42)
```

Custom algorithms implement `initialize(x0, rng, task_spec)` and
`advance(client)`. The task specification contains only public geometry;
the client exposes queries and remaining budget, not the evaluator.

```python
import numpy as np
from wind_benchmark.algorithms.protocol import AdvanceResult

class MyOptimizer:
    name = "MyOptimizer"
    oracle_type = "first-order"
    query_cost_per_step = 1
    geometries = ("euclidean",)

    def initialize(self, x0, rng, task_spec):
        self.x = np.array(x0, dtype=float, copy=True)

    def advance(self, client):
        if client.remaining == 0:
            return AdvanceResult(self.x, False, 0)
        observation = client.query(self.x)
        self.x = self.x - 0.01 * observation.grad
        return AdvanceResult(self.x, True, 1)
```

There is no mandatory initial query. Queries, completed updates and
recommendations are separate counters. Ordinary runner `T` counts updates
(frozen target within an update); the canonical suite counts streaming queries.

## Gymnasium and constrained actions

The optional Gymnasium adapter exposes the same dynamic task to RL policies.
Install the `gym` extra, then wrap a configured environment and oracle:

```python
from wind_benchmark.integrations.gym_env import WindGymEnv

gym_env = WindGymEnv(
    environment,
    oracle=oracle,
    action_mode="delta",
    geometry="auto",
    reward="neg_value",  # use "zero" for a blind gradient-only oracle
)
observation, info = gym_env.reset(seed=42)
```

`geometry="auto"` preserves the original clipped `Box` actions for Euclidean
landscapes. Simplex actions are projected onto the probability simplex. For a
`StiefelLandscape`, absolute actions are projected to an orthonormal frame and
delta actions are projected to the tangent space and retracted. A
`GrassmannLandscape` uses the same feasible representatives but defines a
separate, basis-invariant subspace-tracking task: `X` and `X @ Q` are equivalent
for orthogonal `Q`. Evaluation uses Frobenius frame distance for
Stiefel tasks and principal-angle distance for Grassmann tasks. Oracle modes are
unchanged: first-order, zero-order, hybrid, scheduled and offline replay remain
available through the same core.

The default reward is the negative noisy value present in the observation.
Blind gradient-only tasks must explicitly use `reward="zero"`. Clean rewards
(`neg_regret`, `neg_error`) and hidden-error termination are rejected. `info`
contains only public counters and action-feasibility diagnostics; `env`, `oracle`,
and ground-truth accessors are not exposed, including through `unwrapped`.
To obtain clean metrics, create a trusted `GymEvaluationSession`, pass it at
construction, and read its `records` only after the entire series finishes:

```python
from wind_benchmark.integrations.gym_evaluation import GymEvaluationSession

with GymEvaluationSession() as evaluation:
    gym_env = WindGymEnv(environment, oracle=oracle, T=200,
                         reward="neg_value", evaluation_session=evaluation)
    # Give only gym_env to the policy. Run all episodes to their horizon here.
    observation, info = gym_env.reset()
    for _ in range(gym_env.T):
        observation, reward, terminated, truncated, info = gym_env.step(
            gym_env.action_space.sample()
        )
records = evaluation.records  # includes a reset record followed by each step
```

Sealing the session disables further interaction with its environments. An
incomplete or failed series is discarded. This is an API boundary within Python,
not process-level isolation against malicious introspection.

The resource-allocation transfer experiment uses standard online Q-learning and
external Stable-Baselines3 2.9.2 PPO. Both receive the same five public history
features and 12 discrete actions. Each has curriculum, stationary-only and
no-pretraining conditions, with ten independent training seeds and 30 shared
held-out workloads. Crossed bootstrap resamples both training seeds and workloads.

```powershell
python -m pip install -e ".[rl,dev]"
python -m wind_benchmark.experiments.rl --profile smoke --output-dir local/internal_checks/rl_check_001
python -m wind_benchmark.experiments.rl --profile paper --workers 8 --output-dir results/rl_transfer_v3/run_001 --no-plots
```

The full protocol produces 9000 evaluation episodes. Policies, full evaluation
histories, aggregates, interaction costs and a manifest are saved without
article figures. See [RL protocol](docs/rl_transfer_v3.md).

## Reproducibility

Version 0.7.0 uses `wind_tracking_v1`, `paper_tracking_v4`,
`rl_transfer_v3`, configuration schema 3 and export schema 4. The corrected
`metrics_v2` and `local_series_v2` contracts are retained.
See [metric definitions](docs/metrics_v2.md). Historical artifacts are not
converted, relabelled, or recalculated automatically; use their frozen code.

Reproduce a result using its source version, protocol, effective parameters,
component configuration, seeds and dependency versions, not a seed alone.
Display labels do not select implementations or random streams. Built-in
environment exports include a JSON-safe `reconstruction` snapshot with the
landscape realization, initial state and both oracle noise channels;
`EnvironmentConfig.restore()` reconstructs these components after release.
Custom evaluator callbacks are not accepted in protected live runs; perform
custom analysis offline on released results. Custom algorithms remain supported.

`BenchmarkRunner.run(seed=...)` seeds the algorithm, drift and noise; the
configured landscape/initial state are fixed task data. Pass
`reseed_components=False` to replay constructor streams from an exported task.
Batch factories should accept `seed`; old signatures use a built-in reseeding
fallback. A seed-aware environment factory owns its landscape realization;
the runner does not regenerate its matrix. Algorithms must use the generator supplied to `initialize`.
The default suite accepts stable `rng_key` identities for custom methods.
Presentation names and method order do not define random streams.

In Gymnasium, an explicit reset seed controls drift/noise episode streams;
subsequent unseeded resets draw new streams. The first unseeded reset honors
construction-time streams. `reset(options={"replay": True})` explicitly replays
the current episode's initial streams. Fixed task geometry/initial state and
evaluation-only information remain unchanged.

For publication-grade archival, retain:

- the experiment JSON;
- `experiment_metadata.json` and individual result JSON files;
- the Git commit;
- Python and dependency versions.

The [0.7.0 reproduction kit](reproducibility/0.7.0/README.md) records the
dependencies, source fingerprints and commands for the canonical, paper and RL
experiments. It distinguishes the two source builds used for those experiments.
The new full result arrays, trajectories and trained policies are not included
in this Git publication; they remain local. Previously tracked historical
results are retained and must not be interpreted as 0.7.0 reference scores.
Manuscript drafts, reference papers, notebooks, and machine-local files belong
under the Git-ignored `local/` directory.

## Tests

```powershell
python -m pytest
black --check .
```

Tests cover environment invariants, information barriers, temporal consistency,
reset reproducibility, batch aggregation, Gymnasium compliance, and both Stiefel
and Grassmann geometry. See [`docs/testing.md`](docs/testing.md) for the detailed
test catalogue.

## License

MIT. See [LICENSE](LICENSE).
