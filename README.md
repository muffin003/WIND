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

## Architecture

The installable package is `wind_benchmark` and its source is in `src/`.

| Module | Responsibility |
| --- | --- |
| `core.py` | Dynamic environment, drifts, landscapes and noise models |
| `oracle.py` | First-order, zero-order, hybrid, scheduled and offline oracles |
| `benchmark.py` | Single-run and multi-seed runners, result export |
| `metrics.py` | Tracking error, Lyapunov metrics, regret and adaptation metrics |
| `experiment.py` | 25 reference optimizers and the full experiment suite |
| `gym_env.py` | Optional Gymnasium/POMDP adapter |
| `visualization.py` | Metric, comparison and trajectory plots |
| `manifold.py` | Stiefel-manifold helpers and Riemannian SGD |

The main suite currently contains 25 optimizers: 12 first-order and 13
zero-order methods.

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
python -m wind_benchmark.web
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
| `output_dir` | Result directory |
| `seeds` | Independent reproducibility seeds |
| `steps` | Steps in each run |
| `rho_values` | Hölder exponents |
| `drift_values` | Drift magnitudes `A` |
| `dimensions` | Search-space dimensions |
| `optimizers` | Selected optimizer names, or `null` for all 25 |

For a quick smoke test, use one value in every grid and a small optimizer list:

```json
{
  "output_dir": "results/smoke",
  "seeds": [42],
  "steps": 20,
  "rho_values": [1.0],
  "drift_values": [0.01],
  "dimensions": [5],
  "optimizers": ["SGD", "SPSA"]
}
```

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

Custom optimizers follow `OptimizerProtocol`; no `BaseOptimizer` inheritance is
required:

```python
import numpy as np
from wind_benchmark.benchmark import OptimizerProtocol
from wind_benchmark.oracle import Observation


class MyOptimizer(OptimizerProtocol):
    name = "MyOptimizer"
    oracle_type = "first-order"

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def reset(self) -> None:
        pass

    def step(self, observation: Observation) -> np.ndarray:
        if observation.grad is None:
            raise ValueError("MyOptimizer requires a gradient")
        return observation.x - self.lr * observation.grad
```

## Gymnasium and constrained actions

The optional Gymnasium adapter exposes the same dynamic task to RL policies.
Install the `gym` extra, then wrap a configured environment and oracle:

```python
from wind_benchmark.gym_env import WindGymEnv

gym_env = WindGymEnv(
    environment,
    oracle=oracle,
    action_mode="delta",
    geometry="auto",
    reward="neg_regret",
)
observation, info = gym_env.reset(seed=42)
```

`geometry="auto"` preserves the original clipped `Box` actions for Euclidean
landscapes. Simplex actions are projected onto the probability simplex. For a
`StiefelLandscape`, absolute actions are projected to an orthonormal frame and
delta actions are projected to the tangent space and retracted. A
`GrassmannLandscape` uses the same feasible representatives but defines a
separate, basis-invariant subspace-tracking task: `X` and `X @ Q` are equivalent
for orthogonal `Q`. Accordingly, `neg_error` uses Frobenius frame distance for
Stiefel tasks and principal-angle distance for Grassmann tasks. Oracle modes are
unchanged: first-order, zero-order, hybrid, scheduled and offline replay remain
available through the same core.

The curriculum-to-transfer example treats a simplex decision as the allocation
of a shared compute or bandwidth budget among services. A dependency-free
Q-learning controller receives value-only feedback: the noisy scalar operating
cost, changes in that cost, and its own previous reallocations. It never receives
a gradient, the latent load vector, or Gym `info`. The agent learns whether to
explore a new service-to-service transfer, repeat the last transfer, or reverse
it, and also chooses the amount to reallocate. Training covers stationary,
linear, cyclic, jump, and random-walk loads before transfer to a larger held-out
mixed workload. The comparison contains only three versions of the same RL
agent: curriculum-pretrained, stationary-pretrained, and trained from scratch on
the target task. Its learning reward combines the next observed cost with a
value-difference term and an $\ell_1$ switching cost. Both cost
values come from the zero-order oracle, while clean regret is retained only for
evaluation.

```powershell
python -m wind_benchmark.expRL --profile smoke
python -m wind_benchmark.expRL --profile paper
```

Run-level evaluations, learning curves, learned Q tables, a manifest, and the
summary figure are written below `results/rl_transfer_experiment/`.

## Reproducibility

Each result stores its seed and complete optimizer/environment metadata. The
experiment uses a local NumPy generator for `x0`, while drift and noise objects
restore their initial RNG states on reset. Repeating the same configuration,
seed and dependency versions therefore reproduces a run.

For publication-grade archival, retain:

- the experiment JSON;
- `experiment_metadata.json` and individual result JSON files;
- the Git commit;
- Python and dependency versions.

Run outputs are stored under `results/` and are intentionally included in the
repository. Manuscript drafts, reference papers, notebooks, and machine-local
files belong under the Git-ignored `local/` directory.

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
