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

## Run an experiment

Copy `experiment.example.json`, edit the grid, then run:

```powershell
wind-benchmark --config experiment.example.json
```

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
  "output_dir": "results_smoke",
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

## Tests

```powershell
python -m pytest
black --check .
```

Tests cover environment invariants, information barriers, temporal consistency,
reset reproducibility, batch aggregation, Gymnasium compliance and Stiefel
geometry.

## License

MIT. See [LICENSE](LICENSE).
