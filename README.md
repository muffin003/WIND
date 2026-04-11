![CI: Linter](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/lint.yml/badge.svg)
![CI: Tests](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

#  WIND Benchmark

**WIND** is a modular benchmark for stochastic optimization in non-stationary environments, controlled via configuration dictionaries. You don't need to modify the environment source code to switch from Gaussian-noise testing to testing on the Rosenbrock function — simply change the `cfg` configuration parameters.

---

##  Table of Contents

- [Installation](#-1-installation-and-dependencies)
- [Configuration](#-2-control-panel-config-dictionary)
- [Running Scenarios](#-3-running-scenarios)
- [Adding Your Algorithm](#-4-adding-your-own-algorithm)
- [Visualization](#-5-reading-reports-visualizer)
- [License](#-license)

---

## 1. Installation and Dependencies

The code is written for **Python 3.7+**. Required libraries:

```bash
pip install numpy matplotlib seaborn pandas scipy tqdm
```

If you are working in **Google Colab**, add this at the start of your notebook:

```python
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
```

---

## 2. Control Panel (Config Dictionary)

The core of any scenario is the configuration dictionary. It defines the **physics** of the world in which the algorithm operates.

| Parameter | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| **`dim`** | `int` | 2 to 100+ | Dimension of the search space (d). |
| **`rho`** | `float` | 0.05 to 1.0 | Hölder exponent. |
| **`drift_speed`** | `float` | 0.0 to 0.1 | Speed of optimum drift (A). |
| **`noise_type`** | `str` | 'gaussian', 'pareto' | Gradient noise distribution type. |
| **`noise_scale`** | `float` | 0.0 to 3.0 | Noise strength multiplier. |
| **`geometry`** | `str` | 'ideal', 'distorted', 'rosenbrock' | Function topology. |
| **`condition_number`** | `float` | 1.0 to 1000.0 | Condition number (for 'distorted'). |

---

## 3. Running Scenarios

###  Scenario A: Stability Topology

```python
rho_grid = [0.2, 0.6, 1.0]
drift_grid = [0.0, 0.05, 0.10]

results = []

for d in drift_grid:
    for r in rho_grid:
        cfg = {
            'dim': 2,
            'rho': r,
            'drift_speed': d,
            'noise_type': 'gaussian',
            'noise_scale': 0.5,
            'geometry': 'ideal'
        }

        best_hp = tuner.tune(cfg)

        errors, success, traj = run_adaptive_experiment(
            MyOptimizer, best_hp, cfg, min_runs=30, tol=0.15
        )

        results.append({
            'type': 'heatmap',
            'rho': r,
            'drift': d,
            'success': success,
            'errors': errors
        })
```

---

### Scenario B: Pareto Noise

```python
cfg_pareto = {
    'dim': 2,
    'rho': 0.5,
    'drift_speed': 0.02,
    'noise_type': 'pareto',
    'noise_scale': 1.0,
    'geometry': 'ideal'
}

best_hp = tuner.tune(cfg_pareto)

errs, succ, _ = run_adaptive_experiment(
    MyOptimizer, best_hp, cfg_pareto,
    min_runs=100, max_runs=1000, tol=0.05
)

print(f"Pareto Stability: {succ:.1%}")
```

---

### Scenario C: Distorted Valley

```python
cfg_valley = {
    'dim': 2,
    'rho': 1.0,
    'drift_speed': 0.02,
    'noise_type': 'gaussian',
    'noise_scale': 0.1,
    'geometry': 'distorted',
    'condition_number': 100.0
}

best_hp = tuner.tune(cfg_valley)
errs, succ, _ = run_adaptive_experiment(MyOptimizer, best_hp, cfg_valley)
```

---

### Scenario D: Rosenbrock

```python
cfg_rosen = {
    'dim': 2,
    'rho': 1.0,
    'drift_speed': 0.01,
    'noise_type': 'gaussian',
    'noise_scale': 0.1,
    'geometry': 'rosenbrock'
}

best_hp = tuner.tune(cfg_rosen)
errs, succ, traj = run_adaptive_experiment(MyOptimizer, best_hp, cfg_rosen)
```

---

### Scenario E: Scalability

```python
dims = [2, 10, 50, 100]
results_dim = []

for d in dims:
    cfg_dim = {
        'dim': d,
        'rho': 0.5,
        'drift_speed': 0.05,
        'noise_type': 'gaussian',
        'geometry': 'ideal'
    }

    best_hp = tuner.tune(cfg_dim)
    errs, _, _ = run_adaptive_experiment(MyOptimizer, best_hp, cfg_dim)

    results_dim.append({
        'dim': d,
        'error': np.median(errs)
    })
```

---

## 4. Adding Your Own Algorithm

```python
class MyCustomOptimizer(BaseOptimizer):
    def step(self, x, oracle_data):
        grad = oracle_data.get('grad')
        lr = self.hp.get('lr', 0.01)
        return x - lr * grad
```

---

## 5. Visualization

Use `BenchmarkVisualizer` to generate plots and dashboards.

---