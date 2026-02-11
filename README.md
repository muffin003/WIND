![CI: Linter](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/lint.yml/badge.svg)
![CI: Tests](https://github.com/muffin003/Stochastic-Optimization-Benchmark/actions/workflows/test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Below is a complete technical guide (README) for running and configuring **ST-HRD benchmark scenarios**.

This document is intended for researchers who want to test their optimization algorithms.

---

# 📘 Guide to Running ST-HRD Scenarios

**ST-HRD (Stochastic Tracking under Hölder Regularity & Drift)** is a modular benchmark controlled via configuration dictionaries. You don’t need to modify the environment source code to switch from Gaussian-noise testing to testing on the Rosenbrock function — simply change the `cfg` configuration parameters.

## 🛠 1. Installation and Dependencies

The code is written for Python 3.7+. The following libraries are required:

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

## ⚙️ 2. Control Panel (Config Dictionary)

The core of any scenario is the configuration dictionary. It defines the “physics” of the world in which the algorithm operates.

| Parameter | Type | Range | Description |
| :--- | :--- | :--- | :--- |
| **`dim`** | `int` | $2 \dots 100+$ | Dimension of the search space ($d$). |
| **`rho`** | `float` | $0.05 \dots 1.0$ | Hölder exponent. <br>• `0.1`: Sharp needle (hard). <br>• `1.0`: Smooth bowl (standard). |
| **`drift_speed`** | `float` | $0.0 \dots 0.1$ | Speed of optimum drift ($A$). <br>• `0.0`: Stationary problem. <br>• `0.1`: Very fast drift. |
| **`noise_type`** | `str` | `'gaussian'`, `'pareto'` | Gradient noise distribution type. |
| **`noise_scale`** | `float` | $0.0 \dots 3.0$ | Noise strength multiplier. |
| **`geometry`** | `str` | `'ideal'`, `'distorted'`, `'rosenbrock'` | Function topology. |
| **`condition_number`** | `float` | $1.0 \dots 1000.0$ | Condition number (only for `'distorted'`). Valley elongation level. |

---

## 🚀 3. Running Scenarios

Below are ready-to-use templates for five major research scenario types.

### Scenario A: Stability Topology (Mapping Landscape)

**Goal:** Build a 3D surface and heatmap showing under which combinations of smoothness ($\rho$) and drift speed ($A$) the algorithm can track the target.

```python
# 1. Define research grid
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
            'type': 'heatmap', 'rho': r, 'drift': d, 
            'success': success, 'errors': errors
        })
```

---

### Scenario B: Heavy-Tail Stress Test (Pareto Noise)

**Goal:** Test robustness to rare but extremely large gradient outliers (Pareto distribution, $\alpha=2.5$).

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
    MyOptimizer, best_hp, cfg_pareto, min_runs=100, max_runs=1000, tol=0.05
)

print(f"Pareto Stability: {succ:.1%}")
```

---

### Scenario C: Geometric Hell (Ill-Conditioned Valley)

**Goal:** Test whether the algorithm can descend a narrow rotating valley without oscillations.

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

### Scenario D: Rosenbrock Function (Tracking on Rosenbrock)

**Goal:** Test on the classic non-convex “banana” function adapted to drift tracking.

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

### Scenario E: Scalability (High Dimensions)

**Goal:** Evaluate performance degradation as dimensionality $D$ grows.

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
    results_dim.append({'dim': d, 'error': np.median(errs)})
```

---

## 📥 4. Adding Your Own Algorithm

To test your method, create a class inheriting from `BaseOptimizer`.

```python
class MyCustomOptimizer(BaseOptimizer):
    def step(self, x, oracle_data):
        """
        x: np.array — current coordinates
        oracle_data: dict — {'value': float, 'grad': np.array}
        """
        
        grad = oracle_data.get('grad')
        lr = self.hp.get('lr', 0.01)
        
        x_new = x - lr * grad
        return x_new
```

Then pass this class to `HyperTuner` and `run_adaptive_experiment`.

---

## 📊 5. Reading Reports (Visualizer)

The `BenchmarkVisualizer` class generates a dashboard. How to interpret plots:

1. **3D Error Topology:**  
   If the surface rises sharply at low $\rho$ and high drift — normal.  
   If it rises everywhere — the algorithm is poor.

2. **Robustness Radar:**  
   Larger polygon area = more universal algorithm.  
   Shift toward “Pareto” → robust to outliers.  
   Shift toward “Valley” → handles preconditioning well (Momentum/Adam).

3. **Degradation Curve:**  
   Ideal = flat line. Realistic = gentle increase.  
   Exponential growth → unsuitable for deep learning (`cond` > 1000).

4. **LLN Convergence:**  
   Mean curve should plateau. Continued oscillations → unreliable results  
   (increase `max_runs`).