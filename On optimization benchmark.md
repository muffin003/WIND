# WIND: A Benchmark for Tracking in Non-Stationary Optimization

## 1. Motivation

Classical optimization benchmarks ask how quickly an algorithm finds a fixed
minimum. Many real systems are different: market conditions change, sensors
drift, recommendation policies evolve, and tracked subspaces rotate. An
algorithm with excellent static convergence can perform poorly when it must
continuously follow a moving target.

WIND addresses the following question:

> How well can an algorithm track a hidden moving optimum observed only through
> a noisy oracle, and how can that performance be evaluated rigorously?

The benchmark makes problem difficulty controllable, separates algorithms from
the environment, prevents privileged access to the optimum, and connects
empirical measurements to stabilization and dynamic-regret concepts.

## 2. Core protocol: a clean environment and a hidden target

WIND treats every optimizer as an external black box. The benchmark exposes a
single oracle interface that returns noisy values, gradients, or both. The
environment does not need to know whether the client is SGD, Adam, SPSA,
CMA-ES, or a newly implemented method.

The current optimum, denoted by $\theta_t$, is hidden from the optimizer. It is
used internally to evolve the environment and compute evaluation metrics, but
it is never included in the optimizer's observation. This information barrier
turns the task into genuine tracking rather than fitting a known point.

### 2.1 Interaction cycle

Each time step follows a fixed order:

1. The optimizer commits to $x_t$ before observing the current loss.
2. `start_step(t)` locks the environment at $\theta_t$.
3. One or more calls to `query(x)` observe the same locked state.
4. `end_step()` releases the lock.
5. Metrics are evaluated against the privileged ground truth.
6. The environment advances according to
   $\theta_{t+1}=\mathcal{D}(\theta_t,t,x_t)$.

This protocol is non-anticipative and temporally consistent. Multiple
zero-order queries made during one step cannot accidentally observe different
versions of the objective.

### 2.2 Observation contract

All public feedback is stored in an immutable `Observation` object.

| Field | Meaning | Available to the decision rule |
| --- | --- | --- |
| `x` | Committed point $x_t$ | Yes |
| `t` | Time index | Yes |
| `value` | Noisy function value, if enabled | Yes |
| `grad` | Noisy gradient, if enabled | Yes |
| `optimum_value` | Exact value at the optimum | Metrics only |
| `query_index` | Oracle-query counter | Yes |
| `mode` | Oracle-mode identifier | Yes |

The object rejects fields such as `theta` or `optimal_point`, and array values
are copied to prevent indirect mutation of environment state.

### 2.3 Oracle modes

| Mode | Feedback | Typical use |
| --- | --- | --- |
| First-order | Gradient; value hidden by default | SGD, Adam, RDA |
| Zero-order | Function value only | SPSA, finite differences, CMA-ES |
| Hybrid | Value and gradient simultaneously | Mixed-feedback algorithms |
| Scheduled | Alternating first- and zero-order phases | Intermittent sensors or expensive gradients |
| Offline | Pre-recorded reproducible sequence | Validation and published comparisons |

Zero-order algorithms must construct their own gradient estimates. Their query
cost is recorded explicitly. A hybrid oracle is different from a scheduled
oracle: hybrid feedback is simultaneous, while scheduled feedback changes over
time.

## 3. Three independent axes of controlled difficulty

WIND decomposes difficulty into three interchangeable components:

- drift controls how the optimum moves;
- landscape controls local and global objective geometry;
- noise controls how feedback is corrupted.

Changing one component while keeping the others fixed supports interpretable
stress tests and ablation studies.

### 3.1 Drift models

| Model | Evolution | Interpretation |
| --- | --- | --- |
| Stationary | $\theta_{t+1}=\theta_t$ | Static-convergence baseline |
| Linear | $\theta_{t+1}=\theta_t+v$ | Predictable trend |
| Random walk | $\theta_{t+1}=\theta_t+\mathcal{N}(0,\sigma_d^2I)$ | Unpredictable motion |
| Cyclic | $\theta_t=c+A\sin(2\pi t/T)$ | Seasonal behavior |
| Jump | Periodic displacement of fixed magnitude | Regime change and recovery |
| Adaptive | Motion toward or away from $x_t$ | Pursuit/evasion dynamics |
| Sparse | Only $k$ coordinates move | Sparse concept drift |
| Stiefel | $\Theta_{t+1}=Q_t\Theta_t$ | Subspace tracking |

Randomized drift objects own independent NumPy generators. Calling `reset()`
restores their original generator state so an experiment can be replayed.

### 3.2 Objective landscapes

Every landscape is parameterized by the current optimum and satisfies the core
invariants

$$
L_t(\theta_t)=0,\qquad \nabla L_t(\theta_t)=0,\qquad L_t(x)\geq 0.
$$

| Family | Representative form | Main difficulty |
| --- | --- | --- |
| $p$-norm | $\frac1p\|M_\kappa(x-\theta_t)\|_p^p$ | Hölder geometry and conditioning |
| Quadratic | $\frac12(x-\theta_t)^TA(x-\theta_t)$ | Controlled condition number |
| Rosenbrock | Shifted Rosenbrock valley | Curvature and non-convexity |
| Multi-extremal | Shifted Rastrigin-type objective | Local extrema |
| Robust | Huber-type loss | Outlier-resistant geometry |
| Simplex | Squared distance on $\Delta^{d-1}$ | Constrained optimization |
| Stiefel | $\|X-\Theta_t\|_F^2$ | Manifold constraint |

The Rosenbrock and multi-extremal definitions move with $\theta_t`; they do not
retain a fixed optimum while the nominal environment drifts.

### 3.3 Noise models

| Model | Purpose |
| --- | --- |
| Gaussian | Standard finite-variance corruption |
| Heavy-tailed | Rare large errors and potentially infinite variance |
| Correlated | AR-like temporal dependence |
| Quantized | Limited sensor or communication precision |
| Multiplicative | Error proportional to signal magnitude |
| Sparse | Corruption of a subset of coordinates |

Value and gradient noise can be configured independently. Stateful noise models
also restore their generator and internal state during reset.

## 4. Evaluation principles

### 4.1 Stabilization

For a Hölder exponent $\rho\in(0,1]$, WIND uses the Lyapunov quantity

$$
V_t(x)=\|x-\theta_t\|_{\rho+1}^{\rho+1}.
$$

An algorithm is empirically stable when this quantity remains bounded over the
tested horizon. The tail of the trajectory is used to estimate steady-state
behavior rather than reporting a potentially unrepresentative final point.

### 4.2 Dynamic regret

The instantaneous regret is

$$
r_t=L_t(x_t)-L_t(\theta_t),
$$

and cumulative dynamic regret is $R_T=\sum_{t=1}^T r_t$. Because the comparator
uses the same locked $\theta_t$ as the oracle response, the metric is temporally
consistent.

### 4.3 Drift-normalized quantities

When the drift magnitude is $A$, normalized Lyapunov values make regimes with
different movement speeds easier to compare. The raw measurement is always
retained so normalization does not conceal absolute error.

### 4.4 Query efficiency

Iteration counts are not directly comparable between first- and zero-order
methods. WIND records total, value, and gradient queries so results can be
reported against both time steps and oracle cost.

### 4.5 Recovery and adaptivity

Jump environments support time-to-recovery measurements. Adaptivity metrics
compare the recovery time of an algorithm with a reference oracle response and
are bounded to a documented range.

## 5. Metric catalogue

The implementation includes:

- tracking error in configurable norms;
- maximum coordinate error;
- Lyapunov and normalized Lyapunov metrics;
- asymptotic-bound estimates;
- instantaneous loss;
- dynamic regret;
- time to recovery;
- drift adaptation and adaptivity;
- query efficiency;
- Stiefel geodesic distance.

`MetricsCollection` validates compatible settings and exposes both per-step
histories and tail-aggregated results.

## 6. Reference algorithms and execution

The reference suite contains 25 optimizers: 12 first-order and 13 zero-order
methods. They are baselines and protocol-validation tools, not claims that every
implementation is a state-of-the-art production solver.

First-order examples include SGD, Polyak averaging, Heavy Ball, Nesterov, Adam,
AdamW, AMSGrad, stochastic mirror descent, regularized dual averaging,
proximal SGD, adaptive learning rate, and SignSGD.

Zero-order examples include random search, one-point SPSA, central finite
differences, FDSA, SPSA, Gaussian-smoothing ZO-SGD, ZO-SignSGD, quadratic
interpolation, Kiefer-Wolfowitz, a Nedic-style subgradient method, accelerated
SPSA, CMA-ES, and GP-UCB.

### 6.1 Reproducible configuration

Experiments are controlled by JSON:

```json
{
  "output_dir": "results_lyapunov",
  "seeds": [42, 43, 44, 45, 46],
  "steps": 500,
  "rho_values": [1.0, 0.5, 0.2],
  "drift_values": [0.001, 0.01, 0.1, 0.3, 0.6, 1.0],
  "dimensions": [5],
  "optimizers": null
}
```

Run the suite with:

```bash
wind-benchmark --config experiment.example.json
```

Use `--dry-run` to inspect the resolved configuration without starting the
experiment.

### 6.2 Randomness and replay

The seed controls the initial point, optimizer-level NumPy randomness, drift,
landscape generation, oracle state, and observation noise. Initial points use a
local generator:

```python
rng = np.random.default_rng(seed)
x0 = rng.normal(0.0, 0.1, size=dim)
```

Each result stores the seed, optimizer signature, environment configuration,
runtime, and metric history. Publication-grade replay should additionally pin
the code revision and dependency versions.

### 6.3 Statistical execution

`BatchRunner` executes independent seeds and aggregates metrics. Adaptive
execution continues until the relative half-width of a 95% confidence interval
falls below a requested tolerance or the maximum number of runs is reached.

## 7. Scope and limitations

WIND evaluates consequences of theoretical assumptions; it does not prove that
every configured landscape and noise process satisfies those assumptions. A
user may deliberately select heavy-tailed noise, strong dependence, or extreme
drift to create an out-of-theory stress test. Publications should label such
regimes explicitly.

Recommended parameter ranges are therefore soft. Only physically or
mathematically invalid values are rejected. This design preserves the ability
to study controlled assumption violations.

Reference implementations prioritize transparent protocol behavior. A
comparison intended to rank algorithms must document tuning budgets, equalize
oracle costs where appropriate, and avoid selecting hyperparameters on the
reported test grid.

## 8. Extensions beyond the Euclidean core

### 8.1 Optional Gymnasium adapter

The optional adapter expresses WIND as a partially observed Markov decision
process:

| RL element | WIND interpretation |
| --- | --- |
| Action | Absolute point $x_t$ or increment $\Delta x_t$ |
| Hidden state | $\theta_t$ and internal drift/noise state |
| Observation | Position and oracle feedback, never $\theta_t$ |
| Reward | Negative regret or negative tracking error |
| Transition | $\theta_{t+1}=\mathcal{D}(\theta_t,t,x_t)$ |
| Episode | Truncation after $T$ steps |

Gymnasium is not required for the optimization benchmark. It is installed only
through the optional `gym` dependency group.

### 8.2 Stiefel-manifold optimization

For $X,\Theta_t\in\operatorname{Stiefel}(d,r)$, valid points satisfy
$X^TX=I_r$. A Euclidean update generally leaves the manifold, so WIND provides
tangent projection, retraction, Cayley motion, and geodesic metrics.

The Riemannian gradient is

$$
\operatorname{grad}_R L
=\nabla L-X\,\operatorname{sym}(X^T\nabla L),
$$

and retraction maps a tangent step back to the manifold. This extension is
reported separately because the Euclidean Hölder and stabilization theory does
not automatically transfer to curved decision spaces.

## 9. Software validation

The automated tests cover:

- zero loss and zero gradient at the moving optimum;
- non-anticipative locking and temporal consistency;
- enforcement of the information barrier;
- deterministic replay after reset;
- seed-level reproducibility;
- JSON export and batch aggregation;
- oracle-mode semantics;
- Gymnasium API compliance when the optional dependency is installed;
- Stiefel orthogonality, projection, retraction, and tracking.

Together, these checks validate the benchmark protocol and implementation
invariants. They do not replace scientific validation of a particular optimizer
or experimental claim.
