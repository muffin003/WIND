# WIND Benchmark Test Suite

## Purpose
This suite validates the correctness and reliability of WIND across dynamic,
constrained, and manifold-valued optimization tasks. The tests ensure that:

- Tested mathematical identities agree with the declared definitions
- The interaction between optimizers and dynamic environments follows strict temporal consistency
- No information about optimal points leaks to the optimizer during execution
- Metrics are computed according to their theoretical definitions
- Results are fully reproducible across independent runs
- Exported results contain complete information for verification

## Running Tests
To execute the test suite:

```bash
# Install dependencies
pip install -e ".[gym,dev]"

# Run all tests
python -m pytest -v
```

The tests should pass completely with the current implementation.

## Test Coverage

The test suite includes comprehensive validation of all core components:

| **Component**                     | **Tested Functionality**                                                                 | **Scientific Purpose**                                                                 |
|----------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **Landscape Invariants**         | `L(θ, θ) = 0` for all landscapes<br>`∇L(θ, θ) = 0`                                       | Ensures valid dynamic regret (`f_t(θ_t) = 0`) and first-order optimality               |
| **Temporal Consistency**         | Identical `θ_t` across multiple queries in same step<br>Environment locking during step   | Enforces non-anticipative interaction (Protocol A); prevents lookahead                 |
| **Information Barrier**          | No truth fields in `Observation`<br>Read-only query points<br>Deferred evaluator records<br>Blind-value mode | Verifies the algorithm-facing API; not process isolation |
| **Lyapunov Metrics**             | Correct `p = ρ + 1` norm; algebraic drift normalization; ensemble mean before tail maximum | Validates finite-horizon computations, not a convergence theorem |
| **Dynamic Regret**               | Evaluator-only `optimum_value` computed with the same `θ_t` as `f_t(x_t)`                | Ensures mathematically correct regret calculation                                     |
| **Benchmark Runner**             | End-to-end execution<br>Trajectory recording<br>Metric aggregation                       | Validates full pipeline from setup to result collection                               |
| **Batch Runner**                 | Multi-seed execution<br>Statistical aggregation (mean/std)                               | Enables significance testing across random seeds                                      |
| **Offline Oracle**               | Deterministic replay of recorded `θ_t` sequence                                          | Provides reproducible environment evolution for fair algorithm comparison              |
| **Error Handling**               | Consistent `ρ` validation<br>Protocol violation detection                                | Prevents misconfiguration that would invalidate theoretical claims                    |
| **Reproducibility & Export**     | JSON/CSV export with full signatures<br>Ground truth inclusion<br>Seed-based reproducibility | Meets scientific publishing standards (Pineau et al., 2021)                           |

This coverage ensures the benchmark adheres to theoretical foundations while maintaining engineering robustness for research use.

## Version 0.5 metric and lifecycle regressions

`tests/test_metrics_v2.py` uses hand-computable vectors and trajectories to test
norm scaling, SPD validation, raw regret, zero path variation, first hits at
query zero, auxiliary-query accounting, censored recovery, complete-ensemble
requirements, and resets. Known small canonical Stiefel geodesics test numerical
length, symmetry and endpoint agreement; invalid representatives and failed
logarithm iterations are not replaced by another distance.

The same tests cover export with no trajectory, actual loss/regret CSV histories,
strict JSON nulls, metric identity collisions, caller RNG preservation,
scheduled-mode replay, idempotent Gym close, failed constructors/transitions,
and storage failure after successful evaluation release. The catalog import
test prevents accidental loading of the legacy experiment orchestrator.

The CI matrix covers Linux and Windows on Python 3.11 and 3.12, each with
core+dev, Gym+dev and RL+dev dependencies. Gym-specific tests skip when absent;
core tests must still run. Formatting uses the project's declared dev dependency.
Numerical tests do not establish asymptotic convergence or global minimality of
every numerical Stiefel logarithm.

## Versions 0.6–0.7 algorithms and protocols

`tests/test_algorithms_060.py` and `tests/test_algorithms_070.py` independently expand finite-step updates for all
22 active methods, including moment corrections, lookahead query locations,
Polyak--Ruppert recommendations, regularizers, direction normalization, pair
costs and feasible geometric updates. The source-to-formula mapping and
tolerances are specified in [algorithms_070.md](algorithms_070.md).

`tests/test_protocol_060.py` checks the 32-task design, disjoint seed splits,
22,450 compatible reference runs, export schema 4 (historical schema 3 remains readable), exact count types and RDA's
zero start. `tests/test_rl_060.py` checks Q-learning's unclipped TD update,
the value-only reward independently, actual SB3 rollouts without an extra final
reset, terminal observations, policy serialization, optimizer reset at transfer,
optimizer preservation across checkpoints, crossed resampling, and caller RNG
restoration. An RL extra is needed to execute the external-PPO tests.

CI configuration is not evidence that every platform has been executed locally.
Release verification reports identify the actual Python, OS, dependencies,
executed tests, skipped optional tests and numerical runs.
