# WIND metric definitions: metrics_v2

WIND 0.6.0 retains the `metrics_v2` definitions and `local_series_v2` release
protocol introduced in 0.5.0; new exports use schema 3. The metric correction in
0.5.0 did not change optimizer trajectories. The separate 0.6.0 algorithm and
task-generator corrections intentionally change some trajectories and require
new experiments; historical results are not relabelled.

## Result contract

`final_metrics`, `metrics_history`, `metric_definitions` and `metric_status` use
matching output keys. Definitions contain a full SHA-256 identity derived from
the metric class, exact parameters and definition version, plus formula,
geometry, units, direction and aggregation. `result.metric_value(metric_id)`
resolves a scalar without relying on its label. Labels are not identities. Repeated
identities are rejected. Distinct configurations sharing a display name use
their full IDs as output keys. Normalized Lyapunov collections also retain raw V.

For stepwise statistics, `tail_fraction` must be finite and in `(0,1]`; the tail
starts at `floor((1-gamma)*T)` and is selected once. Cumulative, event and ensemble
statistics have their own scope. Non-estimable Python scalars are NaN internally;
JSON represents them as `null` alongside a status and observation counts. CSV
uses empty cells, not zero, and exports each actual metric history. Derived
instantaneous regret is a history, not an alternative cumulative score.

| Metric | Definition and default scalar |
|---|---|
| TrackingError | Tail mean of Lp distance. Dimension normalization divides by `d**(1/p)`; L-infinity is unchanged. Mahalanobis is `sqrt(e.T @ A @ e)`, optionally divided by `sqrt(d)`. A must be finite, symmetric and positive definite, explicitly supplied or obtained from the quadratic task. |
| MaxCoordinateError | Tail mean of `max(abs(e))`. Use `aggregation='max'` explicitly for the temporal peak. |
| StiefelFrame | Frobenius distance between oriented frames. It can diagnose raw infeasible iterates. |
| GrassmannPrincipalAngle | L2 norm of principal angles. Inputs must be orthonormal; no implicit projection is performed. |
| StiefelGeodesic | Numerical distance for the canonical Riemannian metric, using a matrix-logarithm iteration, endpoint residual and forward/reverse agreement. It is no longer a Grassmann alias. |
| Lyapunov / NormalizedLyapunov | `V=norm(e,rho+1)**(rho+1)` and `V/A**(rho+1)`, with finite `0<rho<=1` and `A>0`. Normalization alone does not imply invariance to drift speed. |
| InstantaneousLoss | Tail mean of clean `L(x_t;theta_t)`. |
| DynamicRegret | Full sum of clean `L(x_t;theta_t)-L(theta_t;theta_t)`. Path normalization is an additional result, never a replacement. |
| DriftAdaptation | Mean cosine of decision and target displacements over valid tail observations. The first step and zero displacement pairs are undefined. |

Path-normalized regret divides by `P=sum(norm(theta[t]-theta[t-1],2))` for
`t=1,...,T-1`; a stationary path produces `null/zero_path_variation`. Undefined
cosines are excluded with their count reported. Other undefined stepwise values
are not silently dropped by the default mean.

Numerical Stiefel estimates are not certificates of shortest paths for arbitrary
distant points. Branch ambiguity, disconnected square-frame components and
nonconvergence produce unavailable values and solver diagnostics, never a
Grassmann or Frobenius substitute. The mathematical limitation is discussed by
[Mataigne, Zimmermann and Miolane](https://arxiv.org/html/2403.11730v2).
The historical low-level `manifold.geodesic_distance` helper remains explicitly
documented as a subspace compatibility helper; use `canonical_stiefel_distance`
for the canonical numerical solver.

## Queries to accuracy

`QueryEfficiencyMetric(epsilon=0.1)` measures the first query index with task
distance at most epsilon: smaller is better. An accurate initial decision gives
zero. The ordinary runner scores committed points only. A committed point seen
at query 4 remains a hit at 4 even if two auxiliary queries follow. All calls
still count toward the recorded budget.

A miss has `state='right_censored'`, its observed budget B, and restricted time
`min(Q_epsilon,B)`. Compare restricted means only at equal budgets and identical
decision observation protocols, together with the reach fraction. A mean over
successful hits alone is explicitly conditional and is not the primary score.
Queried points and recommended iterates are different protocols; the ordinary
runner rejects a metric configured for another stream.

Paper ZO uses recommended decisions at every integer query tick. Mean squared
error is averaged over `q=0,...,B-1`; the terminal error is at B. The plot grid
does not change this calculation. Threshold tests include zero, use radius 0.1
(squared error 0.01), configurable as `ZeroOrderCase.threshold_epsilon`, and report valid observations, early numerical failures,
restricted query time and censoring. Early failures may have different available
budgets, in which case a common-budget restricted mean is not reported.

## Recovery events

A jump closes the previous recovery window before opening the new one. At jump
time tau, the event window ends at the next jump or horizon. First hits include
`TTR=0`; unrecovered events retain the full window and a censor flag. The library
reports mean capped TTR and recovery fraction; no jumps means `no_events`, not
zero. Adaptivity is 1 for immediate recovery and `min(1,oracle_ttr/TTR)` otherwise,
averaged conditionally on completed recoveries with censor counts alongside it.

The paper memory study keeps its median capped TTR within each run and mean of
those medians across seeds. Its threshold remains `max(0.2*jump_size,3*sigma*sqrt(d))`.
The standard suite retains its different consecutive-confirmation rule; its
manifest records confirmation count and window, and event files record time to
confirmation. These protocols must not be pooled as if they were identical.

## Ensemble asymptotic estimate

An individual `AsymptoticBoundMetric` records V but returns no scalar bound.
After the series releases, `BatchRunner.summarize(results)` computes
`max_tail(mean_seeds(V_t))`. It requires at least two distinct non-null seeds,
successful complete trajectories, the same actual task, optimizer parameters,
horizon, tail and metric definitions. Randomized task matrices that differ are
not silently pooled; use a fixed task across replicates. To keep a fixed
landscape in BatchRunner, supply a seed-aware environment factory that constructs
the same landscape; the runner still reseeds the dynamic and observation streams.
If a declared seed is missing or the series is incompatible, the result is
`incomplete_ensemble`. The estimate records N, T, gamma and
`finite_horizon_estimate`, not a proven asymptotic upper bound.

Histories must match the declared observation grid (`metadata.time_grid.size`),
not the number of optimizer updates (`metadata.total_steps`). In the ordinary
runner, V is recorded at every queried point: two SPSA updates therefore supply
four samples. Ensemble `T` is this sample count; `gamma` selects the tail of
those samples once, after averaging seeds at matching query indices. A missing,
truncated, non-finite or misaligned history yields `incomplete_ensemble`, not an
estimate over a silently shortened horizon. Historical histories without a
declared grid retain the step-indexed `total_steps` convention. This aggregation
does not change the separate canonical or paper scoring protocols.

## Compatibility

Read old files as historical records using their source version and metadata.
Do not automatically reinterpret their scores as metrics_v2. Old incorrect
formulas are reproducible through the preserved source snapshot, not through
active classes with correct names. `BatchRunner.run` still returns a list.
Version 0.6.0 has no historical optimizer aliases or re-exports. Use the current
catalog and the initialize/advance contract; old configurations require their
archived source snapshot. The ambiguous `manifold.geodesic_distance` helper is
also retired. Use `principal_angle_distance` for Grassmann subspaces and
`canonical_stiefel_distance` for the numerical canonical Stiefel distance.
The separate CLI workflows remain available.

`Gym.close()` is idempotent; abort cannot become release. Scheduled oracle reset
restores schedule position, within-segment count and current mode. The ordinary
runner restores the caller's NumPy and Python RNG states even on exceptions.
The environment keeps a non-owning registration of its oracle, avoiding a cycle
that retained large prescribed paths until cyclic garbage collection. Keep an
explicit oracle reference when using it independently; environment reset still
resets a live registered oracle, and configuration capture still reads it.
Exports refuse to overwrite result files. A storage error after successful
release is an export failure, not an aborted computation.
