# Paper tracking v4: development selection, not evaluation tuning

Memory uses fixed configurations: SGD/HeavyBall/Nesterov lr=.05 (momentum .9
where applicable), Adam/AMSGrad lr=.02, AdaGrad lr=.1 and eps=1e−8.
Calibration, geometry and RL are not expanded.

Zeroth and dimension compare nine ZO configurations, plus the separate fixed
SGD first-order reference (lr=.04). Each ZO configuration/dimension receives
12 candidates × five development seeds 9001–9005 × 2000 oracle calls:
120,000 maximum calls. Numerical failures consume their actual calls; no dummy
calls or success-only retries. Three spherical batch sizes are three separate
configurations, not one family with a shared tuning allowance.

Let C={.05,.1,.2,.4}, R={.03,.1,.3}. Grids are fixed before evaluation:

| Configuration | Twelve candidates |
|---|---|
| SPSA | alpha in C, beta in R |
| AcceleratedSPSA | h in C, beta in R; a=.1, delta=1.05 |
| SphericalZOSGD m=1,5,20 | lr=c min(1,m/d), c in C, h in R |
| OnePointSPSA | lr=c/d, c in C, r in R |
| ZOSignSGD | lr=c/sqrt(d), c in C, r in R; m=1 |
| KieferWolfowitz | lr in {.01,.02,.04,.08}, h in R |
| RandomSearch | sigma in {.003,.005,.01,.02,.03,.05,.1,.2,.3,.5,1,2} |

Selection uses a stationary quadratic with condition number 5, value-noise
sigma=.001 and frozen updates. The score averages recommended-iterate squared
error over every integer query tick q=0,...,1999, then over five independent
seeds. All five trials must complete with finite scores. Exact ties use numeric
lexicographic parameter order. No eligible candidate means TUNING_FAILED,
null selected parameters and no evaluation trajectory; there is no default
fallback. Candidates and their per-seed scores/statuses/costs are retained.

Selected parameters are fixed across all evaluation cases and frozen/streaming
conditions at a dimension. Evaluation seeds are 1001–1030, budget 10,000 calls;
dimension uses d=5,10,20,50,100,200. Scalar bootstrap has 2000 resamples.
Task/noise streams are shared across candidates and methods; estimator streams
are separate and fixed within a configuration. RandomSearch proposal and order
streams are independent. Grid ordering and presentation labels are not seeds.

When zeroth and dimension are executed together, matching d=20 tuning is reused
inside the protected evaluation series using its full configuration fingerprint.
Only the first use is charged: six dimensions require at most 6.48 million
tuning calls and 3240 tuning trajectories. Separate process invocations do not
silently import previous selections; they disclose their own actual costs.

The smoke profile retains all twelve candidates but reduces seeds and horizons;
its selections never become paper selections. Cache fingerprints distinguish
profiles and budgets. No intermediate tuning truth is exported before release.
Tuning tables are exported as CSV and strict JSON after the whole series.

RandomSearch gradient diagnostics are not applicable. Other cosine diagnostics
compare the estimate at its actual center with the clean gradient at the
post-update query tick; in streaming this is a temporal diagnostic, not an
unbiasedness claim. Paper recommended-iterate and canonical queried-point
scores remain separate. Failed and tuning-failed configurations remain visible.

## Commands (full runs require separate approval)

```powershell
python -m wind_benchmark.experiments.paper --profile smoke --studies all --no-plots --output-dir local/internal_checks/paper_v4 --run-name check_001
python -m wind_benchmark.experiments.paper --profile paper --studies all --no-plots --output-dir results/paper_tracking_v4 --run-name reference_070_001
wind-canonical --profile standard --workers 8 --no-plots --output-dir results/wind_tracking_v1/reference_070_001
```

Canonical uses catalog defaults for every method, with no in-suite tuning.
Paper grids are design choices, not claims of literature-optimal parameters.
