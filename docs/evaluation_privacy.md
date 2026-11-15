# Ground-truth access and release

## Direct optimization

Live queries require `BenchmarkRunner`, `OracleSession`, or `WindGymEnv`.
Raw `Oracle.start_step()` and the evaluator record channel cannot be used as an
unprotected alternative from application code. Standalone core objects remain
available for offline mathematical analysis, not as a hidden-state experiment
runner. Internal oracle-unit tests explicitly instrument evaluator authority;
public access-denial tests never enable that test fixture.

`Oracle.query(x)` returns `Observation(x, t, value, grad, query_index, mode)`.
There is no `theta`, `clean_value`, or `optimum_value` field. The available
value/gradient channels still follow the configured FO, blind FO, ZO, hybrid,
scheduled, or replay contract. Arrays are copied and marked read-only.

The oracle computes a separate `EvaluationRecord` containing the observation,
the state actually used for that query, the clean loss, and the loss at the
optimum. The runner consumes this internal record before calling the algorithm.
It is never a return value of the algorithm-facing query API. In particular,
offline evaluation uses the replay state and replay landscape, not the dummy
environment's current state.

Each run evaluates metrics in a private copy of the configured metric collection.
Metric implementations must support `deepcopy`; the collection supplied at runner
construction is the template for subsequent runs. Metrics still run at the same
time in the interaction cycle, before the environment advances. Regret uses the
two clean loss values recorded for the same query. Every actual query is counted once. Update timing follows the explicitly
selected frozen or streaming protocol. Aggregation follows the
versioned [metrics contract](metrics_v2.md), rather than legacy scalar formulas.

During a run, reading `runner.trajectory`, `runner.metrics_history`, the metric
collection, retained environment/oracle state, or their serialized state raises
`GroundTruthAccessError`. The same rule applies to saved method references and
reads from another thread. `for_analysis=True` is NOT an override of a live
session lock. After all optimizer calls and result validation succeed, a
standalone run releases its result. With `record_trajectory=True`, `result.evaluation_log` contains one
record per oracle call, including auxiliary queries. JSON export includes these
records as numeric arrays. The existing trajectory format (an initial snapshot
followed by evaluated decisions) and metric history format are retained.
With recording disabled, metrics are still returned, but no evaluation log or
trajectory is retained in the result.

An exception propagates without publishing a partial result or journal. The
oracle lock is released and the step-scoped query client is disabled even if an
optimizer fails during a query phase. A new run clears the runner's previously
published results before invoking the optimizer's initialize method.

`BatchRunner.run` and `run_adaptive` lock results until the batch finishes;
progress messages contain no metric values. Adaptive replication may use metrics
internally for its stopping rule. CLI, Web, default, paper and RL suites defer
truth-bearing files until the enclosing series finishes; the legacy CLI no
longer prints intermediate Lyapunov values or writes metric checkpoints.

For a manually organized comparison, declare an outer series:

```python
from wind_benchmark import EvaluationSession

with EvaluationSession(expected_runs=2):
    first = runner.run(method_a, T=100, x0=x0, seed=17)
    # first.final_metrics, first.trajectory and first.to_dict() are locked here.
    second = runner.run(method_b, T=100, x0=x0, seed=17)

print(first.final_metrics, second.final_metrics)  # released together
```

The optional count refers to completed runner/query-session runs. Do not use
it as an episode count for Gym; Gym tracks completion of its episodes itself.
Releasing while a run/query session/episode is active is rejected. An explicit
early release cannot be followed by further runs in that session. Helper suites
return a `PendingResult` when invoked directly inside an outer series; call
`pending.result()` after leaving the context. Trace callbacks are deferred too.
Aborted series discard queued exports and keep partial objects locked.

### Multiple queries in one step

All algorithms use `initialize(x0, rng, task_spec)` and `advance(client)`.
For example, one symmetric pair requires exactly two calls, without an initial
evaluation at x:

```python
def advance(self, client):
    if client.remaining == 0:
        return AdvanceResult(self.x, False, 0)
    u = self.rng.choice((-1.0, 1.0), size=self.x.size)
    plus = client.query(self.x + self.h * u)
    if client.remaining == 0:
        return AdvanceResult(self.x, False, 1)
    minus = client.query(self.x - self.h * u)
    self.x = self.x - self.lr * (plus.value-minus.value) * u / (2*self.h)
    return AdvanceResult(self.x, True, 2)
```

The ordinary runner freezes the target within an update. The canonical suite
advances it after every query and scores both perturbation points. Recommended
iterates are stored separately. QueryEfficiency uses the actual index of its
declared decision stream and includes an accurate initial point at q=0.
An unfinished pair or batch consumes its actual calls but cannot produce a
completed update. The scoped client exposes only `query` and `remaining`;
it has no public environment, time control, or evaluator and expires after
the update callback.

### Compatibility and limits

Removing `Observation.optimum_value` is an API change. Code constructing an
observation must omit that argument. Arbitrary custom noise, drift, landscape,
oracle and metric callbacks are rejected in a protected live run: these functions
would receive clean inputs and could publish them. Use the built-in evaluator
components; custom metrics/analysis can consume released results offline.
Custom optimizers and RL policies remain supported and execute without evaluator
authority, including their initialization hooks, factories and array conversions.
Adding a new evaluator component requires an audited library implementation,
not registering an arbitrary user callback as privileged code.

The `local_series_v2` mechanism is a local Python API boundary, not an OS security
sandbox or a separate server. Ordinary access to guarded private attributes,
`vars()` and serialization is denied, but deliberately modifying library
internals, bypassing Python attribute dispatch, inspecting frames or using a
debugger is outside the guarantee. Strict
isolation requires a separate process and restricted memory, file, and seed
access. Known initial conditions/seeds and previously released replay data cannot
be made secret retroactively. This mechanism prevents runtime access, not
mathematical reconstruction from known task definitions or allowed feedback.
If noise is disabled, permitted feedback naturally equals the clean
quantity; an optimum value fixed at zero can also be known from the task definition.

## Gymnasium

The observation contains the decision and permitted noisy oracle channels. `info`
contains only step/query counters, geometry, and action constraint violation.
Neither contains the latent target, clean loss, optimum value, or tracking error.
The default `neg_value` reward is exactly the negative value visible in the
float32 observation; it may be positive when noise makes that value negative.
For blind gradient-only feedback, `reward="zero"` must be selected explicitly.
There is no clean-loss fallback. Old `neg_regret`/`neg_error` reward settings and
non-None `terminate_on_diverge` values raise an error. Episodes truncate at the
configured horizon and do not terminate based on hidden error.

The Gym environment keeps its core and oracle in private attributes. It exposes
no public `env`, `oracle`, ground-truth getter, or journal handle; `unwrapped`
returns the same restricted Gym environment. An outer standard Gym wrapper may
expose this Gym environment through its own `env`, but not the underlying WIND
core. After reset, retained references to that core and oracle are also locked,
not merely absent from `info`. The process-isolation limitation stated above
still applies; no OS security guarantee is claimed.

A trusted caller can create a `GymEvaluationSession` and pass it as
`evaluation_session` when constructing one or more environments. Do not give the
session, core environment, or raw oracle to the policy. During execution the
session stores each reset/step evaluation privately. `session.records` raises
until the whole series is finished, even between completed episodes. Calling
`finish()` (or leaving its context manager successfully) seals the session and
disables reset/step on all attached environments. Only then are records released
to the evaluator. The caller defines the series boundary and must include all
runs that share evaluation workloads or seeds.

Records contain an `EvaluationRecord`, exact geometry-aware error, constraint
violation, and step number. Step zero is a reset measurement and must be excluded
from sums over rewarded transitions. A failing step, explicit abort, premature
close/reset, or exception escaping the session context discards its journal.
Finishing with incomplete episodes also discards it and raises an error.
Without an attached session no ground-truth trajectory is retained for export.
Retained core references nevertheless stay locked until the environment is
closed; closing an unfinished episode aborts rather than releases it. With an
outer `EvaluationSession`, Gym records additionally wait for that outer series.

Aborted and released states are terminal. Repeated `Gym.close()` calls are
idempotent and cannot release an aborted episode. A failure while exporting a
completed series sets `EvaluationSession.export_status` to `failed`; it does not
change the already released computation into an aborted run. Re-run only the
export into a new destination after correcting the storage error.

The RL experiment still learns from the same noisy-value shaping and switching
cost. Training rollouts contain no true regrets/errors. All evaluation rollouts
share one session spanning the complete comparison; clean metrics are attached
and summarized only after all fine-tuning and evaluation calls finish. Old code
reading `info['theta']`, `info['error']`, or `environment.oracle` must migrate to
post-series records or the public query counters in `info`.

## Verification

The canonical process executor preserves the same outer-series boundary.
Workers retain private, unreleased evaluation sessions for their entire lifetime,
including gaps between assignments; their objects are never released locally.
Evaluator payloads return only to the trusted coordinator over private IPC.
No public future or checkpoint exposes intermediate scores or latent states.
Custom factory imports and exception formatters run without evaluator authority,
including error conversion during algorithm initialization.
Only public recommendations and operational progress can be written before
release. An outer `EvaluationSession` continues to hold all canonical exports.
Cancelled jobs, worker crashes and coordinator loss do not release partial data.
This remains supported-API isolation, not protection against memory inspection,
source modification or reconstructing a known seeded task.

`tests/test_parallel.py` checks worker reuse, retained references, serialization,
unprivileged factory imports, nested sessions, cancellation, process death,
export failure and deterministic serial/parallel results.

`tests/test_session_privacy.py` checks retained aliases and bound methods,
threaded reads, serialization, evaluator authority, callback rejection,
multi-run release, deferred exports, failure handling and Gym array hooks.

`tests/test_evaluation_privacy.py` compares complete histories and final metrics
with the pre-change evaluation loop on fixed linear, random-walk, and jump paths,
including noisy observations and raw/path-normalized regret. It also checks all
oracle response schemas, deferred publication, failure handling, repeated runs,
multi-query state consistency/accounting, offline evaluation, and batch output.
Gym tests cover sanitized observations/rewards/info, public attribute access,
wrapper unwrapping, session sealing, failures, geometry, and the Gymnasium checker.
Version 0.6 adds independent algorithm formula tests, strict query budgets,
full-history exports and external PPO checks. The RL test verifies exact rollout
costs, terminal observations, no final automatic reset and policy roundtrips.
