# Replicated resource allocation transfer

The task is a proxy for assigning a shared compute or bandwidth budget across
services with changing loads. Both agents use `ResourceAllocationEnv`: five
continuous features of public value/action history and `Discrete(12)` actions.
The actions combine four reallocation magnitudes with explore, repeat or reverse.
Q-learning discretizes the features; PPO receives their continuous values.
Neither agent receives the load vector, clean loss, gradient or evaluator.

The reward is `10*(-next_value + 2*(value-next_value) - .02*L1_reallocation)`.
Both values are public noisy measurements. Clean metrics are released only
after the complete comparison. A reset costs one value query, so an H-step
episode costs H+1 queries.

## Agents

Online Q-learning uses learning rate 0.1, discount 0.95, exploration probability
0.2, the fixed five-dimensional bins in `QLearning.bin_edges`, and initial
action value `-.8 - 2*magnitude`. There is no TD clipping. True termination
sets continuation to zero; time-limit truncation bootstraps from the final
public observation. See [Watkins and Dayan](https://doi.org/10.1007/BF00992698).

PPO is the external [Stable-Baselines3 2.9.2 implementation](https://pypi.org/project/stable-baselines3/2.9.2/),
not an in-house update. CPU, one environment, separate [64,64] Tanh actor/critic,
learning rate 0.0003, gamma 0.95, GAE lambda 0.95, rollout 1000, batch 100,
10 epochs, clip 0.2, entropy coefficient 0, value coefficient 0.5 and gradient
norm limit 0.5 are fixed before evaluation. Advantage normalization is enabled;
reward normalization, VecNormalize and value clipping are disabled.

The bounded VecEnv auto-resets between scheduled episodes but not after the
last one. It provides terminal observations and time-limit flags and prohibits
actions after the schedule is exhausted. Checkpoints contain complete rollouts,
so `learn` does not overspend the declared transition budget.

## Splits and budgets

Each algorithm has curriculum, stationary-only and no-pretraining conditions,
ten independent training seeds, and 30 common held-out workloads. Pretraining
has 2000 episodes of 160 steps in dimensions 4, 6 and 8. Adaptation uses dimension
12 and 200 episodes of 300 steps; evaluation checkpoints are 0, 10, 50, 100, 200.
Training dynamics cycle through stationary, linear, cyclic, jump and random walk;
the stationary-only condition changes just the dynamics. Training noise cycles
through none, Gaussian and correlated. Transfer uses mixed dynamics and clipped
heavy-tailed value corruption. No-pretraining consumes no pretraining interactions.

Initialization is matched across conditions within a training seed. Workloads
and exploratory service pairs use separate common streams; policy training has
its own stream. At transfer only Q values or actor/critic weights are carried
over. PPO optimizer state, rollout buffers and episode memory are reset. Optimizer
state is then preserved across adaptation checkpoints with constant schedules.

The complete design has 9000 evaluation episodes and 19201000 value queries:
12880000 for pretraining, 3612000 for adaptation, 2709000 for evaluation.
Evaluation uses deterministic `predict` on complete episodes, without learning,
checkpoint selection or statistic updates. Saved policies identify every training
replicate, condition and checkpoint.

## Parallel execution (package 0.7.0)

`run_experiment(profile, output_root, seed=20261112, workers=1)` and the RL CLI
accept a positive integer `workers` (Python booleans are rejected). The default
is sequential execution in the caller. For example:

```powershell
python -m wind_benchmark.experiments.rl --profile paper --workers 8 --output-dir results/rl_transfer_v3/NEW_RUN --no-plots
```

Add `--dry-run` to inspect the profile and requested/effective process counts
without creating files. Choose an unused output directory. There is no resume;
after an interrupted run, use a new directory. An API call with multiple workers
must be made from an importable script under `if __name__ == "__main__":`.

There are 60 independent jobs: algorithm × training seed × pretraining condition.
Each job keeps pretraining, transfer, adaptation checkpoints and their evaluation
in the original sequential order. PPO still uses one environment; neither its
rollout schedule nor its update rule is vectorized across jobs. Long-lived spawn
workers limit Torch and BLAS/OpenMP to one thread. Process count is execution
metadata, not a change to `rl_transfer_v3`, RNG streams, budgets or statistics.

The coordinator restores declared design order before aggregation and assigns
trace keys independently of completion order. Evaluation arrays remain private
until the entire series releases, including when an external EvaluationSession
is used. A worker never releases its own evaluator objects between jobs. Saved
policies and `execution_status.json` contain only policy/operational information;
the latter reports active phases, episode/query counts and process IDs. Final
CSV/NPZ data and a hashed manifest are exported only after successful completion.

Ctrl+C cancels on episode/transition boundaries and stops only owned workers
after a five-second grace period. Worker failure aborts the series without final
evaluation export; workers monitor their parent to avoid orphan computations.
Partial directories remain marked stopped/failed. An export error is reported
separately as `export_failed`, without undoing a legitimate series release.
Wall time and sampled aggregate RSS are measured under concurrent load; neither
is a per-policy intrinsic cost. Available memory and CPU determine useful
concurrency; linear speedup is not guaranteed.

## Statistical interpretation

The primary comparison is pretraining condition within each algorithm.
The 2000-resample crossed bootstrap independently resamples training seeds and
workloads and preserves pairing across conditions and checkpoints. Comparisons
between Q-learning and PPO are secondary. Workload-only intervals cannot support
claims about training variability. Absence of a curriculum benefit is a valid
outcome; this discrete-action example is neither continuous-control validation
nor a convergence theorem for a partially observed nonstationary process.
