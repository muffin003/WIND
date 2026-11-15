"""Gymnasium curriculum and transfer experiment for WIND.

The experiment trains tabular Q-learning and external SB3 PPO from value-only
feedback.  At every step the controller chooses a reallocation magnitude and one
of three dimension-invariant maneuvers: explore a new transfer between services,
repeat the last transfer, or reverse it.  ``WindGymEnv`` applies the resulting
delta action on the probability simplex.  The controller observes only the noisy
scalar cost and its own action/reward history, never a gradient, the latent
optimum, or the Gym ``info`` dictionary.

The applied proxy is adaptive compute-resource allocation: a simplex point gives
the fractions of a shared CPU/GPU or bandwidth budget assigned to services whose
loads change over time.  The scientific question is whether curriculum pretraining
on controlled WIND dynamics improves zero-shot performance and fine-tuning
efficiency on a held-out mixed workload with more services.  The implementation
requires the optional RL extra with Stable-Baselines3 2.9.2 and Torch.

Run a quick verification with::

    python -m wind_benchmark.experiments.rl --profile smoke --output-dir local/tmp/rl_smoke

Use ``--profile paper`` for the publication schedule. An explicit, unused output
directory is required; no article figures are generated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from wind_benchmark.reporting.common import (
    write_json,
    safe_print as print,
    source_hashes,
)
import hashlib
import importlib.metadata
import platform
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wind_benchmark import __version__
from wind_benchmark.core.privacy import (
    _series,
    _user_call,
    _publish,
    _after_release,
    _library_caller,
    _current,
)
from wind_benchmark.experiments.parallel import (
    Execution,
    execution_options,
    check_cancelled,
    report_progress,
)
from threadpoolctl import threadpool_limits
import pandas as pd

from wind_benchmark.core.environment import (
    CorrelatedNoise,
    Drift,
    DynamicEnvironment,
    GaussianNoise,
    HeavyTailedNoise,
    SimplexLandscape,
)
from wind_benchmark.integrations.gym_env import WindGymEnv
from wind_benchmark.integrations.gym_evaluation import GymEvaluationSession
from wind_benchmark.core.oracle import ZeroOrderOracle
from wind_benchmark.core.randomness import stream_seed

# Explicit protocol identities, independent of presentation labels.
ALGORITHM_RNG_IDS = {"QLearning": 1001, "PPO": 1002}
CONDITION_RNG_IDS = {"curriculum": 101, "stationary_only": 202, "no_pretraining": 303}

MANEUVERS: Tuple[str, ...] = ("explore", "repeat", "reverse")
CONTROL_ACTIONS: Tuple[Tuple[float, str], ...] = tuple(
    (reallocation, maneuver)
    for reallocation in (0.02, 0.06, 0.15, 0.30)
    for maneuver in MANEUVERS
)
SWITCHING_COST = 0.02

FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "observed_loss",
    "relative_improvement",
    "value_surprise",
    "previous_reallocation",
    "success_balance",
    "stagnation",
    "episode_progress",
)

TRAIN_SCENARIOS: Tuple[str, ...] = (
    "stationary",
    "linear",
    "cyclic",
    "jump",
    "random_walk",
)


@dataclass(frozen=True)
class RLProfile:
    """Execution schedule for the transfer experiment."""

    name: str
    pretrain_episodes: int
    fine_tune_episodes: int
    train_horizon: int
    transfer_horizon: int
    evaluation_seeds: int
    fine_tune_checkpoints: Tuple[int, ...]
    train_dimensions: Tuple[int, ...]
    transfer_dimension: int
    training_seeds: int = 10


PROFILES: Dict[str, RLProfile] = {
    "smoke": RLProfile(
        name="smoke",
        pretrain_episodes=10,
        fine_tune_episodes=20,
        train_horizon=100,
        transfer_horizon=100,
        evaluation_seeds=2,
        fine_tune_checkpoints=(0, 10, 20),
        train_dimensions=(4, 6, 8),
        transfer_dimension=12,
        training_seeds=2,
    ),
    "paper": RLProfile(
        name="paper",
        pretrain_episodes=2000,
        fine_tune_episodes=200,
        train_horizon=160,
        transfer_horizon=300,
        evaluation_seeds=30,
        fine_tune_checkpoints=(0, 10, 50, 100, 200),
        train_dimensions=(4, 6, 8),
        transfer_dimension=12,
    ),
}


class SimplexPathDrift(Drift):
    """Replay a deterministic, feasible simplex path as a WIND drift."""

    def __init__(self, path: np.ndarray):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 2:
            raise ValueError("path must have shape (horizon + 1, dimension)")
        if np.min(path) < -1e-12 or not np.allclose(path.sum(axis=1), 1.0):
            raise ValueError("every path point must lie on the probability simplex")
        self.path = path.copy()

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        index = min(t + 1, self.path.shape[0] - 1)
        return self.path[index].copy()

    def reset(self) -> None:
        return None


def _dirichlet_target(rng: np.random.Generator, dim: int) -> np.ndarray:
    return rng.dirichlet(np.full(dim, 1.5))


def build_simplex_path(scenario: str, dim: int, horizon: int, seed: int) -> np.ndarray:
    """Construct a feasible target path for one curriculum regime."""

    if dim < 2:
        raise ValueError("dim must be at least 2")
    if horizon < 4:
        raise ValueError("horizon must be at least 4")

    rng = np.random.default_rng(seed)
    count = horizon + 1
    path = np.empty((count, dim), dtype=float)
    target_a = _dirichlet_target(rng, dim)

    if scenario == "stationary":
        path[:] = target_a
    elif scenario == "linear":
        target_b = _dirichlet_target(rng, dim)
        weights = np.linspace(0.0, 1.0, count)
        path[:] = (1.0 - weights[:, None]) * target_a + weights[:, None] * target_b
    elif scenario == "cyclic":
        target_b = _dirichlet_target(rng, dim)
        phase = np.linspace(0.0, 4.0 * np.pi, count)
        weights = 0.5 * (1.0 - np.cos(phase))
        path[:] = (1.0 - weights[:, None]) * target_a + weights[:, None] * target_b
    elif scenario == "jump":
        interval = max(4, horizon // 4)
        current = target_a
        for index in range(count):
            if index > 0 and index % interval == 0:
                current = _dirichlet_target(rng, dim)
            path[index] = current
    elif scenario == "random_walk":
        path[0] = target_a
        for index in range(1, count):
            ambient = path[index - 1] + rng.normal(0.0, 0.025, size=dim)
            path[index] = SimplexLandscape.project(ambient)
    elif scenario == "mixed":
        # A held-out composition: stationary, ramp, cycle, abrupt shift, and
        # stochastic drift.  The composition itself is never used in pretraining.
        cuts = np.linspace(0, count, 6, dtype=int)
        target_b = _dirichlet_target(rng, dim)
        target_c = _dirichlet_target(rng, dim)
        path[cuts[0] : cuts[1]] = target_a

        length = cuts[2] - cuts[1]
        weights = np.linspace(0.0, 1.0, length, endpoint=False)
        path[cuts[1] : cuts[2]] = (1.0 - weights[:, None]) * target_a + weights[
            :, None
        ] * target_b

        length = cuts[3] - cuts[2]
        phase = np.linspace(0.0, 2.0 * np.pi, length, endpoint=False)
        weights = 0.5 * (1.0 - np.cos(phase))
        path[cuts[2] : cuts[3]] = (1.0 - weights[:, None]) * target_b + weights[
            :, None
        ] * target_c

        jump_target = _dirichlet_target(rng, dim)
        path[cuts[3] : cuts[4]] = jump_target
        if cuts[4] < count:
            path[cuts[4]] = jump_target
        for index in range(cuts[4] + 1, count):
            ambient = path[index - 1] + rng.normal(0.0, 0.035, size=dim)
            path[index] = SimplexLandscape.project(ambient)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    # Convex interpolation preserves feasibility analytically.  Projection here
    # also removes negligible floating-point residue before Gym validates theta.
    return np.vstack([SimplexLandscape.project(point) for point in path])


def _value_noise(kind: str, seed: int):
    if kind == "none":
        return None
    if kind == "gaussian":
        return GaussianNoise(sigma=0.003, seed=seed)
    if kind == "correlated":
        return CorrelatedNoise(sigma=0.004, phi=0.85, dim=1, seed=seed)
    if kind == "heavy_tailed":
        return HeavyTailedNoise(alpha=2.2, scale=0.003, seed=seed)
    raise ValueError(f"unknown noise kind: {kind}")


def make_transfer_env(
    scenario: str,
    dim: int,
    horizon: int,
    seed: int,
    noise_kind: str,
    evaluation_session: Optional[GymEvaluationSession] = None,
) -> WindGymEnv:
    """Create the actual Gymnasium environment used by one episode."""

    path = build_simplex_path(scenario, dim, horizon, seed)
    environment = DynamicEnvironment(
        dim=dim,
        drift=SimplexPathDrift(path),
        landscape=SimplexLandscape(),
        initial_theta=path[0],
        bounds=None,
    )
    oracle = ZeroOrderOracle(
        environment,
        value_noise=_value_noise(noise_kind, seed + 100_003),
        seed=seed + 200_003,
    )
    return WindGymEnv(
        environment,
        oracle=oracle,
        T=horizon,
        action_mode="delta",
        max_step=0.30,
        reward="neg_value",
        x0=np.full(dim, 1.0 / dim),
        geometry="simplex",
        evaluation_session=evaluation_session,
    )


@dataclass
class ControllerMemory:
    """Policy-visible value/action history; it never stores privileged Gym info."""

    dimension: int
    horizon: int
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.exploration_pairs = np.empty((self.horizon, 2), dtype=int)
        for step in range(self.horizon):
            source = int(rng.integers(self.dimension))
            destination = int(rng.integers(self.dimension - 1))
            if destination >= source:
                destination += 1
            self.exploration_pairs[step] = (source, destination)
        self.last_direction = np.zeros(self.dimension, dtype=float)
        self.value_ema: Optional[float] = None
        self.relative_improvement = 0.0
        self.success_ema = 0.5
        self.previous_reallocation = 0.0
        self.stagnation = 0

    def features(self, value: float, step: int) -> np.ndarray:
        value = float(value)
        baseline = value if self.value_ema is None else self.value_ema
        surprise = (value - baseline) / (abs(value) + abs(baseline) + 1e-6)
        progress = 2.0 * step / max(1, self.horizon - 1) - 1.0
        return np.array(
            [
                1.0,
                np.tanh(8.0 * max(0.0, value)),
                np.tanh(4.0 * self.relative_improvement),
                np.tanh(3.0 * surprise),
                np.tanh(4.0 * self.previous_reallocation),
                2.0 * self.success_ema - 1.0,
                np.tanh(self.stagnation / 4.0),
                np.clip(progress, -1.0, 1.0),
            ],
            dtype=float,
        )

    def _exploration_direction(self, current_x: np.ndarray, step: int) -> np.ndarray:
        source_hint, destination = self.exploration_pairs[step]
        positive = np.flatnonzero(np.asarray(current_x) > 1e-8)
        if len(positive) == 0:  # Defensive: Gym's simplex projection prevents this.
            source = int(source_hint)
        else:
            source = int(positive[int(source_hint) % len(positive)])
        if destination == source:
            destination = (destination + 1) % self.dimension
        direction = np.zeros(self.dimension, dtype=float)
        direction[source] = -1.0
        direction[destination] = 1.0
        return direction

    def action_delta(
        self, action_index: int, current_x: np.ndarray, step: int
    ) -> np.ndarray:
        reallocation, maneuver = CONTROL_ACTIONS[int(action_index)]
        exploratory = self._exploration_direction(current_x, step)
        if maneuver == "explore" or not np.any(self.last_direction):
            direction = exploratory
        elif maneuver == "repeat":
            direction = self.last_direction
        else:
            direction = -self.last_direction
        return float(reallocation) * direction

    def update(
        self,
        current_value: float,
        next_value: float,
        realized_delta: np.ndarray,
    ) -> None:
        current_value = float(current_value)
        next_value = float(next_value)
        improvement = current_value - next_value
        denominator = abs(current_value) + abs(next_value) + 1e-6
        self.relative_improvement = improvement / denominator
        success = float(improvement > 0.0)
        self.success_ema = 0.85 * self.success_ema + 0.15 * success
        self.stagnation = 0 if self.relative_improvement > 0.02 else self.stagnation + 1
        self.value_ema = (
            next_value
            if self.value_ema is None
            else 0.9 * self.value_ema + 0.1 * next_value
        )
        realized_delta = np.asarray(realized_delta, dtype=float)
        self.previous_reallocation = float(np.sum(np.abs(realized_delta)))
        scale = float(np.max(np.abs(realized_delta)))
        if scale > 1e-12:
            self.last_direction = realized_delta / scale


class QLearning:
    """Watkins-Dayan online TD update; truncation bootstraps, termination does not."""

    bin_edges = (
        (0.04, 0.12, 0.30),
        (-0.30, 0.05, 0.40),
        (-0.30, 0.05, 0.40),
        (-0.40, 0.20, 0.70),
        (0.20, 0.70),
    )
    state_shape = (4, 4, 4, 4, 3)

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.learning_rate, self.discount, self.exploration = 0.1, 0.95, 0.2
        prior = np.array([-0.8 - 2 * amount for amount, _ in CONTROL_ACTIONS])
        self.q_values = np.tile(prior, (int(np.prod(self.state_shape)), 1))

    def state_index(self, features):
        bins = tuple(
            int(np.digitize(value, edges))
            for value, edges in zip(features, self.bin_edges)
        )
        return int(np.ravel_multi_index(bins, self.state_shape))

    def predict(self, observation, deterministic=True):
        state = self.state_index(observation)
        if not deterministic and self.rng.random() < self.exploration:
            return int(self.rng.integers(12)), None
        return int(np.argmax(self.q_values[state])), None

    def update(self, observation, action, reward, next_observation, terminated):
        state = self.state_index(observation)
        continuation = (
            0
            if terminated
            else float(self.q_values[self.state_index(next_observation)].max())
        )
        target = float(reward) + self.discount * continuation
        self.q_values[state, action] += self.learning_rate * (
            target - self.q_values[state, action]
        )

    def save(self, path):
        np.savez_compressed(
            path,
            q_values=self.q_values,
            bin_edges_json=json.dumps([list(b) for b in self.bin_edges]),
        )

    @classmethod
    def load(cls, path, seed=0):
        result = cls(seed)
        with np.load(path, allow_pickle=False) as data:
            result.q_values = data["q_values"].copy()
        return result


def _q_train(policy, env):
    while not env.exhausted:
        observation, _ = env.reset()
        while True:
            action, _ = policy.predict(observation, deterministic=False)
            following, reward, terminated, truncated, _ = env.step(action)
            policy.update(observation, action, reward, following, terminated)
            observation = following
            if terminated or truncated:
                break


def _evaluate_public(policy, env):
    while not env.exhausted:
        observation, _ = env.reset()
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break


def episode_schedule(profile, root_seed, replicate, phase, start, stop, condition):
    from wind_benchmark.integrations.rl_env import EpisodeSpec

    episodes = []
    for index in range(start, stop):
        workload = stream_seed(
            root_seed, "rl_transfer_v3", phase, replicate, "workload", index
        )
        pair_seed = stream_seed(
            root_seed, "rl_transfer_v3", phase, replicate, "service_pairs", index
        )
        if phase == "training":
            scenario = (
                TRAIN_SCENARIOS[index % 5]
                if condition == "curriculum"
                else "stationary"
            )
            dimension = profile.train_dimensions[index % len(profile.train_dimensions)]
            horizon, noise = (
                profile.train_horizon,
                ("none", "gaussian", "correlated")[index % 3],
            )
        else:
            scenario, dimension = "mixed", profile.transfer_dimension
            horizon, noise = profile.transfer_horizon, "heavy_tailed"
        episodes.append(
            EpisodeSpec(scenario, dimension, horizon, workload, pair_seed, noise)
        )
    return episodes


def _ppo(env, initialization_seed):
    import torch
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1000,
        batch_size=100,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": torch.nn.Tanh,
        },
        seed=initialization_seed,
        device="cpu",
        verbose=0,
    )


def _train(policy, algorithm, env, *, continuation=False):
    if algorithm == "QLearning":
        _user_call(_q_train, policy, env)
    else:
        from wind_benchmark.integrations.rl_env import bounded_vec_env_class

        if env.total_steps % 1000:
            raise ValueError(
                "PPO training checkpoints must contain complete 1000-step rollouts"
            )
        vec = bounded_vec_env_class()(env)
        _user_call(policy.set_env, vec, force_reset=True)
        _user_call(
            policy.learn,
            total_timesteps=env.total_steps,
            reset_num_timesteps=not continuation,
            progress_bar=False,
        )
    if (
        env.transitions != env.total_steps
        or env.queries != env.transitions + env.completed_episodes
    ):
        raise RuntimeError("RL interaction accounting failed")
    env.close()
    return {
        "transitions": env.transitions,
        "queries": env.queries,
        "episodes": env.completed_episodes,
    }


def crossed_bootstrap(array, repetitions=2000, seed=0):
    """Crossed resampling of independent training seeds and held-out workloads.

    Trailing condition/checkpoint axes share both resampling indices, preserving
    pairing. Workloads are NOT treated as independent policy training replicates.
    """
    values = np.asarray(array, dtype=float)
    if values.ndim < 2 or min(values.shape[:2]) < 2 or not np.isfinite(values).all():
        raise ValueError(
            "A complete crossed design with >=2 seeds and workloads is required"
        )
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        a = rng.integers(values.shape[0], size=values.shape[0])
        b = rng.integers(values.shape[1], size=values.shape[1])
        estimates.append(values[a[:, None], b[None, :]].mean(axis=(0, 1)))
    return values.mean(axis=(0, 1)), *np.quantile(estimates, [0.025, 0.975], axis=0)


def aggregate_rl(frame, profile, root_seed):
    conditions = ("curriculum", "stationary_only", "no_pretraining")
    metrics = (
        "mean_regret",
        "mean_tracking_error",
        "post_change_error",
        "dynamic_regret",
        "tail_regret",
        "max_constraint_violation",
    )
    rows, contrasts = [], []
    for algorithm in ("QLearning", "PPO"):
        subset = frame[frame.algorithm == algorithm]
        for metric in metrics:
            tensor = np.empty(
                (
                    profile.training_seeds,
                    profile.evaluation_seeds,
                    3,
                    len(profile.fine_tune_checkpoints),
                )
            )
            for condition_index, condition in enumerate(conditions):
                for checkpoint_index, checkpoint in enumerate(
                    profile.fine_tune_checkpoints
                ):
                    part = subset[
                        (subset.condition == condition)
                        & (subset.checkpoint == checkpoint)
                    ]
                    pivot = part.pivot(
                        index="training_seed", columns="workload", values=metric
                    )
                    tensor[:, :, condition_index, checkpoint_index] = pivot.reindex(
                        index=range(profile.training_seeds),
                        columns=range(profile.evaluation_seeds),
                    ).to_numpy()
            mean, low, high = crossed_bootstrap(
                tensor, 2000, stream_seed(root_seed, "rl", "statistics", metric)
            )
            for i, condition in enumerate(conditions):
                for j, checkpoint in enumerate(profile.fine_tune_checkpoints):
                    rows.append(
                        dict(
                            algorithm=algorithm,
                            condition=condition,
                            checkpoint=checkpoint,
                            metric=metric,
                            mean=mean[i, j],
                            ci_low=low[i, j],
                            ci_high=high[i, j],
                            n_training_seeds=profile.training_seeds,
                            n_workloads=profile.evaluation_seeds,
                        )
                    )
            for other in (1, 2):
                delta = tensor[:, :, 0, :] - tensor[:, :, other, :]
                dm, dl, dh = crossed_bootstrap(
                    delta, 2000, stream_seed(root_seed, "rl", "statistics", metric)
                )
                for j, checkpoint in enumerate(profile.fine_tune_checkpoints):
                    contrasts.append(
                        dict(
                            algorithm=algorithm,
                            contrast="curriculum-minus-" + conditions[other],
                            checkpoint=checkpoint,
                            metric=metric,
                            mean=dm[j],
                            ci_low=dl[j],
                            ci_high=dh[j],
                        )
                    )
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def _episode_scores(records, public, horizon):
    transitions = records[1:]
    if len(records) != horizon + 1 or len(public) != horizon:
        raise RuntimeError("Incomplete evaluation episode")
    regrets = np.array([r.regret for r in transitions])
    errors = np.array([r.error for r in transitions])
    window = max(2, horizon // 30)
    post = np.concatenate(
        [
            errors[int(f * horizon) : int(f * horizon) + window]
            for f in (0.2, 0.4, 0.6, 0.8)
        ]
    )
    scores = dict(
        dynamic_regret=float(regrets.sum()),
        mean_regret=float(regrets.mean()),
        tail_regret=float(regrets[int(0.8 * horizon) :].mean()),
        mean_tracking_error=float(errors.mean()),
        post_change_error=float(post.mean()),
        max_constraint_violation=max(p["constraint_violation"] for p in public),
        value_queries=horizon + 1,
    )
    return scores, regrets, errors


def _run_chain(
    profile, policy_dir, seed, algorithm, replicate, condition, trace_offset
):
    """Private independent chain; no checkpoint or episode is trained out of order."""
    _library_caller()
    import copy
    from wind_benchmark.integrations.rl_env import (
        ResourceAllocationEnv,
        bounded_vec_env_class,
    )

    policy_dir = Path(policy_dir)
    rows, costs, traces = [], [], {}
    session = GymEvaluationSession()
    episode_index = 0
    initialization_seed = stream_seed(
        seed,
        "rl_transfer_v3",
        "initialization",
        ALGORITHM_RNG_IDS[algorithm],
        replicate,
    )
    try:
        print(
            f"{algorithm} training_seed={replicate+1}/{profile.training_seeds} {condition}",
            flush=True,
        )
        # A policy constructor sees spaces only; no reset or oracle call.
        pre_specs = episode_schedule(
            profile,
            seed,
            replicate,
            "training",
            0,
            profile.pretrain_episodes,
            condition,
        )
        pre_env = ResourceAllocationEnv(pre_specs)
        if algorithm == "QLearning":
            policy = QLearning(initialization_seed)
        else:
            policy = _user_call(
                _ppo, bounded_vec_env_class()(pre_env), initialization_seed
            )
        training_seed = stream_seed(
            seed,
            "rl_transfer_v3",
            "training",
            ALGORITHM_RNG_IDS[algorithm],
            replicate,
            CONDITION_RNG_IDS[condition],
        )
        if algorithm == "QLearning":
            policy.rng = np.random.default_rng(training_seed)
        else:
            _user_call(policy.set_random_seed, training_seed)
        report_progress(
            phase="training",
            algorithm=algorithm,
            training_seed=replicate,
            condition=condition,
            episodes=0,
            transitions=0,
            queries=0,
        )
        if condition != "no_pretraining":
            cost = _train(policy, algorithm, pre_env)
        else:
            pre_env.close()
            cost = {"transitions": 0, "queries": 0, "episodes": 0}
        costs.append(
            dict(
                algorithm=algorithm,
                training_seed=replicate,
                condition=condition,
                phase="training",
                checkpoint=0,
                **cost,
            )
        )
        # Transfer only Q-table or actor/critic parameters. Optimizer,
        # buffers and episode state do not cross the phase boundary.
        fine_seed = stream_seed(
            seed,
            "rl_transfer_v3",
            "adaptation",
            ALGORITHM_RNG_IDS[algorithm],
            replicate,
            CONDITION_RNG_IDS[condition],
        )
        if algorithm == "QLearning":
            transferred = QLearning(fine_seed)
            transferred.q_values = policy.q_values.copy()
        else:
            placeholder = ResourceAllocationEnv(
                episode_schedule(
                    profile, seed, replicate, "adaptation", 0, 1, condition
                )
            )
            transferred = _user_call(
                _ppo,
                bounded_vec_env_class()(placeholder),
                initialization_seed,
            )
            _user_call(
                transferred.policy.load_state_dict,
                copy.deepcopy(policy.policy.state_dict()),
            )
            _user_call(transferred.set_random_seed, fine_seed)
            placeholder.close()
        policy = transferred
        completed = 0
        for checkpoint in profile.fine_tune_checkpoints:
            check_cancelled()
            report_progress(
                phase="adaptation",
                checkpoint=checkpoint,
                episodes=0,
                transitions=0,
                queries=0,
            )
            if checkpoint > completed:
                fine_env = ResourceAllocationEnv(
                    episode_schedule(
                        profile,
                        seed,
                        replicate,
                        "adaptation",
                        completed,
                        checkpoint,
                        condition,
                    )
                )
                cost = _train(policy, algorithm, fine_env, continuation=completed > 0)
                costs.append(
                    dict(
                        algorithm=algorithm,
                        training_seed=replicate,
                        condition=condition,
                        phase="adaptation",
                        checkpoint=checkpoint,
                        **cost,
                    )
                )
            completed = checkpoint
            stem = f"{algorithm}_seed{replicate}_{condition}_ep{checkpoint}"
            # Policies contain no evaluation records and may be saved now.
            _user_call(
                policy.save,
                policy_dir / (stem + (".npz" if algorithm == "QLearning" else ".zip")),
            )
            # Evaluation workloads and service pairs are common to ALL
            # training replicates, conditions, algorithms and checkpoints.
            specs = episode_schedule(
                profile,
                seed,
                0,
                "evaluation",
                0,
                profile.evaluation_seeds,
                condition,
            )
            report_progress(phase="evaluation", episodes=0, transitions=0, queries=0)
            eval_env = ResourceAllocationEnv(specs, session, record_public=True)
            _user_call(_evaluate_public, policy, eval_env)
            public_logs = eval_env._ResourceAllocationEnv__logs
            for workload, public in enumerate(public_logs):
                records = session._take_completed_records(episode_index)
                episode_index += 1
                scores, regrets, errors = _episode_scores(
                    records, public, profile.transfer_horizon
                )
                row = dict(
                    algorithm=algorithm,
                    training_seed=replicate,
                    condition=condition,
                    checkpoint=checkpoint,
                    workload=workload,
                    status="SUCCESS",
                    **scores,
                )
                rows.append(row)
                key = f"episode_{trace_offset + episode_index:05d}"
                traces[key + "_regret"] = regrets
                traces[key + "_error"] = errors
                traces[key + "_public"] = np.array(
                    [
                        [
                            p[k]
                            for k in (
                                "action",
                                "reward",
                                "observed_value",
                                "reallocation_l1",
                                "constraint_violation",
                            )
                        ]
                        for p in public
                    ]
                )
                row["trace_key"] = key
            costs.append(
                dict(
                    algorithm=algorithm,
                    training_seed=replicate,
                    condition=condition,
                    phase="evaluation",
                    checkpoint=checkpoint,
                    transitions=eval_env.transitions,
                    queries=eval_env.queries,
                    episodes=eval_env.completed_episodes,
                )
            )
            eval_env.close()
            print(
                f"  checkpoint={checkpoint}; evaluation episodes={len(rows)}",
                flush=True,
            )
        session.finish()
    except BaseException:
        session.abort()
        raise
    return rows, costs, traces


@_series
def _run_experiment(profile, output, seed, workers=1):
    jobs = []
    per_chain = len(profile.fine_tune_checkpoints) * profile.evaluation_seeds
    for algorithm in ("QLearning", "PPO"):
        for replicate in range(profile.training_seeds):
            for condition in ("curriculum", "stationary_only", "no_pretraining"):
                index = len(jobs)
                jobs.append(
                    (
                        index,
                        dict(
                            profile=asdict(profile),
                            policy_dir=str(Path(output).resolve() / "policies"),
                            seed=seed,
                            algorithm=algorithm,
                            replicate=replicate,
                            condition=condition,
                            trace_offset=index * per_chain,
                        ),
                    )
                )
    options = execution_options(workers, len(jobs))
    options.update(
        version="rl_process_v1",
        job_unit="algorithm_training_seed_condition",
        torch_threads=1,
    )
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    (output / "policies").mkdir()
    execution = Execution(output, options, len(jobs))
    started = time.perf_counter()
    owner = _current.get()
    owner._start()
    success = False
    try:
        completed = {}
        for index, payload in execution.rl_results(jobs):
            completed[index] = payload
            execution.collected()
            print(f"Completed RL chains: {len(completed)}/{len(jobs)}", flush=True)
        if set(completed) != set(range(len(jobs))):
            raise RuntimeError("Incomplete RL chain design")
        rows, costs, traces = [], [], {}
        for index in range(len(jobs)):
            chain_rows, chain_costs, chain_traces = completed.pop(index)
            rows.extend(chain_rows)
            costs.extend(chain_costs)
            traces.update(chain_traces)
        frame, cost_frame = pd.DataFrame(rows), pd.DataFrame(costs)
        expected = (
            2
            * 3
            * profile.training_seeds
            * profile.evaluation_seeds
            * len(profile.fine_tune_checkpoints)
        )
        if len(frame) != expected:
            raise RuntimeError("Incomplete RL crossed design")
        summary, contrasts = aggregate_rl(frame, profile, seed)
        manifest = {
            "package_version": __version__,
            "protocol": "rl_transfer_v3",
            "export_schema_version": 4,
            "metrics_protocol": "metrics_v2",
            "privacy_protocol": "local_series_v2",
            "profile": asdict(profile),
            "root_seed": seed,
            "evaluation_episodes": len(frame),
            "dependencies": {
                p: importlib.metadata.version(p)
                for p in ("stable-baselines3", "torch", "gymnasium", "numpy")
            },
            "source_sha256": source_hashes(),
            "statistics": "crossed training-seed/workload bootstrap, paired arms and checkpoints; 2000 resamples",
            "observation": "five continuous public history features; Q discretizes, PPO does not",
            "actions": list(CONTROL_ACTIONS),
            "algorithm_rng_ids": ALGORITHM_RNG_IDS,
            "condition_rng_ids": CONDITION_RNG_IDS,
            "q_learning": {
                "learning_rate": 0.1,
                "discount": 0.95,
                "exploration": 0.2,
                "td_clipping": False,
                "bins": [list(b) for b in QLearning.bin_edges],
                "initial_prior": "-.8 - 2*action_magnitude",
            },
            "ppo": {
                "library": "Stable-Baselines3",
                "version": "2.9.2",
                "device": "cpu",
                "n_envs": 1,
                "net_arch": [64, 64],
                "activation": "Tanh",
                "learning_rate": 0.0003,
                "gamma": 0.95,
                "gae_lambda": 0.95,
                "n_steps": 1000,
                "batch_size": 100,
                "n_epochs": 10,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "normalize_advantage": True,
                "reward_normalization": False,
                "value_clipping": False,
            },
            "interaction_accounting": "H transitions plus one reset value query per complete episode",
            "evaluation_policy": "deterministic predict; no learning or checkpoint selection",
            "training_scenarios": list(TRAIN_SCENARIOS),
            "training_noise_cycle": ["none", "gaussian", "correlated"],
            "transfer_noise": "heavy_tailed",
            "seed_derivation": "rl_transfer_v3 / phase / explicit algorithm and condition IDs / replicate",
            "reward": "10*(-next_noisy_value + 2*(current_noisy_value-next_noisy_value) - .02*L1_reallocation)",
            "transfer": "weights/Q-table only; reset optimizer at transfer; preserve optimizer across checkpoints",
            "execution": execution.options,
            "runtime_seconds": time.perf_counter() - started,
            "total_queries": int(cost_frame.queries.sum()),
            "no_article_figures": True,
        }
        for name, data in (
            ("evaluation_runs", frame),
            ("interaction_costs", cost_frame),
            ("summary", summary),
            ("paired_contrasts", contrasts),
        ):
            execution.defer(data.to_csv, output / (name + ".csv"), index=False)
        execution.defer(
            np.savez_compressed, output / "evaluation_histories.npz", **traces
        )
        execution.computed(manifest)
        _after_release(execution.export)
        success = True
    except BaseException as exc:
        execution.failed(isinstance(exc, KeyboardInterrupt))
        raise
    finally:
        owner._end(success, count=False)
    return output


def run_experiment(profile, output_root, seed=20261112, *, workers=1):
    """Restore NumPy, Python and Torch RNG states even after a failure."""
    import random
    import torch

    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state, threads = torch.get_rng_state(), torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            return _run_experiment(profile, output_root, seed, workers)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(threads)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replicated value-only Q-learning/PPO transfer, no article plots"
    )
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-plots", action="store_true", help="All v3 runs are numerical-only"
    )
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    if args.dry_run:
        options = execution_options(args.workers, 6 * profile.training_seeds)
        options.update(
            version="rl_process_v1",
            job_unit="algorithm_training_seed_condition",
            torch_threads=1,
        )
        print(
            json.dumps(
                dict(
                    profile=asdict(profile),
                    execution=options,
                    chains=6 * profile.training_seeds,
                    evaluation_episodes=6
                    * profile.training_seeds
                    * profile.evaluation_seeds
                    * len(profile.fine_tune_checkpoints),
                ),
                indent=2,
            )
        )
        return 0
    print(run_experiment(profile, args.output_dir, args.seed, workers=args.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
