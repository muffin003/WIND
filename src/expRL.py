"""Gymnasium curriculum and transfer experiment for WIND.

The experiment trains a lightweight tabular Q-controller from value-only
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
depends only on the benchmark's existing NumPy, pandas, matplotlib, and Gymnasium
dependencies.

Run a quick verification with::

    python -m wind_benchmark.expRL --profile smoke

Use ``--profile paper`` for the longer publication-oriented schedule.  Results are
written below ``results/rl_transfer_experiment`` by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    CorrelatedNoise,
    Drift,
    DynamicEnvironment,
    GaussianNoise,
    HeavyTailedNoise,
    SimplexLandscape,
)
from .gym_env import WindGymEnv
from .oracle import ZeroOrderOracle

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


PROFILES: Dict[str, RLProfile] = {
    "smoke": RLProfile(
        name="smoke",
        pretrain_episodes=300,
        fine_tune_episodes=30,
        train_horizon=64,
        transfer_horizon=120,
        evaluation_seeds=8,
        fine_tune_checkpoints=(0, 5, 15, 30),
        train_dimensions=(4, 6, 8),
        transfer_dimension=12,
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
        reward="neg_regret",
        x0=np.full(dim, 1.0 / dim),
        geometry="simplex",
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


class ValueOnlyQController:
    """Dimension-invariant Q-learning controller over reallocation maneuvers."""

    _bin_edges: Tuple[Tuple[float, ...], ...] = (
        (0.04, 0.12, 0.30),  # observed loss
        (-0.30, 0.05, 0.40),  # relative improvement
        (-0.30, 0.05, 0.40),  # surprise relative to value EMA
        (-0.40, 0.20, 0.70),  # recent success balance
        (0.20, 0.70),  # stagnation
    )
    _state_shape: Tuple[int, ...] = (4, 4, 4, 4, 3)

    def __init__(
        self,
        seed: int,
        learning_rate: float = 0.10,
        discount: float = 0.0,
        exploration: float = 0.20,
        reward_scale: float = 10.0,
        diagnostic_temperature: float = 0.10,
    ):
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.exploration = float(exploration)
        self.reward_scale = float(reward_scale)
        self.diagnostic_temperature = float(diagnostic_temperature)
        state_count = int(np.prod(self._state_shape))
        # Costs, and hence rewards, are non-positive.  A zero initialization would
        # make every untried action look better than all sampled actions and is
        # especially unsafe for large reallocations.  The conservative prior keeps
        # unseen actions ordered by their switching risk while epsilon exploration
        # still gives every maneuver a chance to be evaluated during training.
        action_prior = np.array(
            [-0.8 - 2.0 * float(action[0]) for action in CONTROL_ACTIONS],
            dtype=float,
        )
        self.q_values = np.tile(action_prior, (state_count, 1))

    def copy(self, seed: int) -> "ValueOnlyQController":
        clone = ValueOnlyQController(
            seed=seed,
            learning_rate=self.learning_rate,
            discount=self.discount,
            exploration=self.exploration,
            reward_scale=self.reward_scale,
            diagnostic_temperature=self.diagnostic_temperature,
        )
        clone.q_values = self.q_values.copy()
        return clone

    def state_index(self, features: np.ndarray) -> int:
        features = np.asarray(features, dtype=float)
        values = (features[1], features[2], features[3], features[5], features[6])
        bins = tuple(
            int(np.digitize(value, edges))
            for value, edges in zip(values, self._bin_edges)
        )
        return int(np.ravel_multi_index(bins, self._state_shape))

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        q_values = self.q_values[self.state_index(features)]
        logits = (q_values - np.max(q_values)) / self.diagnostic_temperature
        weights = np.exp(np.clip(logits, -30.0, 30.0))
        return weights / np.sum(weights)

    def choose(
        self, features: np.ndarray, deterministic: bool
    ) -> Tuple[int, np.ndarray]:
        state = self.state_index(features)
        if deterministic or self.rng.random() > self.exploration:
            action = int(np.argmax(self.q_values[state]))
        else:
            action = int(self.rng.integers(len(CONTROL_ACTIONS)))
        return action, self.probabilities(features)

    def update(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> Dict[str, float]:
        if len(rewards) == 0:
            return {"return": 0.0, "entropy": 0.0}

        entropies = []
        for index, (vector, action, reward) in enumerate(
            zip(features, actions, rewards)
        ):
            state = self.state_index(vector)
            if index + 1 == len(rewards):
                continuation = 0.0
            else:
                next_state = self.state_index(features[index + 1])
                continuation = float(np.max(self.q_values[next_state]))
            target = self.reward_scale * float(reward) + self.discount * continuation
            difference = target - self.q_values[state, int(action)]
            self.q_values[state, int(action)] += self.learning_rate * np.clip(
                difference, -2.0, 2.0
            )

            probabilities = self.probabilities(vector)
            entropies.append(
                -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
            )

        return {
            "return": float(np.sum(rewards)),
            "entropy": float(np.mean(entropies)),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "exploration": self.exploration,
            "reward_scale": self.reward_scale,
            "diagnostic_temperature": self.diagnostic_temperature,
            "feature_names": list(FEATURE_NAMES),
            "bin_edges": [list(edges) for edges in self._bin_edges],
            "state_shape": list(self._state_shape),
            "control_actions": [list(action) for action in CONTROL_ACTIONS],
            "q_values": self.q_values.tolist(),
        }


@dataclass
class EpisodeResult:
    features: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    regrets: np.ndarray
    observed_values: np.ndarray
    reallocations: np.ndarray
    errors: np.ndarray
    constraint_violations: np.ndarray
    query_count: int

    def summary(self) -> Dict[str, float]:
        tail_start = max(0, int(0.8 * len(self.rewards)))
        horizon = len(self.rewards)
        boundaries = [int(fraction * horizon) for fraction in (0.2, 0.4, 0.6, 0.8)]
        post_change = []
        window = max(2, horizon // 30)
        for boundary in boundaries:
            post_change.extend(self.errors[boundary : min(horizon, boundary + window)])
        selected = [CONTROL_ACTIONS[int(action)] for action in self.actions]
        maneuvers = [item[1] for item in selected]
        return {
            "dynamic_regret": float(np.sum(self.regrets)),
            "mean_regret": float(np.mean(self.regrets)),
            "tail_regret": float(np.mean(self.regrets[tail_start:])),
            "mean_tracking_error": float(np.mean(self.errors)),
            "post_change_error": float(np.mean(post_change)),
            "max_constraint_violation": float(np.max(self.constraint_violations)),
            "mean_step_size": float(np.mean([item[0] for item in selected])),
            "mean_reallocation_l1": float(np.mean(self.reallocations)),
            "exploration_fraction": float(np.mean(np.asarray(maneuvers) == "explore")),
            "reversal_fraction": float(np.mean(np.asarray(maneuvers) == "reverse")),
            "value_queries": float(self.query_count),
        }


def run_episode(
    environment: WindGymEnv,
    policy: ValueOnlyQController,
    *,
    deterministic: bool,
    action_seed: Optional[int] = None,
    switching_cost: float = SWITCHING_COST,
) -> EpisodeResult:
    """Run an episode whose policy receives only value and action history."""

    observation, _evaluation_info = environment.reset()
    dim = environment.dim
    if environment.oracle.n_grad_queries != 0 or observation.shape != (dim + 1,):
        raise RuntimeError("value-only episodes require x plus one scalar observation")
    if action_seed is None:
        action_seed = int(policy.rng.integers(1, 2**31 - 1))
    memory = ControllerMemory(dim, environment.T, action_seed)
    feature_rows: List[np.ndarray] = []
    action_rows: List[int] = []
    learning_rewards: List[float] = []
    regrets: List[float] = []
    observed_values: List[float] = []
    reallocations: List[float] = []
    errors: List[float] = []
    violations: List[float] = []

    for step in range(environment.T):
        current_x = np.asarray(observation[:dim], dtype=float)
        current_value = float(observation[dim])
        features = memory.features(current_value, step)
        action_index, _probabilities = policy.choose(features, deterministic)
        delta = memory.action_delta(action_index, current_x, step)
        next_observation, evaluation_reward, terminated, truncated, evaluation_info = (
            environment.step(delta.astype(np.float32))
        )
        next_x = np.asarray(next_observation[:dim], dtype=float)
        next_value = float(next_observation[dim])
        realized_delta = next_x - current_x
        reallocation = float(np.sum(np.abs(realized_delta)))
        # Value-difference shaping makes the causal effect of a maneuver visible
        # from two consecutive measurements.  It uses no clean objective: both
        # terms come directly from the zero-order observation stream.
        learning_reward = (
            -next_value
            + 2.0 * (current_value - next_value)
            - float(switching_cost) * reallocation
        )
        memory.update(current_value, next_value, realized_delta)

        feature_rows.append(features)
        action_rows.append(action_index)
        learning_rewards.append(learning_reward)
        regrets.append(-float(evaluation_reward))
        observed_values.append(next_value)
        reallocations.append(reallocation)
        # Privileged fields are consumed only by the evaluator after the policy
        # has selected its action; they are never part of policy features.
        errors.append(float(evaluation_info["error"]))
        violations.append(float(evaluation_info["constraint_violation"]))
        observation = next_observation
        if terminated or truncated:
            break

    return EpisodeResult(
        features=np.asarray(feature_rows, dtype=float),
        actions=np.asarray(action_rows, dtype=int),
        rewards=np.asarray(learning_rewards, dtype=float),
        regrets=np.asarray(regrets, dtype=float),
        observed_values=np.asarray(observed_values, dtype=float),
        reallocations=np.asarray(reallocations, dtype=float),
        errors=np.asarray(errors, dtype=float),
        constraint_violations=np.asarray(violations, dtype=float),
        query_count=int(environment.oracle.n_value_queries),
    )


def _pretrain_policy(
    profile: RLProfile,
    *,
    seed: int,
    curriculum: bool,
) -> Tuple[ValueOnlyQController, pd.DataFrame]:
    policy = ValueOnlyQController(seed)
    rng = np.random.default_rng(seed + 17)
    rows: List[Dict[str, object]] = []

    for episode in range(profile.pretrain_episodes):
        if curriculum:
            scenario = TRAIN_SCENARIOS[episode % len(TRAIN_SCENARIOS)]
            noise_kind = ("none", "gaussian", "correlated")[episode % 3]
            dim = profile.train_dimensions[episode % len(profile.train_dimensions)]
        else:
            scenario = "stationary"
            noise_kind = ("none", "gaussian", "correlated")[episode % 3]
            dim = profile.train_dimensions[episode % len(profile.train_dimensions)]

        episode_seed = int(rng.integers(1, 2**31 - 1))
        environment = make_transfer_env(
            scenario,
            dim,
            profile.train_horizon,
            episode_seed,
            noise_kind,
        )
        result = run_episode(
            environment, policy, deterministic=False, action_seed=episode_seed + 7
        )
        update = policy.update(result.features, result.actions, result.rewards)
        rows.append(
            {
                "pretraining": "curriculum" if curriculum else "stationary_only",
                "episode": episode + 1,
                "scenario": scenario,
                "dimension": dim,
                "noise": noise_kind,
                "return": update["return"],
                "q_softmax_entropy": update["entropy"],
            }
        )
    return policy, pd.DataFrame(rows)


def _evaluate_policy(
    profile: RLProfile,
    policy: ValueOnlyQController,
    method: str,
    checkpoint: int,
    seed_base: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for evaluation_seed in range(profile.evaluation_seeds):
        seed = seed_base + evaluation_seed
        environment = make_transfer_env(
            "mixed",
            profile.transfer_dimension,
            profile.transfer_horizon,
            seed,
            "heavy_tailed",
        )
        result = run_episode(
            environment, policy, deterministic=True, action_seed=seed + 7
        )
        row: Dict[str, object] = {
            "method": method,
            "fine_tune_episodes": checkpoint,
            "seed": seed,
            "scenario": "mixed",
            "dimension": profile.transfer_dimension,
            "noise": "heavy_tailed",
        }
        row.update(result.summary())
        rows.append(row)
    return rows


def _fine_tune_curve(
    profile: RLProfile,
    initial_policy: ValueOnlyQController,
    method: str,
    seed: int,
    evaluation_seed_base: int,
) -> Tuple[ValueOnlyQController, List[Dict[str, object]], List[Dict[str, object]]]:
    policy = initial_policy.copy(seed)
    rng = np.random.default_rng(seed + 31)
    evaluation_rows: List[Dict[str, object]] = []
    training_rows: List[Dict[str, object]] = []
    completed = 0

    for checkpoint in profile.fine_tune_checkpoints:
        for episode in range(completed, checkpoint):
            episode_seed = int(rng.integers(1, 2**31 - 1))
            environment = make_transfer_env(
                "mixed",
                profile.transfer_dimension,
                profile.transfer_horizon,
                episode_seed,
                "heavy_tailed",
            )
            result = run_episode(
                environment,
                policy,
                deterministic=False,
                action_seed=episode_seed + 7,
            )
            update = policy.update(result.features, result.actions, result.rewards)
            training_rows.append(
                {
                    "method": method,
                    "episode": episode + 1,
                    "return": update["return"],
                    "q_softmax_entropy": update["entropy"],
                }
            )
        completed = checkpoint
        evaluation_rows.extend(
            _evaluate_policy(
                profile,
                policy,
                method,
                checkpoint,
                evaluation_seed_base,
            )
        )

    return policy, evaluation_rows, training_rows


def _aggregate(evaluation: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "dynamic_regret",
        "mean_regret",
        "tail_regret",
        "mean_tracking_error",
        "post_change_error",
        "max_constraint_violation",
        "mean_step_size",
        "mean_reallocation_l1",
        "exploration_fraction",
        "reversal_fraction",
        "value_queries",
    )
    rows: List[Dict[str, object]] = []
    for (method, checkpoint), group in evaluation.groupby(
        ["method", "fine_tune_episodes"], sort=False
    ):
        row: Dict[str, object] = {
            "method": method,
            "fine_tune_episodes": int(checkpoint),
            "runs": len(group),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            half_width = (
                1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = half_width
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_results(summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    learning_methods = (
        "Curriculum pretrained",
        "Stationary pretrained",
        "No pretraining",
    )
    colors = {
        "Curriculum pretrained": "#1f77b4",
        "Stationary pretrained": "#ff7f0e",
        "No pretraining": "#2ca02c",
    }

    for method in learning_methods:
        group = summary[summary["method"] == method].sort_values("fine_tune_episodes")
        axes[0].errorbar(
            group["fine_tune_episodes"],
            group["mean_regret_mean"],
            yerr=group["mean_regret_ci95"],
            marker="o",
            capsize=3,
            label=method,
            color=colors[method],
        )
    axes[0].set_xlabel("Fine-tuning episodes on held-out task")
    axes[0].set_ylabel("Mean instantaneous regret")
    axes[0].set_title("Transfer to an unseen workload")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    zero_shot = summary[summary["fine_tune_episodes"] == 0].copy()
    order = list(learning_methods)
    zero_shot["order"] = zero_shot["method"].map(
        {name: i for i, name in enumerate(order)}
    )
    zero_shot = zero_shot.sort_values("order")
    axes[1].bar(
        np.arange(len(zero_shot)),
        zero_shot["post_change_error_mean"],
        yerr=zero_shot["post_change_error_ci95"],
        capsize=3,
        color=[colors[name] for name in zero_shot["method"]],
    )
    axes[1].set_xticks(np.arange(len(zero_shot)))
    axes[1].set_xticklabels(zero_shot["method"], rotation=25, ha="right")
    axes[1].set_ylabel("Post-change tracking error")
    axes[1].set_title("Zero-shot recovery after workload changes")
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("Gymnasium curriculum for adaptive resource allocation")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(profile: RLProfile, output_root: Path, seed: int) -> Path:
    """Train, transfer, evaluate, and persist one complete RL experiment."""

    started = time.perf_counter()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = output_root / f"{profile.name}_{timestamp}"
    run_directory.mkdir(parents=True, exist_ok=False)

    curriculum, curriculum_training = _pretrain_policy(
        profile, seed=seed + 101, curriculum=True
    )
    stationary, stationary_training = _pretrain_policy(
        profile, seed=seed + 202, curriculum=False
    )
    no_pretraining = ValueOnlyQController(seed + 303)

    policies = {
        "Curriculum pretrained": curriculum,
        "Stationary pretrained": stationary,
        "No pretraining": no_pretraining,
    }
    evaluation_rows: List[Dict[str, object]] = []
    fine_tune_rows: List[Dict[str, object]] = []
    final_policies: Dict[str, ValueOnlyQController] = {}
    evaluation_seed_base = seed + 900_000

    for index, (method, policy) in enumerate(policies.items()):
        final_policy, method_evaluation, method_training = _fine_tune_curve(
            profile,
            policy,
            method,
            seed + 10_000 + index * 1_000,
            evaluation_seed_base,
        )
        final_policies[method] = final_policy
        evaluation_rows.extend(method_evaluation)
        fine_tune_rows.extend(method_training)

    pretraining = pd.concat(
        [curriculum_training, stationary_training], ignore_index=True
    )
    fine_tuning = pd.DataFrame(fine_tune_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    summary = _aggregate(evaluation)

    pretraining.to_csv(run_directory / "pretraining.csv", index=False)
    fine_tuning.to_csv(run_directory / "fine_tuning.csv", index=False)
    evaluation.to_csv(run_directory / "evaluation_runs.csv", index=False)
    summary.to_csv(run_directory / "summary.csv", index=False)
    _plot_results(summary, run_directory / "rl_transfer.png")

    _write_json(
        run_directory / "policies.json",
        {
            "pretrained": {name: policy.to_dict() for name, policy in policies.items()},
            "fine_tuned": {
                name: policy.to_dict() for name, policy in final_policies.items()
            },
        },
    )
    elapsed = time.perf_counter() - started
    manifest = {
        "experiment": "gymnasium_curriculum_transfer",
        "application_proxy": "adaptive compute-resource allocation",
        "resource_interpretation": "simplex coordinates are service budget shares",
        "profile": asdict(profile),
        "seed": seed,
        "elapsed_seconds": elapsed,
        "training_scenarios": list(TRAIN_SCENARIOS),
        "transfer_scenario": "mixed",
        "transfer_noise": "heavy_tailed",
        "oracle": "zero-order-value-only",
        "controller": "dimension-invariant value-only tabular Q-learning",
        "learning_reward": (
            "-noisy_value_t+1 + 2 * (noisy_value_t - noisy_value_t+1) "
            "- switching_cost * reallocation_l1"
        ),
        "switching_cost": SWITCHING_COST,
        "action_mode": "delta",
        "geometry": "simplex",
        "policy_access": {
            "observation": True,
            "reward_history": True,
            "noisy_value": True,
            "gradient": False,
            "gym_info": False,
            "latent_theta": False,
        },
        "control_actions": [list(action) for action in CONTROL_ACTIONS],
        "files": {
            "pretraining": "pretraining.csv",
            "fine_tuning": "fine_tuning.csv",
            "evaluation": "evaluation_runs.csv",
            "summary": "summary.csv",
            "policies": "policies.json",
            "figure": "rl_transfer.png",
        },
    }
    _write_json(run_directory / "manifest.json", manifest)
    return run_directory


def _print_summary(run_directory: Path) -> None:
    summary = pd.read_csv(run_directory / "summary.csv")
    columns = [
        "method",
        "fine_tune_episodes",
        "mean_regret_mean",
        "post_change_error_mean",
        "max_constraint_violation_mean",
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nResults: {run_directory}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rl_transfer_experiment"),
    )
    arguments = parser.parse_args(argv)
    run_directory = run_experiment(
        PROFILES[arguments.profile], arguments.output, arguments.seed
    )
    _print_summary(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
