"""Unchanged workload and public-memory contracts for RL v3."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from wind_benchmark.experiments.rl import (
    CONTROL_ACTIONS,
    PROFILES,
    ControllerMemory,
    build_simplex_path,
)


@pytest.mark.parametrize(
    "scenario",
    ["stationary", "linear", "cyclic", "jump", "random_walk", "mixed"],
)
def test_curriculum_paths_remain_on_simplex(scenario):
    path = build_simplex_path(scenario, dim=7, horizon=24, seed=123)

    assert path.shape == (25, 7)
    assert np.min(path) >= -1e-12
    assert np.allclose(path.sum(axis=1), 1.0, atol=1e-12)
    if scenario != "stationary":
        assert np.max(np.linalg.norm(path - path[0], axis=1)) > 1e-4


def test_value_only_memory_generates_dimension_invariant_reallocation_actions():
    memory = ControllerMemory(dimension=7, horizon=5, seed=123)
    current_x = np.full(7, 1.0 / 7.0)

    for action_index, (magnitude, _maneuver) in enumerate(CONTROL_ACTIONS):
        delta = memory.action_delta(action_index, current_x, step=0)
        assert delta.shape == (7,)
        assert np.isclose(np.sum(delta), 0.0)
        assert np.max(np.abs(delta)) <= magnitude + 1e-12


def test_profiles_end_at_declared_fine_tuning_budget():
    for profile in PROFILES.values():
        assert profile.fine_tune_checkpoints[0] == 0
        assert profile.fine_tune_checkpoints[-1] == profile.fine_tune_episodes
        assert tuple(sorted(profile.fine_tune_checkpoints)) == (
            profile.fine_tune_checkpoints
        )


def test_ppo_checkpoints_are_whole_rollouts():
    for profile in PROFILES.values():
        assert profile.pretrain_episodes * profile.train_horizon % 1000 == 0
        checkpoints = profile.fine_tune_checkpoints
        for a, b in zip(checkpoints, checkpoints[1:]):
            assert (b - a) * profile.transfer_horizon % 1000 == 0
