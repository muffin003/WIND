"""Protocol tests for the Gymnasium curriculum-transfer experiment."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from wind_benchmark.expRL import (  # noqa: E402
    CONTROL_ACTIONS,
    FEATURE_NAMES,
    PROFILES,
    ControllerMemory,
    ValueOnlyQController,
    build_simplex_path,
    make_transfer_env,
    run_episode,
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


def test_rl_rollout_is_value_only_and_uses_feasible_gym_actions():
    horizon = 20
    environment = make_transfer_env(
        "mixed", dim=9, horizon=horizon, seed=456, noise_kind="heavy_tailed"
    )
    policy = ValueOnlyQController(seed=789)

    result = run_episode(environment, policy, deterministic=True, action_seed=2468)

    assert len(result.rewards) == horizon
    assert result.features.shape == (horizon, len(FEATURE_NAMES))
    assert result.query_count == horizon + 1  # reset observation plus one per action
    assert environment.oracle.n_value_queries == horizon + 1
    assert environment.oracle.n_grad_queries == 0
    assert np.max(result.constraint_violations) < 1e-7
    assert np.all(np.isfinite(result.rewards))
    assert np.all(np.isfinite(result.observed_values))
    assert np.all(result.regrets >= -1e-12)


def test_value_only_memory_generates_dimension_invariant_reallocation_actions():
    memory = ControllerMemory(dimension=7, horizon=5, seed=123)
    current_x = np.full(7, 1.0 / 7.0)

    for action_index, (magnitude, _maneuver) in enumerate(CONTROL_ACTIONS):
        delta = memory.action_delta(action_index, current_x, step=0)
        assert delta.shape == (7,)
        assert np.isclose(np.sum(delta), 0.0)
        assert np.max(np.abs(delta)) <= magnitude + 1e-12


def test_q_learning_update_is_finite_and_changes_values():
    policy = ValueOnlyQController(seed=321)
    features = np.tile(np.linspace(-0.5, 0.5, len(FEATURE_NAMES)), (12, 1))
    actions = np.arange(12, dtype=int)
    rewards = -np.linspace(0.1, 1.2, 12)
    before = policy.q_values.copy()

    diagnostics = policy.update(features, actions, rewards)
    probabilities = policy.probabilities(features[0])

    assert not np.array_equal(policy.q_values, before)
    assert np.all(np.isfinite(policy.q_values))
    assert np.isclose(np.sum(probabilities), 1.0)
    assert np.isfinite(diagnostics["return"])
    assert np.isfinite(diagnostics["entropy"])


def test_profiles_end_at_declared_fine_tuning_budget():
    for profile in PROFILES.values():
        assert profile.fine_tune_checkpoints[0] == 0
        assert profile.fine_tune_checkpoints[-1] == profile.fine_tune_episodes
        assert tuple(sorted(profile.fine_tune_checkpoints)) == (
            profile.fine_tune_checkpoints
        )
