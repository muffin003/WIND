"""Smoke tests for the optional Gymnasium adapter (skipped if gymnasium absent)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from wind_benchmark.core import (
    DynamicEnvironment,
    RandomWalkDrift,
    QuadraticLandscape,
    GaussianNoise,
)
from wind_benchmark.oracle import FirstOrderOracle, ZeroOrderOracle
from wind_benchmark.gym_env import WindGymEnv


def _zo_env(**kw):
    env = DynamicEnvironment(
        dim=3,
        drift=RandomWalkDrift(sigma=0.05, seed=1),
        landscape=QuadraticLandscape(dim=3, condition_number=3.0, seed=2),
    )
    oracle = ZeroOrderOracle(env, value_noise=GaussianNoise(sigma=0.01, seed=3), seed=4)
    return WindGymEnv(env, oracle=oracle, T=15, **kw)


def test_spaces_and_reset():
    env = _zo_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == env.observation_space.shape
    assert "theta" in info and "error" in info
    # ZO oracle exposes value (1) but not grad -> obs dim = d + 1
    assert env.observation_space.shape == (env.dim + 1,)


def test_episode_runs_and_truncates():
    env = _zo_env()
    env.reset(seed=0)
    truncated = False
    steps = 0
    while not truncated:
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        steps += 1
        assert steps <= env.T
    assert steps == env.T


def test_reward_non_positive():
    env = _zo_env(reward="neg_regret")
    env.reset(seed=0)
    _, r, _, _, _ = env.step(env.action_space.sample())
    assert r <= 1e-9  # -regret, regret >= 0


def test_delta_action_mode():
    env = _zo_env(action_mode="delta", max_step=0.5)
    assert np.allclose(env.action_space.high, 0.5)
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(np.full(env.dim, 0.5))
    assert env.observation_space.contains(obs)


def test_first_order_obs_includes_grad():
    env = DynamicEnvironment(
        dim=2,
        drift=RandomWalkDrift(sigma=0.05, seed=1),
        landscape=QuadraticLandscape(dim=2, condition_number=2.0, seed=2),
    )
    oracle = FirstOrderOracle(env, grad_noise=GaussianNoise(sigma=0.01, seed=5))
    wenv = WindGymEnv(env, oracle=oracle, T=10)
    # FO (blind by default): value=None, grad present -> obs dim = d + d
    assert wenv.observation_space.shape == (2 * env.dim,)
    obs, _ = wenv.reset(seed=0)
    assert wenv.observation_space.contains(obs)


def test_episodes_reproducible_after_reset():
    env = _zo_env()

    def rollout():
        env.reset(seed=0)
        thetas = []
        for _ in range(env.T):
            _, _, _, _, info = env.step(np.zeros(env.dim))  # fixed action
            thetas.append(info["theta"].copy())
        return np.array(thetas)

    assert np.allclose(rollout(), rollout())


def test_passes_gymnasium_env_checker():
    from gymnasium.utils.env_checker import check_env

    env = _zo_env()
    # Raises if the environment violates the Gymnasium API contract.
    check_env(env, skip_render_check=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
