"""Smoke tests for the optional Gymnasium adapter (skipped if gymnasium absent)."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from wind_benchmark.core import (
    DynamicEnvironment,
    GrassmannLandscape,
    StationaryDrift,
    RandomWalkDrift,
    QuadraticLandscape,
    SimplexLandscape,
    StiefelDrift,
    StiefelLandscape,
    GaussianNoise,
    make_environment,
)
from wind_benchmark.oracle import FirstOrderOracle, ZeroOrderOracle
from wind_benchmark.gym_env import WindGymEnv
from wind_benchmark.manifold import principal_angle_distance


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


def test_euclidean_geometry_auto_preserves_clipping_behavior():
    env = _zo_env(action_mode="absolute", x_bounds=(-0.25, 0.25))
    assert env.geometry == "euclidean"
    obs, _ = env.reset(seed=0)
    obs, _, _, _, info = env.step(np.full(env.dim, 5.0))
    assert np.allclose(obs[: env.dim], 0.25)
    assert info["constraint_violation"] == 0.0


@pytest.mark.parametrize("action_mode", ["absolute", "delta"])
def test_simplex_actions_remain_feasible(action_mode):
    dim = 4
    theta0 = np.array([0.1, 0.2, 0.3, 0.4])
    env = DynamicEnvironment(
        dim=dim,
        drift=StationaryDrift(),
        landscape=SimplexLandscape(),
        initial_theta=theta0,
    )
    wrapper = WindGymEnv(
        env,
        oracle=ZeroOrderOracle(env),
        T=5,
        action_mode=action_mode,
        x0=np.array([-2.0, 0.0, 1.0, 4.0]),
        reward="neg_error",
    )

    obs, info = wrapper.reset(seed=0)
    assert wrapper.geometry == "simplex"
    assert np.all(obs[:dim] >= 0.0)
    assert np.isclose(np.sum(obs[:dim]), 1.0)
    assert info["constraint_violation"] < 1e-7

    for _ in range(3):
        raw_action = np.array([-3.0, 0.5, 2.0, 7.0])
        obs, reward, _, _, info = wrapper.step(raw_action)
        x = obs[:dim]
        assert np.all(x >= -1e-7)
        assert np.isclose(np.sum(x), 1.0, atol=1e-7)
        assert np.isclose(reward, -np.linalg.norm(x - info["theta"]), atol=1e-6)
        assert info["constraint_violation"] < 1e-7


def _stiefel_env(action_mode="absolute", reward="neg_regret"):
    d, r = 4, 2
    dim = d * r
    theta0 = StiefelLandscape.random_point(d, r, seed=20)
    x0 = StiefelLandscape.random_point(d, r, seed=21)
    env = DynamicEnvironment(
        dim=dim,
        drift=StiefelDrift(d=d, r=r, sigma=0.02, seed=22),
        landscape=StiefelLandscape(d=d, r=r),
        initial_theta=theta0,
        bounds=None,
    )
    wrapper = WindGymEnv(
        env,
        oracle=FirstOrderOracle(env),
        T=8,
        action_mode=action_mode,
        reward=reward,
        x0=x0,
        max_step=0.2,
    )
    return wrapper, d, r


@pytest.mark.parametrize("action_mode", ["absolute", "delta"])
def test_stiefel_actions_remain_orthonormal(action_mode):
    wrapper, d, r = _stiefel_env(action_mode=action_mode)
    obs, info = wrapper.reset(seed=0)
    assert wrapper.geometry == "stiefel"
    assert info["constraint_violation"] < 1e-7

    rng = np.random.default_rng(30)
    for _ in range(5):
        obs, reward, _, _, info = wrapper.step(rng.normal(size=d * r))
        X = obs[: d * r].reshape(d, r)
        assert np.allclose(X.T @ X, np.eye(r), atol=1e-7)
        assert reward <= 1e-9
        assert info["constraint_violation"] < 1e-7


def test_stiefel_negative_error_uses_frame_frobenius_distance():
    wrapper, d, r = _stiefel_env(action_mode="absolute", reward="neg_error")
    wrapper.reset(seed=0)
    raw_action = np.arange(1, d * r + 1, dtype=float)
    obs, reward, _, _, info = wrapper.step(raw_action)
    X = obs[: d * r].reshape(d, r)
    Theta = info["theta"].reshape(d, r)
    assert np.isclose(reward, -np.linalg.norm(X - Theta, ord="fro"), atol=1e-6)


def _grassmann_env(action_mode="absolute", reward="neg_error"):
    d, r = 4, 2
    dim = d * r
    theta0 = GrassmannLandscape.random_point(d, r, seed=50)
    x0 = GrassmannLandscape.random_point(d, r, seed=51)
    env = DynamicEnvironment(
        dim=dim,
        drift=StiefelDrift(d=d, r=r, sigma=0.02, seed=52),
        landscape=GrassmannLandscape(d=d, r=r),
        initial_theta=theta0,
        bounds=None,
    )
    wrapper = WindGymEnv(
        env,
        oracle=FirstOrderOracle(env),
        T=8,
        action_mode=action_mode,
        reward=reward,
        x0=x0,
        max_step=0.2,
    )
    return wrapper, d, r


@pytest.mark.parametrize("action_mode", ["absolute", "delta"])
def test_grassmann_actions_remain_orthonormal_and_use_subspace_error(action_mode):
    wrapper, d, r = _grassmann_env(action_mode=action_mode)
    obs, info = wrapper.reset(seed=0)
    assert wrapper.geometry == "grassmann"
    assert info["constraint_violation"] < 1e-7

    raw_action = np.arange(1, d * r + 1, dtype=float)
    obs, reward, _, _, info = wrapper.step(raw_action)
    X = obs[: d * r].reshape(d, r)
    Theta = info["theta"].reshape(d, r)
    assert np.allclose(X.T @ X, np.eye(r), atol=1e-7)
    assert np.isclose(reward, -principal_angle_distance(X, Theta), atol=1e-6)
    assert info["constraint_violation"] < 1e-7


def test_grassmann_reward_does_not_penalize_an_equivalent_basis():
    wrapper, d, r = _grassmann_env(action_mode="absolute", reward="neg_error")
    _, info = wrapper.reset(seed=0)
    Theta = info["theta"].reshape(d, r)
    Q = np.array([[0.0, -1.0], [1.0, 0.0]])
    _, reward, _, _, _ = wrapper.step((Theta @ Q).reshape(-1))
    assert abs(reward) < 1e-7


def test_constrained_factory_initializes_feasible_latent_states():
    simplex = make_environment(
        {
            "dim": 3,
            "drift": {"type": "stationary"},
            "landscape": {"type": "simplex"},
        },
        seed=5,
    )
    simplex_theta = simplex.get_current_theta(for_analysis=True)
    assert np.all(simplex_theta >= 0.0)
    assert np.isclose(np.sum(simplex_theta), 1.0)

    stiefel = make_environment(
        {
            "dim": 8,
            "drift": {"type": "stiefel", "d": 4, "r": 2, "sigma": 0.02},
            "landscape": {"type": "stiefel", "d": 4, "r": 2},
        },
        seed=6,
    )
    Theta = stiefel.get_current_theta(for_analysis=True).reshape(4, 2)
    assert np.allclose(Theta.T @ Theta, np.eye(2), atol=1e-10)

    grassmann = make_environment(
        {
            "dim": 8,
            "drift": {"type": "stiefel", "d": 4, "r": 2, "sigma": 0.02},
            "landscape": {"type": "grassmann", "d": 4, "r": 2},
        },
        seed=7,
    )
    Theta = grassmann.get_current_theta(for_analysis=True).reshape(4, 2)
    assert np.allclose(Theta.T @ Theta, np.eye(2), atol=1e-10)


def test_stiefel_rejects_coordinate_bounds():
    d, r = 3, 2
    theta0 = StiefelLandscape.random_point(d, r, seed=40)
    env = DynamicEnvironment(
        dim=d * r,
        drift=StationaryDrift(),
        landscape=StiefelLandscape(d=d, r=r),
        initial_theta=theta0,
        bounds=(-1.0, 1.0),
    )
    with pytest.raises(ValueError, match="bounds=None"):
        WindGymEnv(env, oracle=FirstOrderOracle(env))


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
