"""Q-learning formula and external PPO's interaction contract."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from wind_benchmark.experiments.rl import QLearning, crossed_bootstrap
from wind_benchmark.integrations.rl_env import (
    ResourceAllocationEnv,
    EpisodeSpec,
    bounded_vec_env_class,
)
from wind_benchmark.integrations.gym_evaluation import GymEvaluationSession
from wind_benchmark import GroundTruthAccessError


def schedule(episodes=2, horizon=5):
    return [
        EpisodeSpec("stationary", 4, horizon, 300 + i, 900 + i, "none")
        for i in range(episodes)
    ]


def test_q_online_formula_no_clipping_and_truncation_bootstraps():
    q = QLearning(2)
    obs, following = np.zeros(5), np.ones(5)
    state, next_state = q.state_index(obs), q.state_index(following)
    q.q_values[state, 0] = 2
    q.q_values[next_state, :] = 3
    q.update(obs, 0, 100, following, terminated=False)
    assert q.q_values[state, 0] == pytest.approx(2 + 0.1 * (100 + 0.95 * 3 - 2))
    q.q_values[state, 0] = 2
    q.update(obs, 0, 100, following, terminated=True)
    assert q.q_values[state, 0] == pytest.approx(2 + 0.1 * (100 - 2))


def test_shared_features_action_reward_and_hidden_references():
    session = GymEvaluationSession()
    env = ResourceAllocationEnv(schedule(1), session)
    obs, info = env.reset()
    assert obs.shape == (5,) and obs.dtype == np.float32
    assert env.action_space.n == 12 and info == {"value_queries": 1}
    for name in (
        "_ResourceAllocationEnv__env",
        "_ResourceAllocationEnv__evaluation_session",
        "__dict__",
    ):
        with pytest.raises(GroundTruthAccessError):
            getattr(env, name)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(0)
        assert np.isfinite(reward) and obs.shape == (5,)
    assert truncated and not terminated
    assert env.queries == 6 and env.transitions == 5
    with pytest.raises(RuntimeError):
        env.reset()
    with pytest.raises(RuntimeError):
        env.step(0)
    with pytest.raises(RuntimeError):
        _ = session.records
    env.close()
    session.finish()
    assert len(session.records[0]) == 6


def test_crossed_bootstrap_pairing_and_training_variability():
    # Constant paired effect stays exactly -2 regardless of either resampling axis.
    a = np.arange(12).reshape(3, 4)
    values = np.stack([a, a + 2], axis=-1)
    mean, low, high = crossed_bootstrap(values[:, :, 0] - values[:, :, 1], 100)
    assert mean == low == high == -2
    with pytest.raises(ValueError):
        crossed_bootstrap(np.ones((1, 30)))


def test_q_policy_roundtrip_has_no_pickle_and_preserves_actions(tmp_path):
    q = QLearning(81)
    q.q_values[0, 3] = 19.0
    target = tmp_path / "q.npz"
    q.save(target)
    with np.load(target, allow_pickle=False) as payload:
        for name in payload.files:
            assert payload[name].dtype != object
    restored = QLearning.load(target)
    np.testing.assert_array_equal(q.q_values, restored.q_values)


def test_bounded_vec_env_has_no_final_reset_and_true_terminal_observation():
    pytest.importorskip("stable_baselines3")
    env = ResourceAllocationEnv(schedule())
    vec = bounded_vec_env_class()(env)
    assert vec.reset().shape == (1, 5)
    for index in range(10):
        observation, reward, done, infos = vec.step(np.array([0]))
        if index in (4, 9):
            assert done[0] and infos[0]["TimeLimit.truncated"]
            assert infos[0]["terminal_observation"].shape == (5,)
    assert env.queries == 12 and env.completed_episodes == 2
    with pytest.raises(RuntimeError):
        vec.step(np.array([0]))
    vec.close()


def test_real_ppo_exact_rollout_roundtrip_and_no_extra_initial_query(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from wind_benchmark.experiments.rl import _ppo
    import torch

    torch.set_num_threads(1)
    env = ResourceAllocationEnv(schedule(10, 100))
    vec = bounded_vec_env_class()(env)
    model = _ppo(vec, 42)
    assert env.queries == 0
    model.learn(1000)
    assert model.num_timesteps == 1000
    assert env.queries == 1010 and env.completed_episodes == 10
    observation = np.zeros(5, dtype=np.float32)
    before = model.predict(observation, deterministic=True)[0]
    model.save(tmp_path / "policy.zip")
    loaded = PPO.load(tmp_path / "policy.zip", device="cpu")
    np.testing.assert_array_equal(
        before, loaded.predict(observation, deterministic=True)[0]
    )
    vec.close()


def test_reward_uses_only_two_public_values_and_realized_switch(monkeypatch):
    from wind_benchmark.experiments import rl as expRL

    class PublicMeasurements:
        def reset(self):
            return np.array([0.25, 0.25, 0.25, 0.25, 0.7]), {
                "value_queries": 1,
                "gradient_queries": 0,
            }

        def step(self, action):
            return (
                np.array([0.15, 0.35, 0.25, 0.25, 0.4]),
                999.0,
                False,
                True,
                {"constraint_violation": 0.0},
            )

        def close(self):
            pass

    monkeypatch.setattr(expRL, "make_transfer_env", lambda *args: PublicMeasurements())
    env = ResourceAllocationEnv(schedule(1, 1))
    env.reset()
    _, reward, _, truncated, _ = env.step(0)
    # The underlying reward=999 is intentionally ignored. No truth exists here.
    assert reward == pytest.approx(10 * (-0.4 + 2 * (0.7 - 0.4) - 0.02 * 0.2))
    assert truncated and env.queries == 2
    env.close()


def test_ppo_transfer_resets_optimizer_but_checkpoints_preserve_it():
    pytest.importorskip("stable_baselines3")
    from wind_benchmark.experiments.rl import _ppo, _train
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source_env = ResourceAllocationEnv(schedule(10, 100))
        source = _ppo(bounded_vec_env_class()(source_env), 42)
        _train(source, "PPO", source_env)
        assert source.policy.optimizer.state

        first_env = ResourceAllocationEnv(schedule(10, 100))
        target = _ppo(bounded_vec_env_class()(first_env), 42)
        target.policy.load_state_dict(source.policy.state_dict())
        assert not target.policy.optimizer.state
        optimizer = target.policy.optimizer
        _train(target, "PPO", first_env)
        first_steps = [float(s["step"]) for s in optimizer.state.values()]
        next_env = ResourceAllocationEnv(schedule(10, 100))
        _train(target, "PPO", next_env, continuation=True)
        assert target.policy.optimizer is optimizer
        assert target.num_timesteps == 2000
        assert all(
            float(s["step"]) > first_steps[i]
            for i, s in enumerate(optimizer.state.values())
        )
        assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    finally:
        torch.set_num_threads(previous_threads)


def test_rl_restores_caller_rng_on_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    import random
    from wind_benchmark.experiments import rl as expRL

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()

    def broken(*args):
        np.random.random(7)
        random.random()
        torch.rand(7)
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(expRL, "_run_experiment", broken)
    with pytest.raises(RuntimeError, match="controlled failure"):
        expRL.run_experiment(None, None)
    restored = np.random.get_state()
    assert restored[0] == numpy_state[0] and restored[2:] == numpy_state[2:]
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_num_threads() == threads
