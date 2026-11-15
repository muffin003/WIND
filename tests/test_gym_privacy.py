"""No privileged state in the public Gym API; deferred series-level release."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from wind_benchmark.core.environment import (
    DynamicEnvironment,
    StationaryDrift,
    QuadraticLandscape,
)
from wind_benchmark.core.oracle import FirstOrderOracle, ZeroOrderOracle
from wind_benchmark.integrations.gym_env import WindGymEnv
from wind_benchmark.integrations.gym_evaluation import GymEvaluationSession


def make_env(session=None, **kwargs):
    core = DynamicEnvironment(
        dim=2,
        drift=StationaryDrift(),
        landscape=QuadraticLandscape(2),
        initial_theta=np.ones(2),
    )
    return WindGymEnv(core, T=2, evaluation_session=session, **kwargs)


def test_no_truth_in_public_attributes_info_or_unwrapped():
    env = make_env()
    for candidate in (
        env,
        env.unwrapped,
        gym.wrappers.TimeLimit(env, max_episode_steps=2).unwrapped,
    ):
        for field in (
            "env",
            "environment",
            "oracle",
            "theta",
            "get_current_theta",
            "evaluation_session",
            "evaluation_log",
            "records",
        ):
            assert not hasattr(candidate, field)
    obs, info = env.reset()
    allowed = {
        "t",
        "geometry",
        "constraint_violation",
        "query_count",
        "value_queries",
        "gradient_queries",
    }
    assert set(info) == allowed
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert set(info) == allowed
        assert reward == -float(obs[2])
        assert not terminated
    assert truncated


def test_reward_uses_noisy_feedback_even_when_positive():
    from wind_benchmark.core.environment import GaussianNoise

    core = DynamicEnvironment(
        dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(2)
    )
    session = GymEvaluationSession()
    env = WindGymEnv(
        core,
        ZeroOrderOracle(core, value_noise=GaussianNoise(10.0, seed=4)),
        T=1,
        evaluation_session=session,
    )
    env.reset()
    obs, reward, _, _, _ = env.step(np.ones(2))
    assert obs[2] < 0 and reward == -float(obs[2])
    session.finish()
    assert session.records[0][1].regret == pytest.approx(1.0)


@pytest.mark.parametrize("reward", ["neg_regret", "neg_error"])
def test_clean_rewards_are_rejected(reward):
    with pytest.raises(ValueError, match="evaluation-only"):
        make_env(reward=reward)


def test_true_error_termination_is_rejected():
    with pytest.raises(ValueError, match="hidden-state"):
        make_env(terminate_on_diverge=1.0)


def test_blind_gradients_require_explicit_reward():
    core = DynamicEnvironment(
        dim=2, drift=StationaryDrift(), landscape=QuadraticLandscape(2)
    )
    with pytest.raises(ValueError, match="gradient-only"):
        WindGymEnv(core, FirstOrderOracle(core))
    env = WindGymEnv(core, FirstOrderOracle(core), reward="zero", T=1)
    env.reset()
    assert env.step(np.ones(2))[1] == 0.0


def test_release_waits_for_whole_series_and_seals_environments():
    session = GymEvaluationSession()
    env = make_env(session)
    for episode in range(2):
        env.reset()
        for _ in range(2):
            with pytest.raises(RuntimeError, match="only after"):
                _ = session.records
            env.step(np.zeros(2))
        with pytest.raises(RuntimeError, match="only after"):
            _ = session.records
    session.finish()
    assert len(session.records) == 2
    for records in session.records:
        assert len(records) == 3  # reset plus two decisions
        assert [r.step for r in records] == [0, 1, 2]
        assert all(r.regret == pytest.approx(1.0) for r in records)
    for operation in (env.reset, lambda: env.step(np.zeros(2))):
        with pytest.raises(RuntimeError, match="closed"):
            operation()


@pytest.mark.parametrize("failure", ["action", "reset", "close", "finish", "policy"])
def test_failed_or_incomplete_series_never_releases_records(failure):
    session = GymEvaluationSession()
    env = make_env(session)
    env.reset()
    env.step(np.zeros(2))
    if failure == "close":
        env.close()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            if failure == "action":
                env.step(np.ones(3))
            elif failure == "reset":
                env.reset()
            elif failure == "finish":
                session.finish()
            else:
                with session:
                    raise RuntimeError("policy failed")
    with pytest.raises(RuntimeError, match="only after"):
        _ = session.records


def test_non_finite_action_does_not_release_partial_journal():
    session = GymEvaluationSession()
    env = make_env(session)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([np.nan, 1.0]))
    with pytest.raises(RuntimeError, match="only after"):
        _ = session.records


def test_same_public_feedback_hides_different_true_states():
    from wind_benchmark.core.environment import QuantizedNoise

    sessions, outputs = [], []
    for theta in (np.zeros(2), np.ones(2)):
        core = DynamicEnvironment(
            dim=2,
            drift=StationaryDrift(),
            landscape=QuadraticLandscape(2),
            initial_theta=theta,
        )
        session = GymEvaluationSession()
        env = WindGymEnv(
            core,
            ZeroOrderOracle(core, value_noise=QuantizedNoise(10.0)),
            T=2,
            evaluation_session=session,
        )
        reset_obs, reset_info = env.reset()
        rollout = [env.step(np.zeros(2)) for _ in range(2)]
        outputs.append((reset_obs, reset_info, rollout))
        session.finish()
        sessions.append(session)
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1] == outputs[1][1]
    for a, b in zip(outputs[0][2], outputs[1][2]):
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
    assert sessions[0].records[0][1].regret == 0.0
    assert sessions[1].records[0][1].regret == pytest.approx(1.0)
