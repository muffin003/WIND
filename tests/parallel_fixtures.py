"""Importable custom algorithm fixtures for real spawn-process tests."""

import os
import pickle
import time

from wind_benchmark.algorithms.catalog import create_optimizer
from wind_benchmark.core.privacy import (
    EvaluationSession,
    GroundTruthAccessError,
    _is_evaluator,
)

# Import hooks must run as unprivileged user code, including inside workers.
assert not _is_evaluator()
_retained = []
_original_init = None


def rl_crash_worker(connection, cancel):
    os._exit(73)


def rl_inspection_worker(connection, cancel):
    """Real RL worker with adversarial retained-reference checks in callbacks."""
    from wind_benchmark.experiments import rl, parallel

    original_train, original_evaluate = rl._q_train, rl._evaluate_public
    retained = []

    def inspect(env):
        assert not _is_evaluator()
        for previous in [*retained, env]:
            for operation in (
                lambda: vars(previous),
                lambda: pickle.dumps(previous),
                lambda: previous._ResourceAllocationEnv__logs,
            ):
                try:
                    operation()
                except GroundTruthAccessError:
                    pass
                else:
                    raise AssertionError("RL worker released a retained environment")
        retained.append(env)

    def train(policy, env):
        inspect(env)
        return original_train(policy, env)

    def evaluate(policy, env):
        inspect(env)
        return original_evaluate(policy, env)

    rl._q_train, rl._evaluate_public = train, evaluate
    parallel._rl_worker(connection, cancel)


def _assert_sealed(env):
    for operation in (
        lambda: env._theta,
        lambda: env.get_current_theta(True),
        lambda: vars(env),
        lambda: pickle.dumps(env),
    ):
        try:
            operation()
        except GroundTruthAccessError:
            pass
        else:
            raise AssertionError(
                "Retained evaluator data was released between worker jobs"
            )


def inspected_factory(task, seed):
    global _original_init
    from wind_benchmark.core.environment import DynamicEnvironment

    assert not _is_evaluator()
    for env in _retained:
        _assert_sealed(env)
    # Deliberate unit-test instrumentation supplies retained references. It is
    # not how algorithms obtain environments through the supported API.
    if _original_init is None:
        _original_init = DynamicEnvironment.__init__

        def retain(self, *args, **kwargs):
            _original_init(self, *args, **kwargs)
            _retained.append(self)

        DynamicEnvironment.__init__ = retain
    try:
        with EvaluationSession():
            pass
    except GroundTruthAccessError:
        pass
    else:
        raise AssertionError("Factory gained evaluator authority")
    time.sleep((seed % 3) * 0.005)
    return create_optimizer("SGD")


def slow_factory(task, seed):
    time.sleep(60)
    return create_optimizer("SGD")


def crashed_factory(task, seed):
    os._exit(17)


def ordinary_factory(task, seed):
    return create_optimizer("SGD")


def delayed_factory(task, seed):
    time.sleep((seed % 3) * 0.02)
    return create_optimizer("SGD")


class CallbackError(Exception):
    def __str__(self):
        assert not _is_evaluator(), "Exception formatter inherited evaluator access"
        for env in _retained:
            _assert_sealed(env)
        return "exception formatter remained unprivileged"


def exception_factory(task, seed):
    inspected_factory(task, seed)
    raise CallbackError()


class InitializationError(CallbackError, ValueError):
    pass


class FailedInitialization:
    def initialize(self, x0, rng, task_spec):
        raise InitializationError()

    def advance(self, client):
        raise AssertionError("unreachable")


def initialization_error_factory(task, seed):
    inspected_factory(task, seed)
    return FailedInitialization()
