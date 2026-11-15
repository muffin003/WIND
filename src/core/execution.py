"""Library query execution for custom experimental loops.

An optimizer receives immutable observations or a scoped OracleClient only.
The session owns environment transitions, oracle locks, budget checks, and
evaluator records. It is an API boundary, not a hostile-Python sandbox.
"""

from contextlib import contextmanager

import numpy as np

from wind_benchmark.core.oracle import OracleClient
from wind_benchmark.core.privacy import (
    TruthGuard,
    GroundTruthAccessError,
    _internal,
    _evaluation,
    _public_scope,
    _bind,
    _bind_environment,
    _owner_for_session,
    _Lease,
    _is_evaluator,
    _validate_component,
)


class QueryBudgetExhausted(RuntimeError):
    pass


class OracleSession(TruthGuard):
    """Streaming queries or frozen updates on the SAME query-based time clock.

    In frozen mode a callback's queries share the target at its start. After
    unlocking, the environment advances once per consumed query, not once per
    update. This distinction preserves a fixed physical query horizon.
    """

    _public_during_run = frozenset(
        {
            "n_queries",
            "counts",
            "remaining",
            "client",
            "observe",
            "finish",
            "abort",
            "budget",
            "protocol",
        }
    )

    @_internal
    def __init__(self, oracle, budget, *, protocol="streaming"):
        _validate_component(oracle)
        if type(protocol) is not str or type(budget) not in (int, np.int32, np.int64):
            raise ValueError("Expected a string protocol and an integer query budget")
        if protocol not in {"streaming", "frozen"}:
            raise ValueError("Expected streaming or frozen query semantics")
        if int(budget) != budget or budget < 1:
            raise ValueError("Query budget must be a positive integer")
        if oracle.n_queries or oracle._step_active:
            raise ValueError("A session requires a fresh inactive oracle")
        self.__oracle = oracle
        self.__environment = oracle.environment
        self.budget = int(budget)
        self.protocol = protocol
        self.__records = []
        self.__closed = False
        self.__outcome = "running"
        self.__in_update = False
        self.__round_actions = []
        self.__owner = _owner_for_session()
        self.__registered = not isinstance(self.__owner, _Lease)
        _bind_environment(self.__environment, oracle, self.__owner)
        _bind(self, self.__owner)
        if self.__registered:
            self.__owner._start()

    @property
    @_internal
    def n_queries(self):
        return self.__oracle.n_queries

    @property
    @_internal
    def counts(self):
        return {
            "queries": self.__oracle.n_queries,
            "value_queries": self.__oracle.n_value_queries,
            "gradient_queries": self.__oracle.n_grad_queries,
        }

    @property
    def remaining(self):
        return self.budget - self.n_queries

    @_internal
    def __query(self, point):
        if self.__closed or not self.__in_update:
            raise RuntimeError("Queries are available only inside an active update")
        if self.remaining <= 0:
            raise QueryBudgetExhausted("Oracle query budget exhausted")
        point = np.asarray(point, dtype=float)
        if point.shape != (self.__environment.dim,):
            raise ValueError("Query must be a flat decision vector of environment size")
        oracle = self.__oracle
        env = self.__environment
        streaming = self.protocol == "streaming"
        if streaming:
            oracle.start_step(env.t)
        before = oracle.n_queries
        try:
            observation = oracle.query(point)
            record = oracle._take_evaluation()
            _bind(record, self.__owner)
            self.__records.append(record)
        finally:
            oracle._last_evaluation = None
            if streaming:
                oracle.end_step()
            for _ in range(oracle.n_queries - before):
                if streaming:
                    env.step(action=point)
                else:
                    self.__round_actions.append(point.copy())
        return observation

    @contextmanager
    def client(self):
        """Query-only capability valid for exactly one callback/update."""
        with _evaluation():
            if self.__closed or self.__in_update:
                raise RuntimeError("Session closed or update already active")
            frozen = self.protocol == "frozen"
            if frozen:
                self.__oracle.start_step(self.__environment.t)
            self.__in_update = True
            client = OracleClient(self.__query, lambda: self.remaining)
            _bind(client, self.__owner)
        failed = False
        try:
            with _public_scope():
                yield client
        except BaseException:
            failed = True
            raise
        finally:
            with _evaluation():
                try:
                    client._close()
                    self.__in_update = False
                    if frozen:
                        self.__oracle.end_step()
                        actions, self.__round_actions = self.__round_actions, []
                        for action in actions:
                            self.__environment.step(action=action)
                except BaseException:
                    failed = True
                    raise
                finally:
                    if failed:
                        self.abort()

    def observe(self, point):
        with self.client() as client:
            return client.query(point)

    def finish(self):
        """Seal the run, then release immutable records to the evaluator."""
        trusted = _is_evaluator()
        with _evaluation():
            if self.__in_update or self.__closed:
                raise GroundTruthAccessError(
                    "Cannot release an active or closed session"
                )
            if not trusted and self.remaining:
                self.abort()
                raise GroundTruthAccessError("Cannot release an incomplete run")
            self.__closed = True
            self.__outcome = "released"
            if isinstance(self.__owner, _Lease):
                self.__owner._finish(True)
            elif self.__registered:
                self.__owner._end(True)
                self.__registered = False
            return tuple(self.__records)

    @_internal
    def abort(self):
        if self.__outcome == "aborted":
            return
        if self.__outcome == "released":
            raise GroundTruthAccessError("A released session cannot be aborted")
        self.__closed = True
        self.__outcome = "aborted"
        self.__records.clear()
        if isinstance(self.__owner, _Lease):
            self.__owner._finish(False)
        elif self.__registered:
            self.__owner._end(False)
            self.__registered = False
