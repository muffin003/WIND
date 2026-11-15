"""Trusted, deferred evaluation of a series of Gymnasium episodes."""

from copy import deepcopy
from dataclasses import dataclass

from wind_benchmark.core.oracle import EvaluationRecord
from wind_benchmark.core.privacy import (
    TruthGuard,
    _internal,
    _evaluation,
    _check,
    _bind,
    _owner_for_session,
    _Lease,
)


@dataclass(frozen=True)
class GymEvaluationStep(TruthGuard):
    evaluation: EvaluationRecord
    error: float
    constraint_violation: float
    step: int  # zero is the reset observation, not a rewarded transition

    @property
    def regret(self):
        return self.evaluation.clean_value - self.evaluation.optimum_value


class GymEvaluationSession(TruthGuard):
    """Evaluator-owned handle. Never give this object to a policy.

    Records are released only when finish() seals the whole series. Sealing also
    disables further interaction with every attached environment. Exceptions
    inside the context manager discard the journal.
    """

    _public_during_run = frozenset(
        {"finish", "abort", "records", "__enter__", "__exit__"}
    )

    @_internal
    def __init__(self):
        self.__episodes = []
        self.__complete = []
        self.__state = "open"
        self.__owner = _owner_for_session()
        if not isinstance(self.__owner, _Lease):
            self.__owner._start()
        _bind(self, self.__owner)

    def _check_open(self):
        if self.__state != "open":
            raise RuntimeError("Evaluation session is closed")

    def _begin(self):
        self._check_open()
        self.__episodes.append([])
        self.__complete.append(False)
        return len(self.__episodes) - 1

    def _append(self, episode, record):
        self._check_open()
        _bind(record, self.__owner)
        _bind(record.evaluation, self.__owner)
        self.__episodes[episode].append(record)

    def _complete(self, episode):
        self._check_open()
        self.__complete[episode] = True

    def _take_completed_records(self, episode):
        """Trusted evaluator compaction; never an algorithm-facing capability.

        Large studies keep compact numeric histories instead of millions of
        Python record objects. Their summaries remain sealed by the outer series.
        """
        from wind_benchmark.core.privacy import _is_evaluator

        if not _is_evaluator():
            raise PermissionError("Only the internal evaluator may compact a journal")
        self._check_open()
        if not self.__complete[episode]:
            raise RuntimeError("Cannot compact an unfinished episode")
        records = self.__episodes[episode]
        self.__episodes[episode] = []
        return records

    @_internal
    def finish(self):
        self._check_open()
        if not all(self.__complete):
            self.abort()
            raise RuntimeError("Cannot release an incomplete evaluation series")
        self.__state = "finished"
        if isinstance(self.__owner, _Lease):
            self.__owner._finish(True)
        else:
            self.__owner._end(True, count=False)

    @_internal
    def abort(self):
        if self.__state == "aborted":
            return
        if self.__state == "finished":
            raise RuntimeError("A finished evaluation session cannot be aborted")
        was_open = self.__state == "open"
        self.__episodes.clear()
        self.__complete.clear()
        self.__state = "aborted"
        if isinstance(self.__owner, _Lease):
            self.__owner._finish(False)
        elif was_open:
            self.__owner._end(False, count=False)

    @property
    def records(self):
        with _evaluation():
            if self.__state != "finished":
                raise RuntimeError(
                    "Ground truth is available only after the series finishes"
                )
        _check(self)
        with _evaluation():
            return deepcopy(tuple(tuple(episode) for episode in self.__episodes))

    @_internal
    def __enter__(self):
        self._check_open()
        return self

    @_internal
    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.abort()
        elif self.__state == "open":
            self.finish()
        return False
