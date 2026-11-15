"""Local, series-scoped ground-truth access control.

This is an API boundary, not a sandbox against code modification, debuggers or
memory inspection. Policies and user callbacks never inherit evaluator access.
An object's owner remains sealed across gaps between runs and across threads.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps, lru_cache
from pathlib import Path
import inspect
import threading
import weakref
import os
import sys


class GroundTruthAccessError(PermissionError):
    """Ground truth has not been released by the experiment owner."""


_access = ContextVar("wind_evaluator_access", default=False)
_callback = ContextVar("wind_user_callback", default=False)
_current = ContextVar("wind_evaluation_series", default=None)
_owners = {}
_lock = threading.RLock()
_library = Path(__file__).resolve().parents[1]
_modules = {
    "core/privacy.py",
    "core/runner.py",
    "core/environment.py",
    "core/oracle.py",
    "core/execution.py",
    "integrations/gym_env.py",
    "integrations/gym_evaluation.py",
    "experiments/standard.py",
    "experiments/paper.py",
    "experiments/rl.py",
    "integrations/rl_env.py",
    "experiments/configurable.py",
    "interfaces/web_runner.py",
    "core/serialization.py",
    "experiments/parallel.py",
}
_modules = {os.path.normcase(str(_library / name)) for name in _modules}


@lru_cache(maxsize=256)
def _trusted_file(filename):
    path = os.path.normcase(os.path.abspath(filename))
    return path in _modules


def _library_caller():
    """Do not turn an importable internal helper into another for_analysis flag."""
    frame = inspect.currentframe().f_back.f_back
    if not _trusted_file(frame.f_code.co_filename):
        raise GroundTruthAccessError("Evaluator authority is internal to WIND")


@contextmanager
def _evaluation():
    # contextlib invokes generators; inspect the original requester instead.
    frame = inspect.currentframe().f_back
    while frame and frame.f_globals.get("__name__") == "contextlib":
        frame = frame.f_back
    if frame is None or not _trusted_file(frame.f_code.co_filename):
        raise GroundTruthAccessError("Evaluator authority is internal to WIND")
    token = _access.set(True)
    try:
        yield
    finally:
        _access.reset(token)


def _user_call(function, *args, **kwargs):
    _library_caller()
    access = _access.set(False)
    callback = _callback.set(True)
    try:
        return function(*args, **kwargs)
    finally:
        _callback.reset(callback)
        _access.reset(access)


def _owner(obj):
    with _lock:
        entry = _owners.get(id(obj))
        return entry[1] if entry and entry[0]() is obj else None


def _bind(obj, owner=None):
    _library_caller()
    owner = owner or _current.get()
    if obj is None or owner is None:
        return obj
    with _lock:
        previous = _owner(obj)
        if (
            previous is not None
            and previous is not owner
            and previous.state == "running"
        ):
            raise GroundTruthAccessError("Object belongs to an active experiment")
        identity = id(obj)
        _owners[identity] = (
            weakref.ref(obj, lambda ref: _owners.pop(identity, None)),
            owner,
        )
    return obj


def _bind_environment(env, oracle=None, owner=None):
    _library_caller()
    with _evaluation():
        owner = owner or _current.get()
        _validate_component(env)
        objects = [env, env.drift, env.landscape, oracle]
        if oracle is not None:
            objects.extend((oracle.value_noise, oracle.grad_noise))
        # Validate the entire graph before binding any retained references.
        for obj in objects:
            _validate_component(obj)
            previous = _owner(obj)
            if (
                previous is not None
                and previous is not owner
                and previous.state == "running"
            ):
                raise GroundTruthAccessError("Object belongs to an active experiment")
        for obj in objects:
            _bind(obj, owner)


def _validate_component(obj):
    _library_caller()
    if obj is None:
        return
    cls = type(obj)
    allowed_modules = {
        f"{__package__.rsplit('.', 1)[0]}.{name}"
        for name in (
            "core.environment",
            "core.oracle",
            "core.metrics",
            "experiments.standard",
            "experiments.rl",
        )
    }
    module = sys.modules.get(cls.__module__)
    module_name = cls.__module__
    if module_name == "__main__" and _trusted_file(getattr(module, "__file__", "")):
        module_name = getattr(getattr(module, "__spec__", None), "name", None)
    if (
        module_name not in allowed_modules
        or getattr(module, cls.__name__, None) is not cls
    ):
        raise GroundTruthAccessError(
            "Custom evaluator components cannot receive live ground truth. "
            "Use built-in components and perform custom analysis after release. "
            "Custom algorithms remain supported."
        )


def _check(obj):
    if _access.get():
        return
    owner = _owner(obj)
    if owner is not None and owner.state != "released":
        raise GroundTruthAccessError(
            "Ground truth is available only after the entire series finishes"
        )


class TruthGuard:
    """Block ordinary attribute, vars(), repr-state and serialization access."""

    __slots__ = ()
    _public_during_run = frozenset()

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name not in {"__class__", "__weakref__", "__repr__", "__str__"}:
            if name not in type(self)._public_during_run:
                _check(self)
        return value

    def __setattr__(self, name, value):
        _check(self)
        object.__setattr__(self, name, value)


class _Lease:
    """Detached owner for a query session or a collection of Gym episodes."""

    __slots__ = ("__state",)

    def __getattribute__(self, name):
        if name.startswith("_Lease__"):
            _library_caller()
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        _library_caller()
        object.__setattr__(self, name, value)

    def __init__(self):
        _library_caller()
        self.__state = "running"

    @property
    def state(self):
        return self.__state

    def _finish(self, success):
        _library_caller()
        target = "released" if success else "aborted"
        if self.__state == target:
            return
        if self.__state != "running":
            raise GroundTruthAccessError(
                "A closed evaluation cannot change its outcome"
            )
        self.__state = target


def _owner_for_session():
    _library_caller()
    return _current.get() or _Lease()


@contextmanager
def _public_scope():
    access, callback = _access.set(False), _callback.set(True)
    try:
        yield
    finally:
        _callback.reset(callback)
        _access.reset(access)


def _is_evaluator():
    return _access.get()


class PendingResult(TruthGuard):
    """Result of a nested public helper; unwrap after the outer series closes."""

    _public_during_run = frozenset({"result"})

    def __init__(self, value):
        self.__value = value
        _bind(self)

    def result(self):
        _check(self)
        return self.__value


class EvaluationSession:
    """Explicit series boundary; intermediate results are locked until exit.

    Nested runners join this session. A failed session never releases its data.
    Optional expected_runs prevents releasing a declared series prematurely.
    """

    __slots__ = (
        "__state",
        "__active",
        "__completed",
        "__expected",
        "__token",
        "__pending",
        "__export_status",
        "__abort_callbacks",
    )

    def __getattribute__(self, name):
        if name.startswith("_EvaluationSession__"):
            _library_caller()
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        _library_caller()
        object.__setattr__(self, name, value)

    def __init__(self, expected_runs=None):
        if expected_runs is not None and (
            type(expected_runs) is not int or expected_runs < 1
        ):
            raise ValueError("expected_runs must be a positive integer")
        self.__state = "created"
        self.__active = 0
        self.__completed = 0
        self.__expected = expected_runs
        self.__token = None
        self.__pending = []
        self.__export_status = "pending"
        self.__abort_callbacks = []

    @property
    def state(self):
        return self.__state

    @property
    def export_status(self):
        return self.__export_status

    def __enter__(self):
        if _callback.get() or self.__state != "created" or _current.get() is not None:
            raise GroundTruthAccessError(
                "Cannot open or reopen an evaluation series here"
            )
        self.__state = "running"
        self.__token = _current.set(self)
        return self

    def finish(self):
        with _lock:
            if self.__state != "running" or self.__active or _callback.get():
                raise GroundTruthAccessError(
                    "Cannot release an active or closed experiment"
                )
            if self.__expected is not None and self.__completed != self.__expected:
                raise GroundTruthAccessError("Declared runs are not complete")
            self.__state = "released"
            pending, self.__pending = self.__pending, []
        try:
            for function, args, kwargs in pending:
                function(*args, **kwargs)
        except BaseException:
            self.__export_status = "failed"
            raise
        self.__export_status = "complete"

    def _publish_later(self, function, args, kwargs):
        _library_caller()
        if self.__state != "running":
            raise GroundTruthAccessError("Evaluation series is closed")
        self.__pending.append((function, args, kwargs))

    def _on_abort(self, function):
        _library_caller()
        self.__abort_callbacks.append(function)

    def _abort_worker(self):
        """Seal a private worker session permanently, never release its objects."""
        _library_caller()
        if self.__state == "released":
            raise GroundTruthAccessError("A released session cannot be aborted")
        self.__state = "aborted"
        self.__pending.clear()

    def _start(self):
        _library_caller()
        with _lock:
            if self.__state != "running":
                raise GroundTruthAccessError("Evaluation series is closed")
            self.__active += 1

    def _end(self, success, count=True):
        _library_caller()
        with _lock:
            if self.__active < 1 or self.__state == "released":
                raise GroundTruthAccessError(
                    "Unbalanced or already released evaluation session"
                )
            self.__active -= 1
            self.__completed += int(success and count)
            if not success:
                self.__state = "aborted"

    def __exit__(self, kind, value, traceback):
        try:
            if kind is not None and self.__state != "released":
                self.__state = "aborted"
                self.__pending.clear()
            elif self.__state == "running":
                try:
                    self.finish()
                except BaseException:
                    if self.__state != "released":
                        self.__state = "aborted"
                    raise
            elif self.__state == "aborted":
                raise GroundTruthAccessError(
                    "Evaluation series was aborted; no results released"
                )
        finally:
            try:
                if self.__state == "aborted":
                    self.__pending.clear()
                    for callback in self.__abort_callbacks:
                        try:
                            _user_call(callback)
                        except Exception as exc:
                            # Operational status is best effort on a failed disk.
                            # It must not strand the caller in a sealed context.
                            if value is not None:
                                value.add_note(
                                    f"Abort status update failed: {type(exc).__name__}"
                                )
            finally:
                self.__abort_callbacks.clear()
                _current.reset(self.__token)


def _series(function):
    """Trusted orchestration; user code must be invoked through _user_call."""
    _library_caller()

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _callback.get():
            raise GroundTruthAccessError(
                "A policy cannot start a privileged evaluation"
            )
        if _current.get() is not None:
            if _current.get().state != "running":
                raise GroundTruthAccessError("Evaluation series is closed")
            external = not _access.get()
            with _evaluation():
                result = function(*args, **kwargs)
                return (
                    PendingResult(result)
                    if external and not isinstance(result, TruthGuard)
                    else result
                )
        with EvaluationSession():
            with _evaluation():
                return function(*args, **kwargs)

    return wrapped


def _run(function):
    _library_caller()

    @_series
    @wraps(function)
    def wrapped(*args, **kwargs):
        owner = _current.get()
        owner._start()
        success = False
        try:
            result = function(*args, **kwargs)
            success = True
            return result
        finally:
            owner._end(success)

    return wrapped


def _internal(function):
    """A library endpoint returning only public data, not a generic authority API."""
    _library_caller()

    @wraps(function)
    def wrapped(*args, **kwargs):
        if function.__name__ == "__init__" and _callback.get():
            # An already-entered library endpoint may build its next scheduled
            # episode. User callbacks themselves never have evaluator access.
            caller = inspect.currentframe().f_back
            internal_construction = _access.get() and _trusted_file(
                caller.f_code.co_filename
            )
            if not internal_construction:
                raise GroundTruthAccessError(
                    "A policy cannot create a privileged evaluator"
                )
        with _evaluation():
            return function(*args, **kwargs)

    return wrapped


def _after_release(function, *args, **kwargs):
    _library_caller()
    owner = _current.get()
    if owner is not None and owner.state == "running":
        owner._publish_later(function, args, kwargs)
        return None
    if owner is not None and owner.state != "released":
        raise GroundTruthAccessError("Aborted series cannot publish results")
    return function(*args, **kwargs)


def _on_series_abort(function):
    _library_caller()
    owner = _current.get()
    if owner is not None:
        owner._on_abort(function)


def _publish(function):
    _library_caller()

    @wraps(function)
    def wrapped(*args, **kwargs):
        return _after_release(function, *args, **kwargs)

    return wrapped
