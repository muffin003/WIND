"""Versioned random streams; presentation labels must never be stream keys."""

from contextlib import contextmanager
from functools import wraps
from copy import deepcopy
import hashlib
import inspect
import json
import random

import numpy as np

RNG_PROTOCOL = "named_streams_v2"


def stream_seed(seed: int, *identity) -> int:
    payload = json.dumps([int(seed), *identity], separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def reseed_component(component, seed: int):
    """Reset built-in random components, or honor a custom reseed(seed) hook."""
    if component is None:
        return
    hook = getattr(component, "reseed", None)
    if callable(hook):
        hook(seed)
        return
    if isinstance(getattr(component, "rng", None), np.random.Generator):
        component.rng = np.random.default_rng(seed)
        component._rng_state = deepcopy(component.rng.bit_generator.state)
        component._resolved_seed = int(seed)
        if hasattr(component, "_generate_matrix"):
            component.A = component._generate_matrix()
    if hasattr(component, "_seed"):
        component._seed = int(seed)
        component._resolved_seed = int(seed)
        if hasattr(component, "_M"):
            component._M = None
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


def reseed_environment(environment, seed: int, *, landscape=False):
    reseed_component(environment.drift, stream_seed(seed, "drift"))
    if landscape:
        reseed_component(environment.landscape, stream_seed(seed, "landscape"))


def reseed_oracle(oracle, seed: int):
    reseed_component(oracle.value_noise, stream_seed(seed, "value_noise"))
    reseed_component(oracle.grad_noise, stream_seed(seed, "gradient_noise"))
    oracle.rng = np.random.default_rng(stream_seed(seed, "oracle"))
    oracle._resolved_seed = int(seed)


@contextmanager
def global_seed(seed):
    """Compatibility for old optimizers, without changing the caller's RNG."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def factory_accepts_seed(factory):
    parameters = inspect.signature(factory).parameters
    return "seed" in parameters or any(
        p.kind == p.VAR_KEYWORD for p in parameters.values()
    )


def seeded_factory(factory, seed: int, *args, **kwargs):
    """Prefer explicit factory seeds; old call signatures remain supported."""
    with global_seed(seed):
        return factory(
            *args,
            **({**kwargs, "seed": seed} if factory_accepts_seed(factory) else kwargs),
        )


def preserve_global_rng(function):
    """Contain compatibility RNG mutation, including exceptions from user hooks."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            return function(*args, **kwargs)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    return wrapped
