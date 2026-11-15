"""Explicit privilege only for internal component-unit tests, never policy tests."""

import pytest


@pytest.fixture
def trusted_core_unit():
    # These selected tests exercise the evaluator implementation and mathematical
    # oracle contracts directly. This is test instrumentation of library internals,
    # not a supported application API. Public privacy tests must NOT use it.
    from wind_benchmark.core.privacy import _access

    token = _access.set(True)
    try:
        yield
    finally:
        _access.reset(token)
