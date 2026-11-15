"""Subprocess entrypoint for coordinator-death and installed-wheel tests."""

import sys
from pathlib import Path

from parallel_fixtures import slow_factory
from wind_benchmark.experiments.standard import (
    Method,
    Profile,
    Task,
    catalog_method,
    run_suite,
)

if __name__ == "__main__":
    mode, output = sys.argv[1:]
    method = (
        Method("Slow", "first-order", ("euclidean",), slow_factory, rng_key="slow")
        if mode == "slow"
        else catalog_method("SGD")
    )
    run_suite(
        Path(output),
        Profile("probe", 41, (20001, 20002), 100, 3),
        [method],
        [Task("clean", "reference")],
        workers=2,
        plots=False,
        progress=False,
    )
