"""Canonical scope, artifact checks and shared execution contract."""

import json
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from wind_benchmark.algorithms.protocol import AdvanceResult
from wind_benchmark.experiments.canonical import specification, validate_result
from wind_benchmark.experiments.standard import (
    Profile,
    default_tasks,
    catalog_method,
    run_suite,
)
from wind_benchmark.algorithms.catalog import canonical_optimizer_names
from wind_benchmark.core.resources import ResourceSampler


def test_canonical_scope_is_fixed_and_splits_disjoint():
    spec = specification()
    assert len(spec["tasks"]) == 32
    assert set(spec["development_seeds"]).isdisjoint(spec["evaluation_seeds"])
    methods = [catalog_method(n) for n in canonical_optimizer_names()]
    compatible = sum(
        t.geometry in m.geometries for t in default_tasks() for m in methods
    )
    assert compatible * 50 == 22450
    assert compatible * 50 * spec["query_budget"] == 89800000


def test_custom_scope_is_not_mislabelled_canonical(tmp_path):
    output = tmp_path / "custom"
    result = run_suite(
        output,
        replace(Profile.build("smoke"), budget=40),
        [catalog_method("SGD")],
        default_tasks()[:1],
        plots=False,
        progress=False,
    )
    assert result["manifest"]["run_kind"] == "custom_study"
    validation = validate_result(output)
    assert not validation["valid"]
    assert "Custom study is not a canonical result" in validation["errors"]


def test_resource_sampler_is_safe_for_repeated_runs():
    for _ in range(12):
        sampler = ResourceSampler()
        sampler.start()
        allocation = np.ones(10000)
        result = sampler.finish()
        assert result["sampled_peak_rss_bytes"] >= result["rss_baseline_bytes"] > 0
        assert not sampler.thread.is_alive() and allocation.size == 10000


def test_old_configurations_require_explicit_new_schema(tmp_path):
    from wind_benchmark.interfaces.cli import load_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"optimizers": ["Nesterov"]}))
    with pytest.raises(ValueError, match="schema_version=3"):
        load_config(path)
    path.write_text(json.dumps({"schema_version": 3, "optimizers": ["Nesterov"]}))
    assert load_config(path)["schema_version"] == 3


def test_partial_batches_have_strict_integer_counts():
    with pytest.raises(ValueError):
        AdvanceResult(np.zeros(2), False, 0.5)
    with pytest.raises(ValueError):
        AdvanceResult(np.zeros(2), "yes", 1)


def test_batch_accepts_declared_zero_initialization_for_rda():
    from wind_benchmark.core.runner import BatchRunner
    from wind_benchmark.core.environment import (
        DynamicEnvironment,
        QuadraticLandscape,
        StationaryDrift,
    )
    from wind_benchmark.core.oracle import FirstOrderOracle
    from wind_benchmark.core.metrics import MetricsCollection, TrackingErrorMetric

    batch = BatchRunner(
        lambda seed: DynamicEnvironment(
            2, StationaryDrift(), QuadraticLandscape(2), np.ones(2)
        ),
        FirstOrderOracle,
        lambda: MetricsCollection([TrackingErrorMetric()]),
    )
    result = batch.run(
        lambda: catalog_method("RDA").factory(None, 0),
        [31, 32],
        T=3,
        initial_x=np.zeros(2),
    )
    assert len(result) == 2
    assert all(r.metadata["initialization"]["kind"] == "fixed" for r in result)
    assert all(r.metadata["time_grid"]["size"] == 3 for r in result)

    adaptive = batch.run_adaptive(
        lambda: catalog_method("RDA").factory(None, 0),
        T=3,
        min_runs=2,
        max_runs=2,
        initial_x=np.zeros(2),
    )
    assert len(adaptive) == 2
    assert all(r.metadata["initial_x"] == [0.0, 0.0] for r in adaptive)
