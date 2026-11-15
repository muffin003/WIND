import numpy as np
import pandas as pd
import pytest

from wind_benchmark.experiments.paper import (
    ExperimentProfile,
    ZeroOrderCase,
    ZeroOrderHyperparameters,
    ZeroOrderMethodConfig,
    _run_zero_order_method,
    _zero_order_dimension_cases,
    _phase_lag,
    _summarize_runs,
)


def test_threshold_initial_point_and_censoring_status():
    method = ZeroOrderMethodConfig("control", "first_order")
    params = ZeroOrderHyperparameters({"lr": 1e-15})
    for epsilon, expected in ((1.1, 0), (0.1, None)):
        case = ZeroOrderCase(
            "threshold", 2, 1.0, "stationary", threshold_epsilon=epsilon
        )
        _, metrics = _run_zero_order_method(
            method, "streaming", case, 4, np.arange(5), 17, params
        )
        assert metrics["threshold_n_valid"] == 5
        if expected is None:
            assert np.isnan(metrics["queries_to_threshold"])
            assert metrics["threshold_status"] == "right_censored"
            assert metrics["restricted_queries_to_threshold"] == 4
        else:
            assert metrics["queries_to_threshold"] == expected
            assert metrics["restricted_queries_to_threshold"] == expected


def test_undefined_phase_lag_and_single_observation_interval():
    assert np.isnan(_phase_lag(np.ones(12), np.ones(12), 4))
    frame = pd.DataFrame({"group": ["one"], "value": [3.0]})
    result = _summarize_runs(frame, ["group"], ["value"], 100, 0).iloc[0]
    assert result.value_mean == 3
    assert result.value_n_valid == 1
    assert np.isnan(result.value_ci_low)


def test_restricted_query_summary_rejects_mixed_censor_budgets():
    frame = pd.DataFrame(
        {
            "group": ["one", "one"],
            "queries_to_threshold": [1.0, np.nan],
            "threshold_reached": [1.0, 0.0],
            "threshold_budget": [10, 5],
            "restricted_queries_to_threshold": [1.0, 5.0],
        }
    )
    result = _summarize_runs(frame, ["group"], ["queries_to_threshold"], 100, 0).iloc[0]
    assert not result.threshold_budget_comparable
    assert result.threshold_n_censored == 1
    assert np.isnan(result.restricted_queries_to_threshold_mean)


def test_zero_order_batch_query_costs_are_explicit():
    dim = 20
    assert ZeroOrderMethodConfig("fo", "first_order").query_cost(dim) == 1
    assert ZeroOrderMethodConfig("spsa", "spsa").query_cost(dim) == 2
    assert ZeroOrderMethodConfig("g5", "spherical", directions=5).query_cost(dim) == 10
    assert ZeroOrderMethodConfig("fd", "coordinate").query_cost(dim) == 2 * dim


def test_stationary_frozen_and_streaming_runs_are_paired():
    case = ZeroOrderCase("paired_stationary", 20, 5.0, "stationary")
    method = ZeroOrderMethodConfig("Gaussian-m5", "spherical", directions=5)
    parameters = ZeroOrderHyperparameters({"lr": 0.05, "h": 0.1, "m": 5})
    checkpoints = np.linspace(0, 200, 21)

    frozen, frozen_metrics = _run_zero_order_method(
        method, "frozen", case, 200, checkpoints, 123, parameters
    )
    streaming, streaming_metrics = _run_zero_order_method(
        method, "streaming", case, 200, checkpoints, 123, parameters
    )

    np.testing.assert_allclose(frozen, streaming)
    assert frozen_metrics["failed"] == 0.0
    assert streaming_metrics["failed"] == 0.0


def test_dimension_study_holds_query_budget_and_total_drift_fixed():
    profile = ExperimentProfile.build("smoke")
    cases = _zero_order_dimension_cases(profile)

    assert {case.dim for case in cases} == {5, 20, 100}
    assert {(case.path_kind, case.dim) for case in cases} == {
        (path_kind, dim)
        for path_kind in ("stationary", "linear")
        for dim in (5, 20, 100)
    }
    for case in cases:
        if case.path_kind == "linear":
            assert np.isclose(
                case.drift_per_query * profile.zero_order_budget,
                1.0,
            )
