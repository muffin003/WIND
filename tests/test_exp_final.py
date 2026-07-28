import numpy as np

from wind_benchmark.expFinal import (
    ExperimentProfile,
    ZeroOrderCase,
    ZeroOrderHyperparameters,
    ZeroOrderMethodConfig,
    _run_zero_order_method,
    _zero_order_dimension_cases,
)


def test_zero_order_batch_query_costs_are_explicit():
    dim = 20
    assert ZeroOrderMethodConfig("fo", "first_order").query_cost(dim) == 1
    assert ZeroOrderMethodConfig("spsa", "spsa").query_cost(dim) == 2
    assert ZeroOrderMethodConfig("g5", "gaussian", directions=5).query_cost(dim) == 10
    assert ZeroOrderMethodConfig("fd", "coordinate").query_cost(dim) == 2 * dim


def test_stationary_frozen_and_streaming_runs_are_paired():
    case = ZeroOrderCase("paired_stationary", 20, 5.0, "stationary")
    method = ZeroOrderMethodConfig("Gaussian-m5", "gaussian", directions=5)
    parameters = ZeroOrderHyperparameters(learning_rate=0.05, smoothing=0.1)
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
