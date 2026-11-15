"""WIND: benchmark for stochastic optimization in dynamic environments."""

from wind_benchmark.core.runner import BatchRunner, BenchmarkRunner, ExperimentResult
from wind_benchmark.algorithms.catalog import (
    accepted_optimizer_names,
    canonical_optimizer_names,
    create_optimizer,
)
from wind_benchmark.core.environment import DynamicEnvironment, make_environment
from wind_benchmark.core.oracle import FirstOrderOracle, HybridOracle, ZeroOrderOracle
from wind_benchmark.core.privacy import EvaluationSession, GroundTruthAccessError

__all__ = [
    "BatchRunner",
    "BenchmarkRunner",
    "accepted_optimizer_names",
    "canonical_optimizer_names",
    "create_optimizer",
    "DynamicEnvironment",
    "ExperimentResult",
    "FirstOrderOracle",
    "HybridOracle",
    "ZeroOrderOracle",
    "make_environment",
    "EvaluationSession",
    "GroundTruthAccessError",
]

__version__ = "0.7.0"
