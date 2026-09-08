"""WIND: benchmark for stochastic optimization in dynamic environments."""

from .benchmark import BatchRunner, BenchmarkRunner, ExperimentResult
from .catalog import (
    accepted_optimizer_names,
    canonical_optimizer_names,
    create_optimizer,
)
from .core import DynamicEnvironment, make_environment
from .oracle import FirstOrderOracle, HybridOracle, ZeroOrderOracle

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
]

__version__ = "0.3.0"
