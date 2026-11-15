"""Public algorithm boundary for WIND 0.6 (no evaluator capabilities)."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """Public search-space properties, never a target, seed, or loss object."""

    dimension: int
    geometry: str = "euclidean"
    matrix_shape: tuple[int, int] | None = None

    def __post_init__(self):
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or self.dimension < 1
        ):
            raise ValueError("dimension must be a positive integer")
        if self.geometry not in {"euclidean", "simplex", "stiefel", "grassmann"}:
            raise ValueError("Unknown geometry")
        if self.geometry in {"stiefel", "grassmann"}:
            if self.matrix_shape is None:
                raise ValueError("Manifold tasks require matrix_shape=(d, r)")
            d, r = self.matrix_shape
            if not 1 <= r <= d or d * r != self.dimension:
                raise ValueError("matrix_shape does not match dimension")


@dataclass(frozen=True)
class AdvanceResult:
    """One attempted update; recommendations are NOT automatically queried."""

    recommendation: np.ndarray
    update_complete: bool
    queries: int
    estimated_gradient: np.ndarray | None = None
    estimation_point: np.ndarray | None = None

    def __post_init__(self):
        for name in ("recommendation", "estimated_gradient", "estimation_point"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float).copy()
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        if (
            isinstance(self.queries, bool)
            or not isinstance(self.queries, (int, np.integer))
            or self.queries < 0
        ):
            raise ValueError("queries must be a nonnegative integer")
        if not isinstance(self.update_complete, (bool, np.bool_)):
            raise ValueError("update_complete must be boolean")


class QueryClient(Protocol):
    @property
    def remaining(self) -> int: ...

    def query(self, x): ...


class OptimizerProtocol(Protocol):
    def initialize(
        self, x0: np.ndarray, rng: np.random.Generator, task_spec: TaskSpec
    ) -> None: ...

    def advance(self, client: QueryClient) -> AdvanceResult: ...


def require_algorithm(algorithm):
    if not all(
        callable(getattr(algorithm, name, None)) for name in ("initialize", "advance")
    ):
        raise TypeError(
            "Algorithm must implement initialize(x0, rng, task_spec) and "
            "advance(client) -> AdvanceResult. The client exposes query(x) and "
            "remaining; legacy step/reset interfaces are not supported."
        )
