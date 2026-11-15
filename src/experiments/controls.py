"""Explicit controls for custom geometry studies, not catalog baselines."""

from wind_benchmark.algorithms.protocol import AdvanceResult, TaskSpec
from wind_benchmark.algorithms.catalog import create_optimizer
from wind_benchmark.core.geometry import project_to_stiefel


class ProjectedAmbientControl:
    """Catalog SGD followed by polar projection; not Riemannian SGD."""

    geometries = ("stiefel", "grassmann")

    def __init__(self, lr):
        self.algorithm = create_optimizer("SGD", {"lr": lr})

    def initialize(self, x0, rng, task_spec):
        self.shape = task_spec.matrix_shape
        self.algorithm.initialize(x0, rng, TaskSpec(task_spec.dimension))

    def advance(self, client):
        update = self.algorithm.advance(client)
        point = project_to_stiefel(update.recommendation.reshape(self.shape)).ravel()
        self.algorithm.x = point.copy()
        self.algorithm.recommendation = point.copy()
        return AdvanceResult(
            point, update.update_complete, update.queries, update.estimated_gradient
        )
