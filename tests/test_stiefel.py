"""Tests for the Riemannian (Stiefel) add-on: manifold ops, landscape, drift,
metric, the reset fix, and end-to-end tracking with Riemannian SGD."""

import numpy as np
import pytest

from wind_benchmark.core import (
    DynamicEnvironment,
    StationaryDrift,
    StiefelDrift,
    StiefelLandscape,
    QuadraticLandscape,
    make_drift,
    make_landscape,
)
from wind_benchmark.oracle import FirstOrderOracle
from wind_benchmark.metrics import MetricsCollection, StiefelGeodesicMetric
from wind_benchmark.benchmark import BenchmarkRunner
from wind_benchmark.manifold import (
    random_stiefel,
    project_to_stiefel,
    tangent_project,
    retract,
    geodesic_distance,
    cayley_orthogonal,
    RiemannianSGD,
)


# ----------------------------------------------------------------- manifold ops
def test_random_stiefel_is_orthonormal():
    rng = np.random.default_rng(0)
    X = random_stiefel(5, 2, rng)
    assert X.shape == (5, 2)
    assert np.allclose(X.T @ X, np.eye(2), atol=1e-10)


def test_projection_idempotent_on_manifold():
    rng = np.random.default_rng(1)
    X = random_stiefel(4, 2, rng)
    assert np.allclose(project_to_stiefel(X), X, atol=1e-8)  # already on manifold
    M = rng.normal(size=(4, 2))
    P = project_to_stiefel(M)
    assert np.allclose(P.T @ P, np.eye(2), atol=1e-10)


def test_tangent_vector_satisfies_constraint():
    rng = np.random.default_rng(2)
    X = random_stiefel(5, 3, rng)
    G = rng.normal(size=(5, 3))
    Z = tangent_project(X, G)
    XtZ = X.T @ Z
    assert np.allclose(XtZ + XtZ.T, np.zeros((3, 3)), atol=1e-10)  # skew-symmetric


def test_geodesic_distance_zero_at_same_point():
    rng = np.random.default_rng(3)
    X = random_stiefel(4, 2, rng)
    # arccos has infinite slope at 1, so the self-distance is ~1e-8, not ~1e-16.
    assert geodesic_distance(X, X) < 1e-6
    Y = random_stiefel(4, 2, np.random.default_rng(99))
    assert geodesic_distance(X, Y) > 1e-3


def test_cayley_is_orthogonal():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(4, 4))
    A = A - A.T
    Q = cayley_orthogonal(A)
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10)


# --------------------------------------------------------------- landscape/drift
def test_stiefel_landscape_invariant():
    land = StiefelLandscape(d=4, r=2)
    theta = StiefelLandscape.random_point(4, 2, seed=5)
    assert abs(land.loss(theta, theta)) < 1e-12
    assert np.allclose(land.grad(theta, theta), 0.0)
    x = StiefelLandscape.random_point(4, 2, seed=6)
    assert land.loss(x, theta) >= 0.0


def test_stiefel_drift_stays_on_manifold_and_resets():
    d, r = 5, 2
    drift = StiefelDrift(d=d, r=r, sigma=0.05, seed=7)
    theta = StiefelLandscape.random_point(d, r, seed=8)

    def rollout():
        th = theta.copy()
        seq = []
        for t in range(30):
            th = drift.step(th, t)
            seq.append(th.copy())
        return np.array(seq)

    seq1 = rollout()
    # Still orthonormal after many steps
    last = seq1[-1].reshape(d, r)
    assert np.allclose(last.T @ last, np.eye(r), atol=1e-8)
    # reset -> reproducible
    drift.reset()
    seq2 = rollout()
    assert np.allclose(seq1, seq2)


def test_factories_build_stiefel():
    assert isinstance(make_landscape("stiefel", d=4, r=2), StiefelLandscape)
    assert isinstance(make_drift("stiefel", d=4, r=2, seed=1), StiefelDrift)


# ------------------------------------------------------------------- reset fix
def test_environment_reset_restores_initial_theta():
    init = StiefelLandscape.random_point(4, 2, seed=11)
    env = DynamicEnvironment(
        dim=8,
        drift=StationaryDrift(),
        landscape=StiefelLandscape(d=4, r=2),
        initial_theta=init,
    )
    env.step()
    env.reset()  # must restore the on-manifold start, not zeros
    assert np.allclose(env.get_current_theta(for_analysis=True), init)


# --------------------------------------------------------------------- metric
def test_stiefel_geodesic_metric_zero_when_tracking():
    metric = StiefelGeodesicMetric(d=4, r=2)
    theta = StiefelLandscape.random_point(4, 2, seed=12)
    metric.update(0, theta.copy(), theta.copy(), observation=None, environment=None)
    assert metric._current_value < 1e-9
    assert metric.direction == "minimize"


# ------------------------------------------------------------------ end-to-end
def test_riemannian_sgd_tracks_drifting_optimum():
    d, r = 5, 2
    dim = d * r
    theta0 = StiefelLandscape.random_point(d, r, seed=20)
    x0 = StiefelLandscape.random_point(d, r, seed=21)

    env = DynamicEnvironment(
        dim=dim,
        drift=StiefelDrift(d=d, r=r, sigma=0.02, seed=22),
        landscape=StiefelLandscape(d=d, r=r),
        initial_theta=theta0,
        bounds=None,  # never clip — would break orthonormality
    )
    oracle = FirstOrderOracle(env)  # gradient only (blind value), which R-SGD needs
    metrics = MetricsCollection([StiefelGeodesicMetric(d=d, r=r)])
    runner = BenchmarkRunner(env, oracle, metrics, record_trajectory=True)

    opt = RiemannianSGD(d=d, r=r, lr=0.5)
    result = runner.run(opt, T=150, x0=x0, seed=42)

    hist = result.metrics_history["stiefel_geodesic"]
    n = len(hist)
    head = float(np.mean(hist[: n // 5]))
    tail = float(np.mean(hist[-n // 5 :]))
    assert result.status == "SUCCESS"
    assert tail < head  # the optimizer actually closes the gap and tracks
    assert tail < 0.5  # and stays close to the drifting optimum

    # The optimizer keeps its iterates on the manifold.
    xT = np.array(result.trajectory["x"][-1]).reshape(d, r)
    assert np.allclose(xT.T @ xT, np.eye(r), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
