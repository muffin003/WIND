"""Independent finite-step references for every active 0.6 algorithm."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from wind_benchmark.algorithms.protocol import TaskSpec
from wind_benchmark.algorithms.catalog import (
    create_optimizer,
    canonical_optimizer_names,
)
from wind_benchmark.core.oracle import Observation

ATOL, RTOL = 1e-12, 1e-10


class Client:
    def __init__(self, gradients=None, budget=100, value=None):
        self.gradients = gradients
        self.budget = budget
        self.points = []
        self.value = value

    @property
    def remaining(self):
        return self.budget - len(self.points)

    def query(self, x):
        assert self.remaining > 0
        self.points.append(x.copy())
        index = len(self.points) - 1
        grad = None if self.gradients is None else self.gradients[index]
        return Observation(
            x=x,
            t=index,
            grad=grad,
            value=None if self.value is None else self.value(x),
            query_index=index + 1,
        )


def initialized(
    name, x0=(1.0, -2.0), parameters=None, geometry="euclidean", shape=None
):
    algorithm = create_optimizer(name, parameters)
    algorithm.initialize(
        np.array(x0), np.random.default_rng(123), TaskSpec(len(x0), geometry, shape)
    )
    return algorithm


def check(actual, expected):
    assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def test_catalog_has_exactly_22_without_legacy_aliases():
    assert len(canonical_optimizer_names()) == 22
    for name in ("LookaheadNesterov", "SGD_Polyak", "SMD", "ZOSGD", "GPUCB", "CMAES"):
        with pytest.raises(KeyError):
            create_optimizer(name)


@pytest.mark.parametrize(
    "name,expected,second_query",
    [
        ("SGD", [1.1, -2.6], [0.8, -2.4]),
        ("HeavyBall", [0.92, -2.96], [0.8, -2.4]),
        ("Nesterov", [0.96, -2.48], [0.81, -2.38]),
        ("SignSGD", [1.0, -2.1], [0.95, -2.05]),
        ("ProxSGD", [1.098, -2.598], [0.799, -2.399]),
    ],
)
def test_gradient_updates_two_steps(name, expected, second_query):
    alg = initialized(name)
    client = Client([np.array([2.0, 4.0]), np.array([-3.0, 2.0])])
    first, second = alg.advance(client), alg.advance(client)
    check(first.queries, 1)
    check(client.points[0], [1.0, -2.0])
    check(client.points[1], second_query)
    check(second.recommendation, expected)


def test_pr_average_never_feeds_back_and_uses_updated_iterates():
    alg = initialized("PolyakRuppertSGD")
    client = Client([np.array([2.0, 4.0]), np.array([-3.0, 2.0]), np.zeros(2)])
    a = np.array([0.8, -2.4])
    b = a - 0.1 * 2 ** (-0.75) * np.array([-3.0, 2.0])
    check(alg.advance(client).recommendation, a)
    check(alg.advance(client).recommendation, (a + b) / 2)
    alg.advance(client)
    check(client.points[2], b)
    assert not np.allclose(client.points[2], (a + b) / 2)


@pytest.mark.parametrize("name", ["Adam", "AdamW", "AMSGrad"])
def test_adaptive_moments_against_hand_expansion(name):
    alg = initialized(
        name, parameters={"beta1": 0.5, "beta2": 0.75, "eps": 0.1, "lr": 0.2}
    )
    client = Client([np.array([2.0, -4.0]), np.array([-1.0, 2.0])])
    x0 = np.array([1.0, -2.0])
    m1, v1 = np.array([1.0, -2.0]), np.array([1.0, 4.0])
    m2, v2 = np.array([0.0, 0.0]), np.array([1.0, 4.0])
    if name == "AMSGrad":
        first = x0 - 0.2 * m1 / (np.sqrt(v1) + 0.1)
        second = first - 0.2 * m2 / (np.sqrt(v2) + 0.1)
    else:
        first = x0 - 0.2 * np.array([2.0, -4.0]) / (np.array([2.0, 4.0]) + 0.1)
        second = first.copy()
        if name == "AdamW":
            first -= 0.2 * 0.01 * x0
            second = first * (1 - 0.2 * 0.01)
    check(alg.advance(client).recommendation, first)
    check(alg.m, m1)
    check(alg.v, v1)
    check(alg.advance(client).recommendation, second)
    check(alg.m, m2)
    check(alg.v, v2)
    if name == "AMSGrad":
        check(alg.vmax, v2)


def test_rda_square_root_schedule_and_initial_state():
    with pytest.raises(ValueError, match="x0=0"):
        initialized("RDA")
    alg = initialized("RDA", (0.0, 0.0), {"lr": 0.2, "lam": 0.5})
    client = Client([np.array([2.0, -0.1]), np.array([1.0, -0.1])])
    check(alg.advance(client).recommendation, [-0.3, 0])
    check(alg.advance(client).recommendation, [-0.4 / np.sqrt(2), 0])


@pytest.mark.parametrize("name", ["Adam", "AdamW", "AMSGrad"])
def test_adaptive_nonzero_momentum_after_variance_decreases(name):
    alg = initialized(
        name, (1.0,), {"lr": 0.2, "beta1": 0.5, "beta2": 0.75, "eps": 0.1}
    )
    client = Client([np.array([2.0]), np.array([0.0]), np.array([1.0])])
    # Explicit expansion, including a decreasing raw v on the second step.
    moments = [
        (1.0, 1.0, 1.0, 0.5, 0.25),
        (0.5, 0.75, 1.0, 0.75, 0.4375),
        (0.75, 0.8125, 1.0, 0.875, 0.578125),
    ]
    expected = 1.0
    for m, v, vmax, b1, b2 in moments:
        if name == "AMSGrad":
            expected -= 0.2 * m / (np.sqrt(vmax) + 0.1)
        else:
            previous = expected
            expected -= 0.2 * (m / b1) / (np.sqrt(v / b2) + 0.1)
            if name == "AdamW":
                expected -= 0.2 * 0.01 * previous
        check(alg.advance(client).recommendation, [expected])
        check(alg.v, [v])
        if name == "AMSGrad":
            check(alg.vmax, [vmax])


def test_haar_qr_column_signs_follow_positive_r_diagonal():
    from wind_benchmark.core.geometry import random_stiefel

    seed = 881
    matrix = np.random.default_rng(seed).normal(size=(6, 2))
    frame = random_stiefel(6, 2, np.random.default_rng(seed))
    triangular = frame.T @ matrix
    assert np.all(np.diag(triangular) > 0)
    check(triangular[np.tril_indices(2, -1)], 0)
    check(frame @ triangular, matrix)


@pytest.mark.parametrize(
    "name", ["SPSA", "GaussianZOSGD", "SphericalZOSGD", "KieferWolfowitz"]
)
def test_paired_estimators_and_exact_query_count(name):
    d = 3
    params = {"lr": 0.1, "h": 0.02}
    m = 2
    if name not in {"KieferWolfowitz", "SPSA"}:
        params["m"] = m
    x0 = np.array([1.0, -2.0, 0.5])
    # For f(x)=||x||^2/2, the central directional derivative equals x.dot(u).
    rng = np.random.default_rng(123)
    if name == "SPSA":
        params = {"alpha": 0.1, "beta": 0.02}
        directions = [rng.choice((-1.0, 1.0), size=d) / np.sqrt(d)]
        weight = 1
    elif name == "KieferWolfowitz":
        directions = list(np.eye(d))
        weight = 1.0
    else:
        directions = [rng.normal(size=d) for _ in range(m)]
        weight = 1 / m
        if name == "SphericalZOSGD":
            directions = [u / np.sqrt(sum(u * u)) for u in directions]
            weight = d / m
    expected_gradient = sum(weight * float(x0 @ u) * u for u in directions)
    alg = initialized(name, x0, params)
    client = Client(value=lambda x: 0.5 * float(x @ x))
    update = alg.advance(client)
    assert update.queries == 2 * len(directions)
    assert update.update_complete
    check(update.estimated_gradient, expected_gradient)
    check(update.recommendation, x0 - 0.1 * expected_gradient)
    for i, u in enumerate(directions):
        sign = -1 if name == "SPSA" else 1
        check(client.points[2 * i], x0 + sign * 0.02 * u)
        check(client.points[2 * i + 1], x0 - sign * 0.02 * u)


def test_incomplete_batch_charges_calls_but_keeps_recommendation():
    alg = initialized("SphericalZOSGD", parameters={"m": 2})
    client = Client(budget=3, value=lambda x: 0.5 * float(x @ x))
    update = alg.advance(client)
    assert update.queries == 3 and not update.update_complete
    assert alg.t == 0 and client.remaining == 0
    check(update.recommendation, [1.0, -2.0])
    assert alg.advance(client).queries == 0


def test_exponentiated_gradient_exact_zero_support_and_large_gradient():
    alg = initialized("ExponentiatedGradient", (0.25, 0.75, 0), geometry="simplex")
    client = Client([np.array([10.0, 0.0, -1e6])])
    denom = 0.25 * np.exp(-1) + 0.75
    check(
        alg.advance(client).recommendation, [0.25 * np.exp(-1) / denom, 0.75 / denom, 0]
    )


def test_simplex_projection_hand_solution():
    alg = initialized("ProjectedGradient", (0.2, 0.3, 0.5), geometry="simplex")
    client = Client([np.array([4.0, -2.0, 1.0])])
    # Raw (-.2,.5,.4); threshold -0.05 on the last two coordinates.
    check(alg.advance(client).recommendation, [0, 0.55, 0.45])


def test_riemannian_sphere_tangent_and_polar():
    alg = initialized(
        "RiemannianSGD", (1.0, 0.0, 0.0), geometry="stiefel", shape=(3, 1)
    )
    client = Client([np.array([2.0, 3.0, 4.0])])
    # Tangent=(0,3,4); polar normalization of (1,-.3,-.4).
    check(
        alg.advance(client).recommendation, np.array([1.0, -0.3, -0.4]) / np.sqrt(1.25)
    )


@pytest.mark.parametrize("name", canonical_optimizer_names())
def test_initialize_resets_all_algorithm_state(name):
    geometry, shape, x0 = "euclidean", None, np.zeros(3)
    if name in {"ExponentiatedGradient", "ProjectedGradient"}:
        geometry, x0 = "simplex", np.ones(3) / 3
    elif name == "RiemannianSGD":
        geometry, shape, x0 = "stiefel", (3, 1), np.array([1.0, 0.0, 0.0])
    alg = initialized(name, x0, geometry=geometry, shape=shape)

    def client():
        return Client([np.ones(3)] * 10, value=lambda x: 0.5 * float(x @ x))

    expected = alg.advance(client()).recommendation
    alg.advance(client())
    alg.initialize(x0, np.random.default_rng(123), TaskSpec(3, geometry, shape))
    check(alg.advance(client()).recommendation, expected)
