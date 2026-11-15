"""Source-specified algorithms; all observations pass through a query client.

No historical aliases, hidden initial queries, or cross-update value reuse.
The catalog records precise variants and primary sources.
"""

import numpy as np
from wind_benchmark.algorithms.protocol import AdvanceResult, TaskSpec
from wind_benchmark.core.geometry import (
    tangent_project,
    project_to_stiefel,
    validate_frame,
)


def _positive(name, value, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return value


def _beta(name, value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must lie in [0, 1)")
    return value


def simplex_projection(x):
    """Exact Euclidean projection onto {x >= 0, sum x = 1}."""
    x = np.asarray(x, dtype=float)
    order = np.sort(x)[::-1]
    offsets = (np.cumsum(order) - 1) / np.arange(1, x.size + 1)
    index = np.flatnonzero(order > offsets)[-1]
    return np.maximum(x - offsets[index], 0)


class _Algorithm:
    geometries = ("euclidean",)
    oracle_type = "first-order"
    query_cost_per_step = 1
    memory_complexity = "O(d)"
    update_schedule = "one-gradient-per-update"

    def __init__(self, lr=0.1):
        self.lr = _positive("lr", lr)
        self.parameters = {"lr": self.lr}
        self.name = type(self).__name__
        self._initialized = False

    def initialize(self, x0, rng, task_spec):
        self._initialized = False
        if not isinstance(task_spec, TaskSpec):
            raise TypeError("Expected public TaskSpec")
        if task_spec.geometry not in self.geometries:
            raise ValueError(f"{self.name} requires geometry in {self.geometries}")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        x = np.asarray(x0, dtype=float)
        if x.shape != (task_spec.dimension,) or not np.isfinite(x).all():
            raise ValueError("x0 must be a finite vector matching TaskSpec")
        if task_spec.geometry == "simplex":
            if np.any(x < 0) or not np.isclose(x.sum(), 1, atol=1e-12, rtol=0):
                raise ValueError("x0 must be on the probability simplex")
        if task_spec.geometry in {"stiefel", "grassmann"}:
            validate_frame(x.reshape(task_spec.matrix_shape))
        self.x = x.copy()
        self.recommendation = x.copy()
        self.rng = rng
        self.task_spec = task_spec
        self.t = 0
        self._reset_state()
        self._initialized = True

    def _reset_state(self):
        pass

    def _point(self):
        return self.x.copy()

    def _update(self, point, gradient):
        return point - self.lr * gradient

    def _recommend(self):
        return self.x.copy()

    def advance(self, client):
        if not self._initialized:
            raise RuntimeError("Call initialize before advance")
        if client.remaining <= 0:
            return AdvanceResult(self.recommendation, False, 0)
        point = self._point()
        obs = client.query(point.copy())
        if obs.grad is None:
            raise ValueError(f"{self.name} requires gradient observations")
        gradient = np.asarray(obs.grad, dtype=float)
        if gradient.shape != point.shape or not np.isfinite(gradient).all():
            raise FloatingPointError("Invalid gradient observation")
        self.t += 1
        candidate = self._update(point, gradient)
        if not np.isfinite(candidate).all():
            raise FloatingPointError("Nonfinite algorithm iterate")
        self.x = candidate.copy()
        self.recommendation = self._recommend()
        return AdvanceResult(self.recommendation, True, 1, gradient, point)


class SGD(_Algorithm):
    """x[t+1] = x[t] - lr * g(x[t]); no momentum."""


class PolyakRuppertSGD(_Algorithm):
    """Raw SGD with lr*t**(-0.75), recommendation mean(x[1],...,x[t])."""

    def _reset_state(self):
        self.average = np.zeros_like(self.x)

    def _update(self, point, gradient):
        return point - self.lr * self.t ** (-0.75) * gradient

    def _recommend(self):
        self.average += (self.x - self.average) / self.t
        return self.average.copy()


class HeavyBall(_Algorithm):
    def __init__(self, lr=0.1, beta=0.9):
        super().__init__(lr)
        self.beta = _beta("beta", beta)
        self.parameters.update(beta=self.beta)

    def _reset_state(self):
        self.velocity = np.zeros_like(self.x)

    def _update(self, point, gradient):
        self.velocity = self.beta * self.velocity - self.lr * gradient
        return point + self.velocity


class Nesterov(HeavyBall):
    """y=x+beta*(x-x_previous); x_next=y-lr*g(y)."""

    update_schedule = "one-gradient-at-lookahead-per-update"

    def __init__(self, lr=0.05, beta=0.9):
        super().__init__(lr, beta)

    def _reset_state(self):
        self.previous = self.x.copy()

    def _point(self):
        return self.x + self.beta * (self.x - self.previous)

    def _update(self, point, gradient):
        self.previous = self.x.copy()
        return point - self.lr * gradient


class Adam(_Algorithm):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1, self.beta2 = _beta("beta1", beta1), _beta("beta2", beta2)
        self.eps = _positive("eps", eps)
        self.parameters.update(beta1=self.beta1, beta2=self.beta2, eps=self.eps)

    def _reset_state(self):
        self.m = np.zeros_like(self.x)
        self.v = np.zeros_like(self.x)

    def _moments(self, gradient):
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

    def _update(self, point, gradient):
        self._moments(gradient)
        mhat = self.m / (1 - self.beta1**self.t)
        vhat = self.v / (1 - self.beta2**self.t)
        return point - self.lr * mhat / (np.sqrt(vhat) + self.eps)


class AdamW(Adam):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, decay=0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.decay = _positive("decay", decay, zero=True)
        self.parameters.update(decay=self.decay)

    def _update(self, point, gradient):
        return super()._update(point, gradient) - self.lr * self.decay * point


class AMSGrad(Adam):
    """Raw moments and raw second-moment maximum; constant-lr variant."""

    def _reset_state(self):
        super()._reset_state()
        self.vmax = np.zeros_like(self.x)

    def _update(self, point, gradient):
        self._moments(gradient)
        self.vmax = np.maximum(self.vmax, self.v)
        return point - self.lr * self.m / (np.sqrt(self.vmax) + self.eps)


class SignSGD(_Algorithm):
    def __init__(self, lr=0.05):
        super().__init__(lr)

    def _update(self, point, gradient):
        return point - self.lr * np.sign(gradient)


class AdaGrad(_Algorithm):
    """Diagonal accumulated-square preconditioner; epsilon outside sqrt."""

    def __init__(self, lr=0.1, eps=1e-8):
        super().__init__(lr)
        self.eps = _positive("eps", eps)
        self.parameters.update(eps=self.eps)

    def _reset_state(self):
        self.squares = np.zeros_like(self.x)

    def _update(self, point, gradient):
        self.squares += gradient**2
        return point - self.lr * gradient / (np.sqrt(self.squares) + self.eps)


class ProxSGD(_Algorithm):
    def __init__(self, lr=0.1, lam=0.01):
        super().__init__(lr)
        self.lam = _positive("lam", lam, zero=True)
        self.parameters.update(lam=self.lam)

    def _update(self, point, gradient):
        proposal = point - self.lr * gradient
        return np.sign(proposal) * np.maximum(np.abs(proposal) - self.lr * self.lam, 0)


class RDA(ProxSGD):
    """Quadratic prox h=||x||^2/2; beta[t]=sqrt(t)/lr, L1 regularizer."""

    def _reset_state(self):
        if np.any(self.x != 0):
            raise ValueError("RDA requires x0=0 (minimizer of quadratic prox and L1)")
        self.gradient_sum = np.zeros_like(self.x)

    def _update(self, point, gradient):
        self.gradient_sum += gradient
        beta_t = np.sqrt(self.t) / self.lr
        proposal = -self.gradient_sum / beta_t
        return np.sign(proposal) * np.maximum(
            np.abs(proposal) - self.t * self.lam / beta_t, 0
        )


class ExponentiatedGradient(_Algorithm):
    geometries = ("simplex",)

    def _reset_state(self):
        self.log_weights = np.full_like(self.x, -np.inf)
        positive = self.x > 0
        self.log_weights[positive] = np.log(self.x[positive])

    def _update(self, point, gradient):
        self.log_weights -= self.lr * gradient
        self.log_weights -= np.logaddexp.reduce(self.log_weights)
        return np.exp(self.log_weights)


class ProjectedGradient(_Algorithm):
    geometries = ("simplex",)

    def _update(self, point, gradient):
        return simplex_projection(point - self.lr * gradient)


class RiemannianSGD(_Algorithm):
    geometries = ("stiefel", "grassmann")

    def _update(self, point, gradient):
        shape = self.task_spec.matrix_shape
        frame, ambient = point.reshape(shape), gradient.reshape(shape)
        return project_to_stiefel(
            frame - self.lr * tangent_project(frame, ambient)
        ).ravel()


class _PairedZO(_Algorithm):
    oracle_type = "zero-order"
    query_cost_per_step = "2*m"
    update_schedule = "fresh-symmetric-pairs-per-update"

    def __init__(self, lr=0.005, h=0.01, m=1):
        super().__init__(lr)
        self.h = _positive("h", h)
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("m must be a positive integer")
        self.m = int(m)
        self.parameters.update(h=self.h, m=self.m)

    def _directions(self):
        raise NotImplementedError

    def advance(self, client):
        if not self._initialized:
            raise RuntimeError("Call initialize before advance")
        if client.remaining <= 0:
            return AdvanceResult(self.recommendation, False, 0)
        gradient = np.zeros_like(self.x)
        queries = 0
        for direction, weight in self._directions():
            values = []
            for sign in (1, -1):
                if client.remaining <= 0:
                    return AdvanceResult(self.recommendation, False, queries)
                obs = client.query(self.x + sign * self.h * direction)
                queries += 1
                if obs.value is None or not np.isfinite(obs.value):
                    raise FloatingPointError("A finite value observation is required")
                values.append(float(obs.value))
            gradient += weight * (values[0] - values[1]) / (2 * self.h) * direction
        candidate = self.x - self.lr * gradient
        if not np.isfinite(candidate).all():
            raise FloatingPointError("Nonfinite algorithm iterate")
        point = self.x.copy()
        self.x = candidate
        self.recommendation = self.x.copy()
        self.t += 1
        return AdvanceResult(self.recommendation, True, queries, gradient, point)


class _ValueAlgorithm(_Algorithm):
    oracle_type = "zero-order"

    def _ready(self, client):
        if not self._initialized:
            raise RuntimeError("Call initialize before advance")
        return client.remaining > 0

    @staticmethod
    def _value(client, point):
        value = client.query(point.copy()).value
        if value is None or not np.isfinite(value):
            raise FloatingPointError("A finite value observation is required")
        return float(value)

    def _commit(self, candidate, queries, gradient=None, point=None):
        if not np.isfinite(candidate).all():
            raise FloatingPointError("Nonfinite algorithm iterate")
        self.x = candidate.copy()
        self.recommendation = self.x.copy()
        self.t += 1
        return AdvanceResult(self.recommendation, True, queries, gradient, point)


class SPSA(_ValueAlgorithm):
    """Normalized Bernoulli tracking variant; E[g]=grad/d on a quadratic."""

    query_cost_per_step = 2
    update_schedule = "fresh-minus-plus-pair"

    def __init__(self, alpha=0.005, beta=0.1):
        self.alpha = _positive("alpha", alpha)
        self.beta = _positive("beta", beta)
        self.parameters = dict(alpha=self.alpha, beta=self.beta)
        self.name = type(self).__name__
        self._initialized = False

    def _center(self):
        return self.x.copy()

    def _next(self, point, gradient):
        return point - self.alpha * gradient

    def advance(self, client):
        if not self._ready(client):
            return AdvanceResult(self.recommendation, False, 0)
        point = self._center()
        direction = self.rng.choice((-1.0, 1.0), size=self.x.size) / np.sqrt(
            self.x.size
        )
        minus = self._value(client, point - self.beta * direction)
        if client.remaining <= 0:
            return AdvanceResult(self.recommendation, False, 1)
        plus = self._value(client, point + self.beta * direction)
        gradient = direction * ((plus - minus) / (2 * self.beta))
        candidate = self._next(point, gradient)
        return self._commit(candidate, 2, gradient, point)


class AcceleratedSPSA(SPSA):
    """Constant-coefficient adaptation of Sergeenko/Granichin (2022), Eq.8."""

    def __init__(self, h=0.005, beta=0.1, a=0.1, delta=1.05):
        super().__init__(alpha=h, beta=beta)
        self.h = self.alpha
        self.a = _beta("a", a)
        if self.a == 0:
            raise ValueError("a must lie in (0, 1)")
        self.delta = _positive("delta", delta)
        self.parameters = dict(h=self.h, beta=self.beta, a=self.a, delta=self.delta)

    def _reset_state(self):
        self.z = self.x.copy()
        self.gamma = self.delta

    def _center(self):
        return (self.x + self.a * self.z) / (1 + self.a)

    def _next(self, point, gradient):
        candidate = point - self.h * gradient
        z = (1 - self.a) * self.z + self.a * point - (self.a / self.delta) * gradient
        if not np.isfinite(z).all() or not np.isfinite(candidate).all():
            raise FloatingPointError("Nonfinite accelerated state")
        self.z = z
        return candidate


class OnePointSPSA(_ValueAlgorithm):
    query_cost_per_step = 1
    update_schedule = "fresh-one-point-per-update"

    def __init__(self, lr=0.005, r=0.1):
        super().__init__(lr)
        self.r = _positive("r", r)
        self.parameters.update(r=self.r)

    def advance(self, client):
        if not self._ready(client):
            return AdvanceResult(self.recommendation, False, 0)
        point = self.x.copy()
        direction = self.rng.choice((-1.0, 1.0), size=self.x.size)
        value = self._value(client, point + self.r * direction)
        gradient = value / (self.r * direction)
        return self._commit(point - self.lr * gradient, 1, gradient, point)


class ZOSignSGD(_ValueAlgorithm):
    query_cost_per_step = "m+1"
    update_schedule = "fresh-shared-baseline-and-m-sphere-directions"

    def __init__(self, lr=0.005, r=0.01, m=1):
        super().__init__(lr)
        self.r = _positive("r", r)
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError("m must be a positive integer")
        self.m = int(m)
        self.parameters.update(r=self.r, m=self.m)

    def advance(self, client):
        if not self._ready(client):
            return AdvanceResult(self.recommendation, False, 0)
        point = self.x.copy()
        base = self._value(client, point)
        gradient = np.zeros_like(point)
        queries = 1
        for _ in range(self.m):
            if client.remaining <= 0:
                return AdvanceResult(self.recommendation, False, queries)
            direction = self.rng.normal(size=point.size)
            while np.linalg.norm(direction) == 0:
                direction = self.rng.normal(size=point.size)
            direction /= np.linalg.norm(direction)
            value = self._value(client, point + self.r * direction)
            queries += 1
            gradient += (value - base) * direction
        gradient *= point.size / (self.m * self.r)
        return self._commit(
            point - self.lr * np.sign(gradient), queries, gradient, point
        )


class RandomSearch(_ValueAlgorithm):
    query_cost_per_step = 2
    update_schedule = "fresh-random-order-base-candidate-pair"

    def __init__(self, sigma=0.1):
        self.sigma = _positive("sigma", sigma)
        self.parameters = dict(sigma=self.sigma)
        self.name = type(self).__name__
        self._initialized = False

    def _reset_state(self):
        # Fixed, independent child streams. No task/environment seed is exposed.
        seeds = self.rng.integers(0, 2**63, size=2, dtype=np.int64)
        self.proposal_rng = np.random.default_rng(int(seeds[0]))
        self.order_rng = np.random.default_rng(int(seeds[1]))

    def advance(self, client):
        if not self._ready(client):
            return AdvanceResult(self.recommendation, False, 0)
        candidate = self.x + self.sigma * self.proposal_rng.normal(
            size=self.x.size
        ) / np.sqrt(self.x.size)
        points = (self.x, candidate)
        order = (0, 1) if self.order_rng.integers(2) == 0 else (1, 0)
        values = {}
        for index in order:
            if client.remaining <= 0:
                return AdvanceResult(self.recommendation, False, len(values))
            values[index] = self._value(client, points[index])
        return self._commit(candidate if values[1] < values[0] else self.x, 2)


class GaussianZOSGD(_PairedZO):
    def _directions(self):
        for _ in range(self.m):
            yield self.rng.normal(size=self.x.size), 1 / self.m


class SphericalZOSGD(_PairedZO):
    def _directions(self):
        for _ in range(self.m):
            direction = self.rng.normal(size=self.x.size)
            norm = np.linalg.norm(direction)
            while norm == 0:
                direction = self.rng.normal(size=self.x.size)
                norm = np.linalg.norm(direction)
            yield direction / norm, self.x.size / self.m


class KieferWolfowitz(_PairedZO):
    query_cost_per_step = "2*d"

    def __init__(self, lr=0.02, h=1e-4):
        super().__init__(lr, h, m=1)
        del self.parameters["m"]

    def _directions(self):
        for index in range(self.x.size):
            direction = np.zeros_like(self.x)
            direction[index] = 1
            yield direction, 1.0
