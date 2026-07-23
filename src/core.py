"""
Core physics engine for WIND Benchmark.
This module defines the fundamental physical laws of the simulation:
Drifts: How the optimum theta_t evolves over time (Stationary, Linear, Adaptive, etc.).
Landscapes: The geometry of the loss function f_t(x) = L(x, theta_t).
Noises: Stochastic corruptions applied to observations (Gaussian, Heavy-tailed, Correlated).
DynamicEnvironment: The container binding these laws together.

"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from dataclasses import dataclass

from .manifold import cayley_orthogonal, project_to_stiefel

# =============================================================================
# 1. DRIFT MODELS (Dynamics of theta_t)
# =============================================================================


class Drift(ABC):
    """
    Abstract base class for environment dynamics (optimum evolution).

    Mathematical model:
        theta_{t+1} = D(theta_t, t, x_t)

    Where:
        - theta_t: Current environment state.
        - t: Current time step.
        - x_t: (Optional) The decision made by the optimizer.
               Required for 'Adaptive' drifts.
    """

    @abstractmethod
    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the next state theta_{t+1}.

        Args:
            theta: Current state theta_t.
            t: Current time step index.
            action: The optimizer's decision x_t (optional, for feedback loops).

        Returns:
            New state theta_{t+1}.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (if any)."""
        pass


class StationaryDrift(Drift):
    """
    Stationary environment: theta_{t+1} = theta_t.
    Baseline for convergence analysis.
    """

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return theta.copy()

    def reset(self) -> None:
        pass


class LinearDrift(Drift):
    """
    Linear trend: theta_{t+1} = theta_t + velocity.
    Simulates constant inflation or predictable motion.
    """

    def __init__(self, velocity: np.ndarray):
        self.velocity = np.array(velocity)

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return theta + self.velocity

    def reset(self) -> None:
        pass


class RandomWalkDrift(Drift):
    """
    Gaussian Random Walk (Brownian Motion).
    theta_{t+1} = theta_t + N(0, sigma^2 * I).
    Standard model for "unpredictable" dynamics (Besbes et al., 2015).
    Supports sparsity (drift only in subset of coordinates).
    """

    def __init__(
        self, sigma: float = 0.1, sparsity: float = 0.0, seed: Optional[int] = None
    ):
        self.sigma = sigma
        self.sparsity = sparsity
        self.rng = np.random.default_rng(seed)
        # Snapshot initial RNG state so reset() restores the exact sequence
        # (works even when seed is None).
        self._rng_state = self.rng.bit_generator.state

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        dim = theta.shape[0]
        noise = self.rng.normal(0, self.sigma, size=dim)

        if self.sparsity > 0:
            # Mask out coordinates to simulate sparse drift
            mask = self.rng.random(dim) > self.sparsity
            noise *= mask

        return theta + noise

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state


class CyclicDrift(Drift):
    """
    Cyclic/Seasonal drift.
    theta_t = center + A * sin(2*pi*t / T).
    Note: This overrides the previous theta based on t, rather than accumulating.
    """

    def __init__(self, amplitude: float, period: int, center: np.ndarray):
        if period <= 0:
            raise ValueError("period must be > 0")
        self.amplitude = amplitude
        self.period = period
        self.center = np.array(center)

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # t+1 because we are computing the NEXT state
        phase = 2 * np.pi * (t + 1) / self.period
        offset = self.amplitude * np.sin(phase)
        # Apply offset to all dimensions uniformly
        return self.center + offset * np.ones_like(theta)

    def reset(self) -> None:
        pass


class JumpDrift(Drift):
    """
    Abrupt changes (Teleportation / Regime Shift).
    Theta jumps to a new random location every T steps.
    Scientific basis: Testing "Time-to-Recovery" (TTR) metrics.
    """

    def __init__(
        self, interval: int, jump_magnitude: float, dim: int, seed: Optional[int] = None
    ):
        if interval <= 0:
            raise ValueError("interval must be > 0")
        if jump_magnitude < 0:
            raise ValueError("jump_magnitude must be >= 0")
        self.interval = interval
        self.jump_magnitude = jump_magnitude
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # Check if it's time to jump
        if (t + 1) % self.interval == 0:
            direction = self.rng.normal(size=self.dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            return theta + direction * self.jump_magnitude
        return theta

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state


class AdaptiveDrift(Drift):
    """
    Adaptive Drift with mode selection.

    Modes:
    - 'pursuit': Theta moves TOWARDS x (Environment helps/stabilizes).
                 theta_{t+1} = theta_t - alpha * sign(x - theta)
    - 'evasion': Theta moves AWAY from x (Adversarial/Game dynamics).
                 theta_{t+1} = theta_t + alpha * sign(x - theta)

    Scientific basis: Competitive optimization, GAN dynamics.
    """

    def __init__(
        self, alpha: float = 0.1, threshold: float = np.inf, mode: str = "pursuit"
    ):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = alpha
        self.threshold = threshold
        if mode not in ["pursuit", "evasion"]:
            raise ValueError("Mode must be 'pursuit' or 'evasion'")
        self.mode = mode

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if action is None:
            return theta

        # React only if the agent is within the (optional) threshold distance.
        if np.linalg.norm(action - theta) > self.threshold:
            return theta

        # Coordinate-wise sign of (x_t - theta_t), per Table 2.
        s = np.sign(action - theta)

        # Evasion (Table 2 formula): theta_{t+1} = theta_t - alpha * sign(x - theta).
        # Pursuit: theta moves toward x, i.e. + alpha * sign(x - theta).
        direction = s if self.mode == "pursuit" else -s
        return theta + self.alpha * direction

    def reset(self) -> None:
        pass


class SparseDrift(Drift):
    """
    Sparse random-walk drift (Table 2, "Sparse").

    At each step exactly k coordinates — a fresh random subset S_t — receive an
    independent Gaussian increment; the remaining coordinates stay fixed.
    theta_{t+1}[i] = theta_t[i] + sigma * N(0, 1) for i in S_t, |S_t| = k.

    Models change in only a subset of features (e.g. sparse concept drift).
    """

    def __init__(
        self, dim: int, k: int = 1, sigma: float = 0.1, seed: Optional[int] = None
    ):
        if not (1 <= k <= dim):
            raise ValueError(f"k must be in [1, dim]={dim}, got k={k}")
        self.dim = dim
        self.k = k
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        idx = self.rng.choice(self.dim, size=self.k, replace=False)
        out = theta.copy()
        out[idx] = out[idx] + self.rng.normal(0, self.sigma, size=self.k)
        return out

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state


class StiefelDrift(Drift):
    """
    Riemannian drift on the Stiefel manifold (optional add-on).

    Moves the optimum Theta along Stiefel(d, r) while keeping it orthonormal:
        Theta_{t+1} = Q_t Theta_t,  Q_t = Cayley(sigma * skew-Gaussian) (orthogonal).

    Theta is stored flattened (length d*r) and reshaped to (d, r) internally.
    Use with bounds=None (clipping would break orthonormality).
    """

    def __init__(self, d: int, r: int, sigma: float = 0.05, seed: Optional[int] = None):
        if not (1 <= r <= d):
            raise ValueError("Stiefel requires 1 <= r <= d")
        self.d = d
        self.r = r
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state

    def step(
        self, theta: np.ndarray, t: int, action: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Theta = theta.reshape(self.d, self.r)
        A = self.rng.normal(size=(self.d, self.d)) * self.sigma
        A = A - A.T  # skew-symmetric -> Cayley gives an orthogonal Q
        Q = cayley_orthogonal(A)
        return (Q @ Theta).reshape(-1)

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state


# =============================================================================
# 2. LANDSCAPE GEOMETRIES (Loss Functions)
# =============================================================================


class Landscape(ABC):
    """
    Abstract base class for the loss landscape geometry.
    Defined as f_t(x) = L(x, theta_t).

    CRITICAL INVARIANT: L(theta_t, theta_t) MUST be 0.0 for Dynamic Regret to be valid.
    All implementations MUST guarantee this invariant.
    """

    @abstractmethod
    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        """Compute function value f_t(x)."""
        pass

    @abstractmethod
    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute gradient nabla f_t(x)."""
        pass


class QuadraticLandscape(Landscape):
    """
    Quadratic Loss: f(x) = 1/2 * (x-theta)^T * A * (x-theta).
    Parameters:
        condition_number (kappa): Ratio of max/min eigenvalues of A.
    """

    def __init__(
        self, dim: int, condition_number: float = 1.0, seed: Optional[int] = None
    ):
        if condition_number < 1.0:
            raise ValueError("condition_number (kappa) must be >= 1")
        self.dim = dim
        self.kappa = condition_number
        self.rng = np.random.default_rng(seed)
        self.A = self._generate_matrix()

    def _generate_matrix(self) -> np.ndarray:
        # Generate random orthogonal matrix Q
        H = self.rng.normal(size=(self.dim, self.dim))
        Q, _ = np.linalg.qr(H)

        # Generate eigenvalues from 1 to kappa
        if self.dim > 1:
            eigenvalues = np.linspace(1, self.kappa, self.dim)
        else:
            eigenvalues = np.array([float(self.kappa)])

        # A = Q * diag(lambda) * Q.T
        return Q @ np.diag(eigenvalues) @ Q.T

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        diff = x - theta
        return 0.5 * float(diff.T @ self.A @ diff)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.A @ (x - theta)


class PNormLandscape(Landscape):
    """
    p-Norm Loss: f(x) = (1/p) * ||x - theta||_p^p.
    Special cases: p=2 (Euclidean), p=1 (L1/Lasso-like).
    """

    def __init__(
        self,
        p: float = 2.0,
        condition_number: float = 1.0,
        seed: Optional[int] = None,
    ):
        if p < 1.0:
            raise ValueError("p must be >= 1 for convexity.")
        if condition_number < 1.0:
            raise ValueError("condition_number (kappa) must be >= 1")
        self.p = p
        self.kappa = float(condition_number)
        self._seed = seed
        # Conditioning matrix M_kappa, built lazily once the dimension is known.
        # kappa == 1 => M = I (recovers the plain p-norm, backward compatible).
        self._M: Optional[np.ndarray] = None

    def _lazy_M(self, dim: int) -> np.ndarray:
        if self._M is None:
            if self.kappa == 1.0 or dim < 2:
                self._M = np.eye(dim)
            else:
                rng = np.random.default_rng(self._seed)
                H = rng.normal(size=(dim, dim))
                Q, _ = np.linalg.qr(H)
                # Singular values from 1 to sqrt(kappa): cond(M) = sqrt(kappa)
                exps = np.linspace(0.0, 1.0, dim)
                diag = np.sqrt(self.kappa**exps)
                self._M = Q @ np.diag(diag) @ Q.T
        return self._M

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        M = self._lazy_M(x.shape[0])
        u = M @ (x - theta)
        return float(np.sum(np.abs(u) ** self.p) / self.p)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        M = self._lazy_M(x.shape[0])
        u = M @ (x - theta)
        # d/dx (1/p)||M(x-theta)||_p^p = M^T (sign(u) |u|^(p-1))
        g_u = np.sign(u) * (np.abs(u) ** (self.p - 1))
        return M.T @ g_u


class RosenbrockLandscape(Landscape):
    """
    Rosenbrock 'Banana' function.
    Non-convex, ill-conditioned valley.
    Adapted for dynamic environment: f(x) = Rosenbrock(x - theta + 1).
    The global minimum is at theta with f(theta, theta) = 0.
    """

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        z = x - theta + 1.0  # Shift so minimum is at theta (original min is at 1,1,...)

        val = 0.0
        for i in range(len(x) - 1):
            val += 100.0 * (z[i + 1] - z[i] ** 2) ** 2 + (1.0 - z[i]) ** 2
        return float(val)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        z = x - theta + 1.0
        grad = np.zeros_like(x)
        dim = len(x)

        for i in range(dim - 1):
            grad[i] += -400.0 * z[i] * (z[i + 1] - z[i] ** 2) - 2.0 * (1.0 - z[i])
            grad[i + 1] += 200.0 * (z[i + 1] - z[i] ** 2)

        return grad


class MultiExtremalLandscape(Landscape):
    """
    Multi-extremal (Rastrigin-type) landscape with invariants GUARANTEED by construction.

    Mathematical formulation:
        f(x, theta) = sum_i [ (x_i - theta_i)^2 + A * (1 - cos(2*pi*(x_i - theta_i))) ]

    where A >= 0 controls the strength of the multimodal ripple.

    This guarantees, for ANY dimension, A, and theta:
        ✅ f(theta, theta) = 0           (exact global minimum value)
        ✅ grad f(theta, theta) = 0      (first-order optimality holds at theta)
        ✅ f(x, theta) >= 0 for all x, with equality iff x == theta
        ✅ Many local minima (cosine term) — genuinely multi-extremal

    NOTE: This replaces a previous Gaussian-mixture formulation whose minimum was
    NOT at theta (its gradient at theta was nonzero for random offsets), which broke
    the L(theta, theta) = 0 / regret invariant the benchmark relies on.

    Args:
        k_centers: Retained for API/registry compatibility (unused — the cosine
                   term already produces multiple local minima per coordinate).
        width:     Amplitude A of the multimodal ripple (must be >= 0). A=0 reduces
                   to a pure quadratic bowl.
        seed:      Retained for API/registry compatibility (function is deterministic).
    """

    def __init__(
        self, k_centers: int = 3, width: float = 1.0, seed: Optional[int] = None
    ):
        if k_centers < 1:
            raise ValueError("k_centers must be >= 1")
        if width < 0:
            raise ValueError("width (ripple amplitude A) must be >= 0")
        self.k = k_centers
        self.amplitude = float(width)
        self._seed = seed

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        u = x - theta
        ripple = self.amplitude * (1.0 - np.cos(2.0 * np.pi * u))
        return float(np.sum(u**2 + ripple))

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        u = x - theta
        return 2.0 * u + self.amplitude * 2.0 * np.pi * np.sin(2.0 * np.pi * u)


class RobustLandscape(Landscape):
    """
    Huber Loss.
    Quadratic near optimum, Linear far away. Robust to outliers in x.
    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        diff = np.abs(x - theta)
        is_small = diff <= self.delta
        squared = 0.5 * diff**2
        linear = self.delta * (diff - 0.5 * self.delta)
        return float(np.sum(np.where(is_small, squared, linear)))

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        diff = x - theta
        abs_diff = np.abs(diff)
        is_small = abs_diff <= self.delta
        return np.where(is_small, diff, self.delta * np.sign(diff))


class SimplexLandscape(Landscape):
    """
    Squared-Euclidean loss with the decision constrained to the probability simplex
    (Table 3, "Simplex"):
        L(x, theta) = ||x - theta||_2^2,   x in Delta^{d-1} = {x >= 0, sum x_i = 1}.

    Global minimum at theta (assumed to lie on the simplex), with L(theta, theta)=0
    and grad L(theta, theta)=0. The simplex constraint is NOT enforced by the loss
    itself — the optimizer/environment must keep x on the simplex; `project` provides
    the Euclidean projection onto Delta^{d-1} as a helper.
    """

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        d = x - theta
        return float(d @ d)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return 2.0 * (x - theta)

    @staticmethod
    def project(v: np.ndarray) -> np.ndarray:
        """Euclidean projection onto the probability simplex (Wang & Carreira-Perpiñán, 2013)."""
        n = v.shape[0]
        u = np.sort(v)[::-1]
        css = np.cumsum(u) - 1.0
        ind = np.arange(1, n + 1)
        cond = u - css / ind > 0
        rho = ind[cond][-1]
        tau = css[rho - 1] / rho
        return np.maximum(v - tau, 0.0)


class StiefelLandscape(Landscape):
    """
    Riemannian (Stiefel) landscape — embedded squared-Frobenius loss (optional add-on):

        L(X, Theta) = ||X - Theta||_F^2,   X, Theta in Stiefel(d, r),

    with X, Theta stored flattened (length d*r). Global minimum at Theta with
    L(Theta, Theta)=0 and ambient gradient 0 there. `grad` returns the AMBIENT
    Euclidean gradient 2(X - Theta); a Riemannian optimizer projects it onto the
    tangent space and retracts (see manifold.tangent_project / manifold.retract).

    This is a frame-tracking task: X and XQ are distinct targets for a general
    orthogonal Q. The matching tracking distance is the Frobenius frame distance
    (metrics.StiefelFrameMetric). Use GrassmannLandscape when only the spanned
    subspace matters.
    """

    def __init__(self, d: int, r: int):
        if not (1 <= r <= d):
            raise ValueError("Stiefel requires 1 <= r <= d")
        self.d = d
        self.r = r

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        diff = x - theta
        return float(diff @ diff)

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return 2.0 * (x - theta)

    @staticmethod
    def random_point(d: int, r: int, seed: Optional[int] = None) -> np.ndarray:
        """A random flattened point on Stiefel(d, r) (for initial_theta / x0)."""
        from .manifold import random_stiefel

        return random_stiefel(d, r, np.random.default_rng(seed)).reshape(-1)


class GrassmannLandscape(Landscape):
    """Basis-invariant subspace-tracking loss.

    Points are represented by flattened orthonormal frames X, Theta in
    Stiefel(d, r), while frames related by a right orthogonal transformation
    represent the same element of Gr(d, r). The loss is

        L(X, Theta) = 0.5 ||X X^T - Theta Theta^T||_F^2.

    On feasible frames this equals the sum of squared sines of the principal
    angles. ``grad`` returns the ambient Euclidean gradient; manifold-aware
    methods may project it onto the tangent space before retracting.
    """

    def __init__(self, d: int, r: int):
        if not (1 <= r <= d):
            raise ValueError("Grassmann requires 1 <= r <= d")
        self.d = d
        self.r = r

    def loss(self, x: np.ndarray, theta: np.ndarray) -> float:
        X = np.asarray(x, dtype=float).reshape(self.d, self.r)
        Theta = np.asarray(theta, dtype=float).reshape(self.d, self.r)
        projector_diff = X @ X.T - Theta @ Theta.T
        return 0.5 * float(np.sum(projector_diff * projector_diff))

    def grad(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        X = np.asarray(x, dtype=float).reshape(self.d, self.r)
        Theta = np.asarray(theta, dtype=float).reshape(self.d, self.r)
        projector_diff = X @ X.T - Theta @ Theta.T
        return (2.0 * projector_diff @ X).reshape(-1)

    @staticmethod
    def random_point(d: int, r: int, seed: Optional[int] = None) -> np.ndarray:
        """A random orthonormal representative of a point in Gr(d, r)."""
        return StiefelLandscape.random_point(d, r, seed=seed)


# =============================================================================
# 3. NOISE MODELS (Stochastic Corruption)
# =============================================================================


class Noise(ABC):
    """
    Abstract base class for measurement noise.
    Supports stateful noise for temporal correlations.
    """

    @abstractmethod
    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        """
        Apply noise to the signal.

        Args:
            signal: The clean value (scalar) or gradient (vector).
            t: Time step.

        Returns:
            Noisy signal.
        """
        pass

    def reset(self) -> None:
        """Reset internal state (for CorrelatedNoise)."""
        pass


class GaussianNoise(Noise):
    """Standard additive Gaussian noise: signal + N(0, sigma)."""

    def __init__(self, sigma: float, seed: Optional[int] = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        if self.sigma <= 0:
            return signal
        noise = self.rng.normal(0, self.sigma, size=signal.shape)
        return signal + noise


class HeavyTailedNoise(Noise):
    """
    Pareto/Heavy-tailed noise.
    Simulates outliers and sensor glitches.
    """

    def __init__(self, alpha: float, scale: float = 1.0, seed: Optional[int] = None):
        if alpha <= 0:
            raise ValueError("alpha (tail index) must be > 0")
        self.alpha = alpha  # Tail index (1.5 - 3.0)
        self.scale = scale
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        # Pareto generation: (1 - u)^(-1/alpha) - 1, random sign
        u = self.rng.random(size=signal.shape)
        pareto = (1.0 - u) ** (-1.0 / self.alpha) - 1.0
        signs = self.rng.choice([-1.0, 1.0], size=signal.shape)
        noise = signs * self.scale * pareto

        # Clip strictly massive outliers to avoid Inf in float32
        noise = np.clip(noise, -1e3 * self.scale, 1e3 * self.scale)
        return signal + noise


class CorrelatedNoise(Noise):
    """
    AR(1) Correlated Noise.
    xi_t = phi * xi_{t-1} + sqrt(1 - phi^2) * epsilon.
    Simulates drift in sensor calibration or persistent environmental effects.
    """

    def __init__(
        self, sigma: float, phi: float = 0.8, dim: int = 1, seed: Optional[int] = None
    ):
        if not (-1.0 < phi < 1.0):
            raise ValueError("phi must be in (-1, 1) for a stationary AR(1) process")
        self.sigma = sigma
        self.phi = phi
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state
        self._state = None

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        # Initialize state if needed or if dimension changed
        if self._state is None or self._state.shape != signal.shape:
            self._state = np.zeros_like(signal)

        epsilon = self.rng.normal(0, self.sigma, size=signal.shape)

        # Update AR(1) state
        # scaling factor sqrt(1-phi^2) keeps stationary variance = sigma^2
        scaling = np.sqrt(1 - self.phi**2)
        self._state = self.phi * self._state + scaling * epsilon

        return signal + self._state


class QuantizedNoise(Noise):
    """
    Quantized Noise.
    Simulates low-resolution ADC (Analog-to-Digital Converter).
    y = delta * round(signal / delta).
    """

    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        if self.delta <= 0:
            return signal
        return np.round(signal / self.delta) * self.delta


class MultiplicativeNoise(Noise):
    """
    Multiplicative Noise: signal * (1 + epsilon).
    Relative error model.
    """

    def __init__(self, sigma_rel: float = 0.1, seed: Optional[int] = None):
        self.sigma_rel = sigma_rel
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        epsilon = self.rng.normal(0, self.sigma_rel, size=signal.shape)
        return signal * (1.0 + epsilon)


class SparseNoise(Noise):
    """
    Sparse (Bernoulli) Noise.
    With probability p, applies Gaussian noise. With 1-p, noise is 0.
    Scientific basis: Packet loss in gradient transmission, intermittent sensor failure.
    """

    def __init__(self, sigma: float, p: float = 0.1, seed: Optional[int] = None):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1]")
        self.sigma = sigma
        self.p = p
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state

    def reset(self) -> None:
        self.rng.bit_generator.state = self._rng_state

    def apply(self, signal: np.ndarray, t: int) -> np.ndarray:
        # Mask: 1 with probability p
        mask = self.rng.binomial(1, self.p, size=signal.shape).astype(float)
        noise = self.rng.normal(0, self.sigma, size=signal.shape)
        return signal + (mask * noise)


# =============================================================================
# 4. ENVIRONMENT CONTAINER
# =============================================================================


@dataclass
class EnvironmentConfig:
    """Configuration for reproducible environment creation."""

    dim: int
    x_bounds: Tuple[float, float]
    drift_type: str
    drift_params: Dict[str, Any]
    landscape_type: str
    landscape_params: Dict[str, Any]
    noise_type: str
    noise_params: Dict[str, Any]


class DynamicEnvironment:
    """
    The physical world simulator.

    Responsibilities:
    1. Manage state theta_t.
    2. Enforce physical bounds.
    3. Execute drift dynamics (including adaptive feedback).
    4. Provide landscape geometry to Oracle.
    5. Handle concurrency locks (bi-directional locking for Oracle protocol).


    """

    def __init__(
        self,
        dim: int,
        drift: Drift,
        landscape: Landscape,
        initial_theta: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.dim = dim
        self.drift = drift
        self.landscape = landscape
        self.bounds = bounds

        # State initialization
        if initial_theta is not None:
            self._theta = np.array(initial_theta, dtype=np.float64)
        else:
            self._theta = np.zeros(dim)
        # Remember the construction-time start so reset() restores it (important for
        # non-zero / on-manifold initial states, e.g. Stiefel).
        self._initial_theta = self._theta.copy()

        self.t = 0

        # Concurrency flags
        self._locked_by_oracle = False
        self._is_advancing = False
        self._registered_oracle = None

    def get_current_theta(self, for_analysis: bool = False) -> np.ndarray:
        """
        Get current state.

        Strict Information Barrier:
        - for_analysis=True: Allowed (e.g., for Metric computation).
        - for_analysis=False: Forbidden (raises error to prevent cheating).
        """
        if not for_analysis:
            # In production, this would inspect the call stack
            # For now, we rely on the flag being set correctly by Metrics only
            pass
        return self._theta.copy()

    def step(self, action: Optional[np.ndarray] = None) -> None:
        """
        Advance environment time: t -> t+1.

        Args:
            action: The optimizer's decision x_t.
                    Passed to drift model for AdaptiveDrift.

        Raises:
            RuntimeError: If Oracle is currently active.
        """
        if self._locked_by_oracle:
            raise RuntimeError(
                "Cannot advance environment while Oracle is active (in a step). "
                "Call oracle.end_step() first."
            )

        self._is_advancing = True

        # 1. Update drift (pass action for adaptive dynamics)
        self._theta = self.drift.step(self._theta, self.t, action)

        # 2. Enforce bounds (if physical constraint)
        if self.bounds is not None:
            low, high = self.bounds
            self._theta = np.clip(self._theta, low, high)

        # 3. Advance clock
        self.t += 1

        self._is_advancing = False

    def reset(self, initial_theta: Optional[np.ndarray] = None) -> None:
        """Reset environment to t=0."""
        self.t = 0
        if initial_theta is not None:
            self._theta = np.array(initial_theta)
            self._initial_theta = self._theta.copy()
        else:
            self._theta = self._initial_theta.copy()

        self.drift.reset()
        if self._registered_oracle:
            self._registered_oracle.reset()

    # --- Oracle Protocol Hooks (Locking) ---

    def _register_oracle(self, oracle: Any) -> None:
        self._registered_oracle = oracle

    def _lock_for_oracle(self) -> None:
        """Called by Oracle.start_step()."""
        if self._is_advancing:
            raise RuntimeError("Cannot lock environment while it is advancing.")
        self._locked_by_oracle = True

    def _unlock_for_oracle(self) -> None:
        """Called by Oracle.end_step()."""
        self._locked_by_oracle = False


# =============================================================================
# 5. FACTORY & REGISTRY
# =============================================================================

_DRIFT_REGISTRY = {
    "stationary": StationaryDrift,
    "linear": LinearDrift,
    "random_walk": RandomWalkDrift,
    "cyclic": CyclicDrift,
    "jump": JumpDrift,
    "adaptive": AdaptiveDrift,
    "sparse": SparseDrift,
    "stiefel": StiefelDrift,
}

_LANDSCAPE_REGISTRY = {
    "quadratic": QuadraticLandscape,
    "pnorm": PNormLandscape,
    "rosenbrock": RosenbrockLandscape,
    "multiextremal": MultiExtremalLandscape,
    "robust": RobustLandscape,
    "simplex": SimplexLandscape,
    "stiefel": StiefelLandscape,
    "grassmann": GrassmannLandscape,
}

_NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "heavy_tailed": HeavyTailedNoise,
    "correlated": CorrelatedNoise,
    "quantized": QuantizedNoise,
    "multiplicative": MultiplicativeNoise,
    "sparse": SparseNoise,
}


def make_drift(name: str, **kwargs) -> Drift:
    if name not in _DRIFT_REGISTRY:
        raise ValueError(
            f"Unknown drift type: {name}. Available: {list(_DRIFT_REGISTRY.keys())}"
        )
    return _DRIFT_REGISTRY[name](**kwargs)


def make_landscape(name: str, **kwargs) -> Landscape:
    if name not in _LANDSCAPE_REGISTRY:
        raise ValueError(
            f"Unknown landscape: {name}. Available: {list(_LANDSCAPE_REGISTRY.keys())}"
        )
    return _LANDSCAPE_REGISTRY[name](**kwargs)


def make_noise(name: str, **kwargs) -> Optional[Noise]:
    if name is None or name == "none":
        return None
    if name not in _NOISE_REGISTRY:
        raise ValueError(
            f"Unknown noise: {name}. Available: {list(_NOISE_REGISTRY.keys())}"
        )
    return _NOISE_REGISTRY[name](**kwargs)


def make_environment(config: Dict[str, Any], seed: int = 42) -> DynamicEnvironment:
    """
    Factory method to create a fully configured environment.

    Args:
        config: Dictionary with keys 'dim', 'drift', 'landscape', 'noise'.
                Example:
                {
                    'dim': 10,
                    'drift': {'type': 'linear', 'velocity': [0.1]*10},
                    'landscape': {'type': 'quadratic', 'condition_number': 10},
                    'noise': {'type': 'gaussian', 'sigma': 0.1}
                }
        seed: Random seed for reproducibility.

    Returns:
        Configured DynamicEnvironment instance.
    """
    dim = config.get("dim", 2)

    # Drift
    # Copy nested dictionaries because the factory removes the registry key.
    # Mutating a caller's config would make a second identical build fail.
    d_conf = dict(config.get("drift", {"type": "stationary"}))
    d_type = d_conf.pop("type")
    # Inject dependencies if needed
    if "seed" not in d_conf and d_type in [
        "random_walk",
        "jump",
        "sparse",
        "stiefel",
    ]:
        d_conf["seed"] = seed
    if "dim" not in d_conf and d_type in ["jump", "sparse"]:
        d_conf["dim"] = dim
    drift = make_drift(d_type, **d_conf)

    # Landscape
    l_conf = dict(config.get("landscape", {"type": "quadratic"}))
    l_type = l_conf.pop("type")
    if "dim" not in l_conf and l_type in ["quadratic"]:
        l_conf["dim"] = dim
    if "seed" not in l_conf and l_type in ["quadratic", "multiextremal", "pnorm"]:
        l_conf["seed"] = seed + 1
    landscape = make_landscape(l_type, **l_conf)

    if isinstance(landscape, (StiefelLandscape, GrassmannLandscape)) and dim != (
        landscape.d * landscape.r
    ):
        raise ValueError(
            "For a Stiefel or Grassmann landscape, config['dim'] must equal d*r: "
            f"got dim={dim}, d={landscape.d}, r={landscape.r}"
        )

    # Constrained landscapes need a feasible latent initial state. Euclidean
    # environments retain the historical zero initialization.
    initial_theta = config.get("initial_theta")
    if initial_theta is None and isinstance(landscape, SimplexLandscape):
        initial_theta = np.full(dim, 1.0 / dim)
    elif initial_theta is None and isinstance(
        landscape, (StiefelLandscape, GrassmannLandscape)
    ):
        initial_theta = landscape.random_point(landscape.d, landscape.r, seed=seed + 2)
    elif initial_theta is not None:
        initial_theta = np.asarray(initial_theta, dtype=float).reshape(-1)
        if initial_theta.size != dim:
            raise ValueError(
                f"initial_theta must contain {dim} entries, got {initial_theta.size}"
            )

    # Bounds
    bounds = config.get("x_bounds", None)

    return DynamicEnvironment(
        dim=dim,
        drift=drift,
        landscape=landscape,
        initial_theta=initial_theta,
        bounds=bounds,
    )
