"""
WIND Benchmark Experiment: Theoretically Grounded Stabilization Analysis
Validates Definitions 1-2 for 25 optimization algorithms across (ρ, A) regimes.

Scientific Guarantees:
  ✓ Lyapunov function V_n(x) = ‖x - θ_n‖_{ρ+1}^{ρ+1} for stabilization analysis
  ✓ Optimal ℓ_{ρ+1} geometry for ρ-Hölder smooth gradients
  ✓ Drift normalization V_n / A^{ρ+1} for fair comparison across speeds
  ✓ Ground truth θ_t inclusion for exact regret computation
  ✓ Single-file export with full optimizer signature + environment config
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import warnings

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .core import (
    DynamicEnvironment,
    LinearDrift,
    QuadraticLandscape,
    PNormLandscape,
    RobustLandscape,
    make_environment,
    GaussianNoise,
    HeavyTailedNoise,
)
from .oracle import FirstOrderOracle, ZeroOrderOracle, Observation
from .metrics import (
    MetricsCollection,
    TrackingErrorMetric,
    LyapunovMetric,
    NormalizedLyapunovMetric,
    AsymptoticBoundMetric,
    DynamicRegretMetric,
    DriftAdaptationMetric,
)
from .benchmark import BenchmarkRunner, OptimizerProtocol, ExperimentResult

# ============================================================================
# 1. OPTIMIZERS (25 algorithms with theoretically motivated parameters)
# ============================================================================

# ============================================================================
# FIRST-ORDER METHODS (12 algorithms)
# ============================================================================


class SGD(OptimizerProtocol):
    """Stochastic Gradient Descent with momentum support."""

    def __init__(self, lr: float = 0.1, momentum: float = 0.0, name: str = "SGD"):
        self.lr = lr
        self.momentum = momentum
        self.name = name
        self.velocity = None

    def reset(self):
        self.velocity = None

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("SGD requires gradient access")
        x = obs.x.copy()
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = np.zeros_like(x)
            self.velocity = self.momentum * self.velocity - self.lr * obs.grad
            return x + self.velocity
        return x - self.lr * obs.grad


class SGDPolyak(OptimizerProtocol):
    """SGD with Polyak averaging for reduced variance."""

    def __init__(self, lr: float = 0.1, name: str = "SGD_Polyak"):
        self.lr = lr
        self.name = name
        self.x_avg = None
        self.step_count = 0

    def reset(self):
        self.x_avg = None
        self.step_count = 0

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("SGD_Polyak requires gradient access")
        x = obs.x.copy() - self.lr * obs.grad
        self.step_count += 1
        if self.x_avg is None:
            self.x_avg = x.copy()
        else:
            self.x_avg = (1 - 1 / self.step_count) * self.x_avg + (
                1 / self.step_count
            ) * x
        return self.x_avg.copy()


class HeavyBall(OptimizerProtocol):
    """Heavy Ball momentum method."""

    def __init__(self, lr: float = 0.1, beta: float = 0.9, name: str = "HeavyBall"):
        self.lr = lr
        self.beta = beta
        self.name = name
        self.velocity = None

    def reset(self):
        self.velocity = None

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("HeavyBall requires gradient access")
        x = obs.x.copy()
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        self.velocity = self.beta * self.velocity - self.lr * obs.grad
        return x + self.velocity


class Nesterov(OptimizerProtocol):
    """Nesterov Accelerated Gradient."""

    def __init__(self, lr: float = 0.05, beta: float = 0.9, name: str = "Nesterov"):
        self.lr = lr
        self.beta = beta
        self.name = name
        self.velocity = None

    def reset(self):
        self.velocity = None

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("Nesterov requires gradient access")
        x = obs.x.copy()
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        # Lookahead position
        x_lookahead = x + self.beta * self.velocity
        # Gradient at lookahead (approximated using current gradient)
        grad_lookahead = obs.grad
        # Update velocity
        self.velocity = self.beta * self.velocity - self.lr * grad_lookahead
        return x + self.velocity


class Adam(OptimizerProtocol):
    """Adam optimizer with bias correction."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        name: str = "Adam",
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.name = name
        self.m = None
        self.v = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("Adam requires gradient access")
        x = obs.x.copy()
        grad = obs.grad
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(OptimizerProtocol):
    """Adam with weight decay."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        name: str = "AdamW",
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.name = name
        self.m = None
        self.v = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("AdamW requires gradient access")
        x = obs.x.copy()
        grad = obs.grad + self.weight_decay * x
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AMSGrad(OptimizerProtocol):
    """AMSGrad with monotonic decay."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        name: str = "AMSGrad",
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.name = name
        self.m = None
        self.v = None
        self.v_hat = None
        self.t = 0

    def reset(self):
        self.m = None
        self.v = None
        self.v_hat = None
        self.t = 0

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("AMSGrad requires gradient access")
        x = obs.x.copy()
        grad = obs.grad
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
            self.v_hat = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        self.v_hat = np.maximum(self.v_hat, self.v)
        m_hat = self.m / (1 - self.beta1**self.t)
        return x - self.lr * m_hat / (np.sqrt(self.v_hat) + self.eps)


class SMD(OptimizerProtocol):
    """Stochastic Mirror Descent with entropy mirror map."""

    def __init__(self, lr: float = 0.1, name: str = "SMD"):
        self.lr = lr
        self.name = name

    def reset(self):
        pass

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("SMD requires gradient access")
        x = obs.x.copy()
        # Exponentiated gradient update (entropy mirror map)
        exp_grad = np.exp(-self.lr * obs.grad)
        x_new = x * exp_grad
        x_new = np.maximum(x_new, 1e-10)  # Avoid zeros
        x_new = x_new / np.sum(x_new)  # Project to simplex
        return x_new


class RDA(OptimizerProtocol):
    """Regularized Dual Averaging."""

    def __init__(self, lr: float = 0.1, lambda_reg: float = 0.01, name: str = "RDA"):
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.name = name
        self.grad_sum = None
        self.t = 0

    def reset(self):
        self.grad_sum = None
        self.t = 0

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("RDA requires gradient access")
        x = obs.x.copy()
        if self.grad_sum is None:
            self.grad_sum = np.zeros_like(obs.grad)
        self.t += 1
        self.grad_sum += obs.grad
        # Proximal operator for L1 regularization (soft thresholding)
        threshold = self.lr * self.lambda_reg * self.t
        x_new = -self.lr * self.grad_sum
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - threshold, 0.0)
        return x_new


class ProxSGD(OptimizerProtocol):
    """Proximal SGD with L1 regularization."""

    def __init__(
        self, lr: float = 0.1, lambda_reg: float = 0.01, name: str = "ProxSGD"
    ):
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.name = name

    def reset(self):
        pass

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("ProxSGD requires gradient access")
        x = obs.x.copy()
        # Gradient step + proximal operator
        x_grad = x - self.lr * obs.grad
        # Soft thresholding for L1 proximal operator
        threshold = self.lr * self.lambda_reg
        x_new = np.sign(x_grad) * np.maximum(np.abs(x_grad) - threshold, 0.0)
        return x_new


class AdaptiveLR(OptimizerProtocol):
    """Adaptive learning rate based on gradient norm."""

    def __init__(self, lr0: float = 0.1, name: str = "AdaptiveLR"):
        self.lr0 = lr0
        self.name = name

    def reset(self):
        pass

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("AdaptiveLR requires gradient access")
        x = obs.x.copy()
        grad_norm = np.linalg.norm(obs.grad) + 1e-8
        lr = self.lr0 / (1.0 + grad_norm)
        return x - lr * obs.grad


class SignSGD(OptimizerProtocol):
    """1-bit gradient compression for heavy-tailed robustness."""

    def __init__(self, lr: float = 0.05, name: str = "SignSGD"):
        self.lr = lr
        self.name = name

    def reset(self):
        pass

    def step(self, obs: Observation) -> np.ndarray:
        if obs.grad is None:
            raise ValueError("SignSGD requires gradient access")
        return obs.x.copy() - self.lr * np.sign(obs.grad)


# ============================================================================
# ZERO-ORDER METHODS (13 algorithms) — STABILIZED IMPLEMENTATIONS
# ============================================================================


class RandomSearch(OptimizerProtocol):
    """Random search with memory of best point."""

    def __init__(self, lr: float = 0.1, scale: float = 0.5, name: str = "RandomSearch"):
        self.lr = lr
        self.scale = scale
        self.name = name
        self.best_x = None
        self.best_val = float("inf")

    def reset(self):
        self.best_x = None
        self.best_val = float("inf")

    def step(self, obs: Observation) -> np.ndarray:
        if self.best_x is None:
            self.best_x = obs.x.copy()
            self.best_val = obs.value
            return self.best_x + np.random.randn(*self.best_x.shape) * self.scale

        if obs.value < self.best_val:
            self.best_x = obs.x.copy()
            self.best_val = obs.value

        return self.best_x + np.random.randn(*self.best_x.shape) * self.scale


class OnePointSPSA(OptimizerProtocol):
    """
    One-point SPSA with single measurement per step.

    Args:
        lr: Learning rate (0.005 is recommended for stability)
        perturb: Perturbation magnitude for gradient estimation
        name: Algorithm identifier
    """

    def __init__(
        self, lr: float = 0.005, perturb: float = 0.1, name: str = "OnePointSPSA"
    ):
        self.lr = lr
        self.perturb = perturb  # Explicit perturbation scale
        self.name = name
        self.prev_x = None
        self.prev_val = None
        self.delta = None

    def reset(self):
        self.prev_x = None
        self.prev_val = None
        self.delta = None

    def step(self, obs: Observation) -> np.ndarray:
        if self.prev_x is None:
            self.prev_x = obs.x.copy()
            self.prev_val = obs.value if obs.value is not None else 0.0
            # Generate a random perturbation direction.
            self.delta = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(float)
            # Return the perturbed point for the next query.
            return self.prev_x + self.perturb * self.delta

        # Estimate the gradient from the difference in values.
        grad_est = (obs.value - self.prev_val) / (self.perturb + 1e-12) * self.delta

        # Guard against numerical overflow.
        grad_est = np.clip(grad_est, -10.0, 10.0)

        # Update the current position.
        x_new = self.prev_x - self.lr * grad_est

        # Retain the current state for the next iteration.
        self.prev_x = obs.x.copy()
        self.prev_val = obs.value if obs.value is not None else 0.0
        self.delta = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(float)

        # Return the perturbed point for the next query.
        return x_new + self.perturb * self.delta


class FiniteDiffCentral(OptimizerProtocol):
    """Central finite differences (2d queries per step)."""

    def __init__(
        self, lr: float = 0.02, h: float = 1e-4, name: str = "FiniteDiffCentral"
    ):
        self.lr = lr
        self.h = h
        self.name = name
        self.x_base = None
        self.dim = None
        self.query_buffer = {}
        self.query_idx = 0

    def reset(self):
        self.x_base = None
        self.dim = None
        self.query_buffer = {}
        self.query_idx = 0

    def step(self, obs: Observation) -> np.ndarray:
        if self.x_base is None:
            self.x_base = obs.x.copy()
            self.dim = self.x_base.shape[0]
            self.query_idx = 0
            self.query_buffer = {}
            # Query first perturbed point (+h along dim 0)
            e1 = np.zeros(self.dim)
            e1[0] = self.h
            return self.x_base + e1

        # Store current query result
        self.query_buffer[self.query_idx] = (obs.x.copy(), obs.value)

        # If we have all 2d queries, compute gradient and update
        if len(self.query_buffer) == 2 * self.dim:
            grad = np.zeros(self.dim)
            for i in range(self.dim):
                x_plus, f_plus = self.query_buffer[2 * i]
                x_minus, f_minus = self.query_buffer[2 * i + 1]
                grad[i] = (f_plus - f_minus) / (2 * self.h)

            # Update position
            x_new = self.x_base - self.lr * grad

            # Reset for next gradient estimate
            self.x_base = x_new.copy()
            self.query_buffer = {}
            self.query_idx = 0

            # Return first perturbed point for next gradient estimate
            e1 = np.zeros(self.dim)
            e1[0] = self.h
            return x_new + e1

        # Otherwise, return next perturbed point
        self.query_idx += 1
        e = np.zeros(self.dim)
        dim_idx = self.query_idx // 2
        sign = 1 if self.query_idx % 2 == 0 else -1
        e[dim_idx] = sign * self.h
        return self.x_base + e


class FDSA(OptimizerProtocol):
    """Finite Difference Stochastic Approximation (Spall, 1992)."""

    def __init__(self, lr: float = 0.02, h: float = 1e-4, name: str = "FDSA"):
        self.lr = lr
        self.h = h
        self.name = name
        self.x_prev = None
        self.x_prev_val = None  # Previous evaluation point

    def reset(self):
        self.x_prev = None
        self.x_prev_val = None

    def step(self, obs: Observation) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.x_prev_val = obs.value
            # Return perturbed point
            delta = np.random.randn(*obs.x.shape) * self.h
            return self.x_prev + delta

        # Estimate gradient from two-point difference
        delta = obs.x - self.x_prev
        delta_norm = np.linalg.norm(delta) + 1e-8
        grad_est = (obs.value - self.x_prev_val) / delta_norm * (delta / delta_norm)

        # Update position
        x_new = self.x_prev - self.lr * grad_est

        # Store current state
        self.x_prev = obs.x.copy()
        self.x_prev_val = obs.value

        # Return perturbed point
        delta_next = np.random.randn(*obs.x.shape) * self.h
        return x_new + delta_next


class SPSA(OptimizerProtocol):
    """SPSA with lr=0.005 and numerical stability."""

    def __init__(self, lr=0.005, perturb=0.1, name="SPSA"):
        self.lr = lr
        self.perturb = perturb
        self.name = name
        self.x_prev = None
        self.x_prev_val = None
        self.delta = None

    def reset(self):
        self.x_prev = None
        self.x_prev_val = None
        self.delta = None

    def step(self, obs):
        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.x_prev_val = obs.value if obs.value is not None else 0.0
            self.delta = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(float)
            return self.x_prev + self.perturb * self.delta

        grad_est = (
            (obs.value - self.x_prev_val) / (2 * self.perturb + 1e-12) * self.delta
        )
        grad_est = np.clip(grad_est, -10.0, 10.0)  # Overflow guard

        x_new = self.x_prev - self.lr * grad_est

        self.x_prev = obs.x.copy()
        self.x_prev_val = obs.value if obs.value is not None else 0.0
        self.delta = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(float)

        return x_new + self.perturb * self.delta


class ZOSGD(OptimizerProtocol):
    """Zero-Order SGD with Gaussian smoothing and overflow protection."""

    def __init__(self, lr: float = 0.005, mu: float = 0.01, name: str = "ZOSGD"):
        self.lr = lr
        self.mu = max(mu, 1e-6)  # Avoid division by zero
        self.name = name
        self.x_prev = None
        self.x_prev_val = None

    def reset(self):
        self.x_prev = None
        self.x_prev_val = None

    def step(self, obs: Observation) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.x_prev_val = obs.value
            # Return Gaussian perturbed point
            delta = np.random.randn(*obs.x.shape) * self.mu
            return self.x_prev + delta

        # Estimate gradient via Gaussian smoothing with protection
        delta = obs.x - self.x_prev
        delta_norm_sq = np.sum(delta**2) + 1e-12  # Avoid division by zero
        grad_est = (obs.value - self.x_prev_val) / delta_norm_sq * delta

        # Update position with gradient clipping
        grad_est = np.clip(grad_est, -1e3, 1e3)  # Overflow guard
        x_new = self.x_prev - self.lr * grad_est

        # Store current state
        self.x_prev = obs.x.copy()
        self.x_prev_val = obs.value

        # Return perturbed point
        delta_next = np.random.randn(*obs.x.shape) * self.mu
        return x_new + delta_next


class ZOSignSGD(OptimizerProtocol):
    """Zero-Order SignSGD with gradient sign estimation."""

    def __init__(self, lr: float = 0.005, mu: float = 0.01, name: str = "ZOSignSGD"):
        self.lr = lr
        self.mu = mu
        self.name = name
        self.x_prev = None
        self.x_prev_val = None

    def reset(self):
        self.x_prev = None
        self.x_prev_val = None

    def step(self, obs: Observation) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.x_prev_val = obs.value
            # Return Gaussian perturbed point
            delta = np.random.randn(*obs.x.shape) * self.mu
            return self.x_prev + delta

        # Estimate gradient sign via finite differences
        delta = obs.x - self.x_prev
        delta_norm = np.linalg.norm(delta) + 1e-8
        grad_sign = np.sign((obs.value - self.x_prev_val) * delta / delta_norm)

        # Update position with sign-based step
        x_new = self.x_prev - self.lr * grad_sign

        # Store current state
        self.x_prev = obs.x.copy()
        self.x_prev_val = obs.value

        # Return perturbed point
        delta_next = np.random.randn(*obs.x.shape) * self.mu
        return x_new + delta_next


class QuadraticInterpolation(OptimizerProtocol):
    """Quadratic interpolation along random direction."""

    def __init__(self, lr: float = 0.1, name: str = "QuadraticInterpolation"):
        self.lr = lr
        self.name = name
        self.x_prev = None
        self.state = "base"  # 'base', 'plus', 'minus'
        self.f_base = None
        self.f_plus = None
        self.f_minus = None
        self.direction = None

    def reset(self):
        self.x_prev = None
        self.state = "base"
        self.f_base = None
        self.f_plus = None
        self.f_minus = None
        self.direction = None

    def step(self, obs: Observation) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.f_base = obs.value
            self.state = "plus"
            # Query point in random direction
            self.direction = np.random.randn(*obs.x.shape)
            self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
            return self.x_prev + self.lr * self.direction

        if self.state == "plus":
            self.f_plus = obs.value
            self.state = "minus"
            return self.x_prev - self.lr * self.direction

        # state == 'minus'
        self.f_minus = obs.value
        self.state = "base"

        # Quadratic interpolation to find minimum along direction
        f0, f1, f_1 = self.f_base, self.f_plus, self.f_minus
        a = (f1 + f_1 - 2 * f0) / 2
        b = (f1 - f_1) / 2

        if abs(a) > 1e-8:
            t_opt = -b / (2 * a)
        else:
            t_opt = 0.0

        # Update position with clipping to prevent explosion
        t_opt = np.clip(t_opt, -2.0, 2.0)  # Guard against excessively large steps
        x_new = self.x_prev + t_opt * self.direction

        # Store for next iteration
        self.x_prev = x_new.copy()
        self.f_base = obs.value  # Approximate

        # Next random direction
        self.direction = np.random.randn(*obs.x.shape)
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        self.state = "plus"
        return x_new + self.lr * self.direction


class KieferWolfowitz(OptimizerProtocol):
    """Classical Kiefer-Wolfowitz stochastic approximation."""

    def __init__(
        self, lr: float = 0.005, cn: float = 0.1, name: str = "KieferWolfowitz"
    ):
        self.lr = lr
        self.cn = cn
        self.name = name
        self.x_prev = None
        self.x_prev_val = None
        self.n = 0
        self.perturbations = None

    def reset(self):
        self.x_prev = None
        self.x_prev_val = None
        self.n = 0
        self.perturbations = None

    def step(self, obs: Observation) -> np.ndarray:
        self.n += 1
        cn = self.cn / np.sqrt(self.n)

        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.x_prev_val = obs.value
            # Generate perturbation for each coordinate
            self.perturbations = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(
                float
            )
            # Query perturbed point
            return self.x_prev + cn * self.perturbations

        # Estimate gradient component-wise
        grad_est = np.zeros_like(obs.x)
        for i in range(len(obs.x)):
            grad_est[i] = (
                (obs.value - self.x_prev_val) / (2 * cn) * self.perturbations[i]
            )

        # Update position with gradient clipping
        grad_est = np.clip(grad_est, -10.0, 10.0)  # Overflow guard
        x_new = self.x_prev - (self.lr / np.sqrt(self.n)) * grad_est

        # Store current state
        self.x_prev = obs.x.copy()
        self.x_prev_val = obs.value
        self.perturbations = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(
            float
        )

        # Return perturbed point
        return x_new + cn * self.perturbations


class NedicSubgradient(OptimizerProtocol):
    """Nedic's subgradient method for zero-order optimization."""

    def __init__(self, lr: float = 0.005, name: str = "NedicSubgradient"):
        self.lr = lr
        self.name = name
        self.x_prev = None
        self.x_prev_val = None
        self.step_count = 0

    def reset(self):
        self.x_prev = None
        self.x_prev_val = None
        self.step_count = 0

    def step(self, obs: Observation) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.x_prev_val = obs.value
            # Return point with small random perturbation
            delta = np.random.randn(*obs.x.shape) * 0.01
            return self.x_prev + delta

        # Subgradient approximation
        delta = obs.x - self.x_prev
        delta_norm = np.linalg.norm(delta) + 1e-8
        subgrad = (obs.value - self.x_prev_val) / delta_norm * (delta / delta_norm)

        # Update with diminishing step size
        self.step_count += 1
        lr_t = self.lr / np.sqrt(self.step_count)

        x_new = self.x_prev - lr_t * subgrad

        # Store current state
        self.x_prev = obs.x.copy()
        self.x_prev_val = obs.value

        # Return perturbed point
        delta_next = np.random.randn(*obs.x.shape) * 0.01
        return x_new + delta_next


class AcceleratedSPSA(OptimizerProtocol):
    """Accelerated SPSA (Granichin, 2002) with momentum — lr=0.005."""

    def __init__(
        self,
        lr: float = 0.005,
        perturb: float = 0.1,
        beta: float = 0.9,
        name: str = "AcceleratedSPSA",
    ):
        self.lr = lr
        self.perturb = perturb
        self.beta = beta
        self.name = name
        self.x_prev = None
        self.x_prev_val = None
        self.momentum = None
        self.delta = None

    def reset(self):
        self.x_prev = None
        self.x_prev_val = None
        self.momentum = None
        self.delta = None

    def step(self, obs: Observation) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = obs.x.copy()
            self.x_prev_val = obs.value
            self.delta = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(float)
            return self.x_prev + self.perturb * self.delta

        # Estimate gradient
        grad_est = (obs.value - self.x_prev_val) / (2 * self.perturb) * self.delta

        # Momentum update
        if self.momentum is None:
            self.momentum = np.zeros_like(grad_est)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * grad_est

        # Update position
        x_new = self.x_prev - self.lr * self.momentum

        # Store current state
        self.x_prev = obs.x.copy()
        self.x_prev_val = obs.value
        self.delta = np.random.choice([-1, 1], size=obs.x.shape[0]).astype(float)

        # Return perturbed point
        return x_new + self.perturb * self.delta


class CMAES(OptimizerProtocol):
    """Covariance Matrix Adaptation Evolution Strategy (simplified)."""

    def __init__(
        self,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
        name: str = "CMAES",
    ):
        self.population_size = population_size
        self.sigma = sigma
        self.name = name
        self.dim = None
        self.population = None
        self.fitness = None
        self.generation = 0

    def reset(self):
        self.dim = None
        self.population = None
        self.fitness = None
        self.generation = 0

    def step(self, obs: Observation) -> np.ndarray:
        if self.dim is None:
            self.dim = obs.x.shape[0]
            if self.population_size is None:
                self.population_size = 4 + int(3 * np.log(self.dim))
            # Initialize population around current point
            self.population = np.array(
                [obs.x.copy() for _ in range(self.population_size)]
            )
            self.fitness = np.full(self.population_size, np.inf)
            self.current_idx = 0
            return self.population[0]

        # Store fitness for current individual
        self.fitness[self.current_idx] = obs.value

        # If population complete, select parents and generate offspring
        self.current_idx += 1
        if self.current_idx >= self.population_size:
            self.generation += 1

            # Select top 50% as parents
            idx_sorted = np.argsort(self.fitness)
            n_parents = self.population_size // 2
            parents = self.population[idx_sorted[:n_parents]]

            # Compute mean and covariance of parents
            mean = np.mean(parents, axis=0)
            if n_parents > 1:
                cov = np.cov(parents.T) + np.eye(self.dim) * 1e-6
            else:
                cov = np.eye(self.dim) * self.sigma**2

            # Generate new population (offspring)
            self.population = np.random.multivariate_normal(
                mean, cov, size=self.population_size
            )
            self.fitness = np.full(self.population_size, np.inf)
            self.current_idx = 0

        return self.population[self.current_idx]


class GPUCB(OptimizerProtocol):
    """Simplified Gaussian Process UCB for Bayesian optimization."""

    def __init__(self, beta: float = 2.0, name: str = "GPUCB"):
        self.beta = beta
        self.name = name
        self.X = []
        self.y = []
        self.dim = None

    def reset(self):
        self.X = []
        self.y = []
        self.dim = None

    def step(self, obs: Observation) -> np.ndarray:
        if self.dim is None:
            self.dim = obs.x.shape[0]
            # Initial random point
            return np.random.randn(self.dim) * 0.1

        # Store observation
        self.X.append(obs.x.copy())
        self.y.append(obs.value)

        if len(self.X) < 5:
            # Initial exploration phase
            return np.random.randn(self.dim) * 0.5

        # Simplified GP-UCB: select point with highest uncertainty estimate
        X_arr = np.array(self.X)
        y_arr = np.array(self.y)

        # Distance-based uncertainty (simplified) with protection
        if len(X_arr) > 1:
            distances = np.linalg.norm(X_arr[-1] - X_arr[:-1], axis=1)
            uncertainty = np.mean(distances) if len(distances) > 0 else 1.0
        else:
            uncertainty = 1.0

        # UCB acquisition: move in random direction scaled by uncertainty (with clipping)
        direction = np.random.randn(self.dim)
        direction_norm = np.linalg.norm(direction) + 1e-8
        direction = direction / direction_norm

        step_size = np.clip(
            self.beta * uncertainty, 0.0, 2.0
        )  # Guard against excessively large steps
        x_new = obs.x.copy() + step_size * direction

        return x_new


# ============================================================================
# 2. ENVIRONMENT AND METRIC FACTORIES
# ============================================================================


def create_environment(
    rho: float, A: float, dim: int, noise_type: str = "gaussian", seed: int = 42
) -> DynamicEnvironment:
    """
    Create environment with controlled Hölder exponent ρ and drift magnitude A.

    Args:
        rho: Hölder exponent ∈ (0, 1] for gradient smoothness
        A: Drift magnitude bound ‖θ_n - θ_{n-1}‖_1 ≤ A
        dim: Problem dimensionality
        noise_type: 'gaussian' or 'heavy_tailed'
        seed: Random seed

    Returns:
        Configured DynamicEnvironment instance
    """
    # Select landscape based on Hölder exponent ρ
    if rho == 1.0:
        landscape = QuadraticLandscape(dim=dim, condition_number=5.0, seed=seed)
    elif rho == 0.5:
        landscape = PNormLandscape(p=1.5)  # p = ρ+1 = 1.5
    elif rho == 0.2:
        landscape = RobustLandscape(delta=0.1)  # Approximate ρ=0.2 smoothness
    else:
        raise ValueError(f"Unsupported rho value: {rho}")

    # Linear drift with velocity scaled to achieve ‖Δθ‖_1 ≈ A
    # For uniform direction: ‖v‖_1 = d * |v_i|, so set v_i = A / d
    velocity = np.ones(dim) * (A / dim)

    env = DynamicEnvironment(
        dim=dim,
        drift=LinearDrift(velocity=velocity.tolist()),
        landscape=landscape,
        bounds=[-10, 10],
    )
    return env


def create_metrics(rho: float, A: float) -> MetricsCollection:
    """
    Create metrics collection with theoretically grounded Lyapunov analysis.

    Args:
        rho: Hölder exponent ∈ (0, 1]
        A: Drift magnitude bound

    Returns:
        MetricsCollection with Lyapunov metrics in ℓ_{ρ+1} geometry
    """
    metrics = [
        # Stabilization metrics (Definition 1)
        LyapunovMetric(rho=rho),
        NormalizedLyapunovMetric(rho=rho, drift_magnitude_A=A),
        AsymptoticBoundMetric(rho=rho),
        # Tracking performance in optimal ℓ_{ρ+1} norm
        TrackingErrorMetric(norm=rho + 1.0, normalize_by_dim=True),
        # Regret analysis
        DynamicRegretMetric(normalize_by_path=True),
        # Adaptation behavior
        DriftAdaptationMetric(),
    ]
    return MetricsCollection(metrics)


# ============================================================================
# 3. MAIN EXPERIMENT
# ============================================================================


def run_single_experiment(
    optimizer: OptimizerProtocol,
    rho: float,
    A: float,
    dim: int,
    T: int = 500,
    seed: int = 42,
    noise_type: str = "gaussian",
) -> ExperimentResult:
    """
    Execute single experiment with full Lyapunov analysis.

    Returns:
        ExperimentResult with ground truth theta_t and Lyapunov metrics
    """
    # Create environment
    env = create_environment(rho=rho, A=A, dim=dim, noise_type=noise_type, seed=seed)

    # Create oracle (FO for gradient methods, ZO for black-box)
    optimizer_name = (
        optimizer.name if hasattr(optimizer, "name") else optimizer.__class__.__name__
    )
    is_zero_order = optimizer_name in [
        "RandomSearch",
        "OnePointSPSA",
        "FiniteDiffForward",
        "FiniteDiffCentral",
        "FDSA",
        "SPSA",
        "ZOSGD",
        "ZOSignSGD",
        "QuadraticInterpolation",
        "KieferWolfowitz",
        "NedicSubgradient",
        "AcceleratedSPSA",
        "CMAES",
        "GPUCB",
    ]

    # Authoritative oracle-type tag (consumed by OptimizerInfo.from_optimizer),
    # avoiding the unreliable step()-signature heuristic.
    optimizer.oracle_type = "zero-order" if is_zero_order else "first-order"

    if not is_zero_order:
        oracle = FirstOrderOracle(
            env,
            value_noise=GaussianNoise(sigma=0.01, seed=seed),
            grad_noise=GaussianNoise(sigma=0.01, seed=seed),
            seed=seed,
            blind_value=False,  # legacy experiment relies on value being available
        )
    else:
        oracle = ZeroOrderOracle(
            env, value_noise=GaussianNoise(sigma=0.05, seed=seed), seed=seed
        )

    # Create metrics
    metrics = create_metrics(rho=rho, A=A)

    # Create runner
    runner = BenchmarkRunner(
        environment=env,
        oracle=oracle,
        metrics=metrics,
        record_trajectory=True,
        tail_fraction=0.2,
    )

    # Generate x0 from the experiment seed before the runner starts. Keeping this
    # generator local makes a saved seed sufficient to reproduce a single run.
    x0 = np.random.default_rng(seed).normal(0.0, 0.1, size=dim)

    # Run experiment
    result = runner.run(optimizer=optimizer, T=T, x0=x0, seed=seed)

    return result


def run_full_experiment_suite(
    output_dir: str = "results/lyapunov",
    seeds: Optional[List[int]] = None,
    T: int = 500,
    rho_values: Optional[List[float]] = None,
    A_values: Optional[List[float]] = None,
    dimensions: Optional[List[int]] = None,
    optimizer_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run complete experimental suite across (ρ, A) regimes.

    Returns:
        Dictionary with aggregated results and metadata
    """
    seeds = [42, 43, 44, 45, 46] if seeds is None else seeds
    rho_values = [1.0, 0.5, 0.2] if rho_values is None else rho_values
    A_values = [0.001, 0.01, 0.1, 0.3, 0.6, 1.0] if A_values is None else A_values
    dimensions = [5] if dimensions is None else dimensions

    # Optimizer configurations (12 first-order + 13 zero-order algorithms)
    optimizers_config = [
        # First-Order methods (12)
        ("SGD", lambda: SGD(lr=0.1)),
        ("SGD_Polyak", lambda: SGDPolyak(lr=0.1)),
        ("HeavyBall", lambda: HeavyBall(lr=0.1, beta=0.9)),
        ("Nesterov", lambda: Nesterov(lr=0.05, beta=0.9)),
        ("Adam", lambda: Adam(lr=0.001, beta1=0.9, beta2=0.999)),
        ("AdamW", lambda: AdamW(lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)),
        ("AMSGrad", lambda: AMSGrad(lr=0.001, beta1=0.9, beta2=0.999)),
        ("SMD", lambda: SMD(lr=0.1)),
        ("RDA", lambda: RDA(lr=0.1, lambda_reg=0.01)),
        ("ProxSGD", lambda: ProxSGD(lr=0.1, lambda_reg=0.01)),
        ("AdaptiveLR", lambda: AdaptiveLR(lr0=0.1)),
        ("SignSGD", lambda: SignSGD(lr=0.05)),
        # Zero-order methods (13); SPSA-family methods use lr=0.005.
        ("RandomSearch", lambda: RandomSearch(lr=0.1, scale=0.5)),
        ("OnePointSPSA", lambda: OnePointSPSA(lr=0.005, perturb=0.1)),  # ← lr=0.005
        # ("FiniteDiffForward", lambda: FiniteDiffForward(lr=0.02, h=1e-4)),
        ("FiniteDiffCentral", lambda: FiniteDiffCentral(lr=0.02, h=1e-4)),
        ("FDSA", lambda: FDSA(lr=0.02, h=1e-4)),
        ("SPSA", lambda: SPSA(lr=0.005, perturb=0.1)),  # ← lr=0.005
        ("ZOSGD", lambda: ZOSGD(lr=0.005, mu=0.01)),  # ← lr=0.005
        ("ZOSignSGD", lambda: ZOSignSGD(lr=0.005, mu=0.01)),  # ← lr=0.005
        ("QuadraticInterpolation", lambda: QuadraticInterpolation(lr=0.1)),
        ("KieferWolfowitz", lambda: KieferWolfowitz(lr=0.005, cn=0.1)),  # ← lr=0.005
        ("NedicSubgradient", lambda: NedicSubgradient(lr=0.005)),  # ← lr=0.005
        (
            "AcceleratedSPSA",
            lambda: AcceleratedSPSA(lr=0.005, perturb=0.1, beta=0.9),
        ),  # ← lr=0.005
        ("CMAES", lambda: CMAES(sigma=0.5)),
        ("GPUCB", lambda: GPUCB(beta=2.0)),
    ]

    if optimizer_names is not None:
        requested = set(optimizer_names)
        known = {name for name, _ in optimizers_config}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown optimizer(s): {sorted(unknown)}")
        optimizers_config = [item for item in optimizers_config if item[0] in requested]

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_path / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Total number of runs
    total_runs = (
        len(rho_values)
        * len(A_values)
        * len(dimensions)
        * len(optimizers_config)
        * len(seeds)
    )
    print(f"🚀 Starting WIND experiment suite")
    print(f"   Configurations: ρ ∈ {rho_values}, A ∈ {A_values}, dim ∈ {dimensions}")
    print(
        f"   Algorithms: {len(optimizers_config)} optimizers (SPSA family with lr=0.005)"
    )
    print(f"   Seeds: {seeds}")
    print(f"   Total runs: {total_runs}")
    print(f"   Steps per run: {T}")
    print()

    # Progress tracking
    results = []
    run_counter = 0
    start_time = time.time()

    # Main experiment loop
    for rho in rho_values:
        for A in A_values:
            for dim in dimensions:
                for opt_name, opt_factory in optimizers_config:
                    for seed in seeds:
                        run_counter += 1

                        # Skip expensive methods in high-dim regimes (optimization)
                        if dim > 20 and opt_name in [
                            "CMAES",
                            "GPUCB",
                            "FiniteDiffForward",
                            "FiniteDiffCentral",
                        ]:
                            continue

                        # Create optimizer
                        try:
                            optimizer = opt_factory()
                            if not hasattr(optimizer, "name"):
                                optimizer.name = opt_name
                        except Exception as e:
                            print(f"⚠️  Failed to create {opt_name}: {e}")
                            continue

                        # Run experiment
                        try:
                            result = run_single_experiment(
                                optimizer=optimizer,
                                rho=rho,
                                A=A,
                                dim=dim,
                                T=T,
                                seed=seed,
                                noise_type="gaussian",
                            )

                            # Extract Lyapunov metrics for stabilization analysis
                            lyapunov_val = result.final_metrics.get(
                                f"lyapunov_rho{rho:.2f}", np.nan
                            )
                            normalized_lyapunov = result.final_metrics.get(
                                f"norm_lyapunov_rho{rho:.2f}_A{A:.3f}", np.nan
                            )
                            asymptotic_bound = result.final_metrics.get(
                                f"asymptotic_bound_rho{rho:.2f}", np.nan
                            )
                            tracking_error = result.final_metrics.get(
                                f"error_p{rho+1.0:.2f}", np.nan
                            )

                            # Check stabilization (Definition 1): ∃ C > 0 such that 𝔼[V_n] ≤ C
                            is_stabilized = (
                                not np.isnan(lyapunov_val) and lyapunov_val < 100.0
                            )

                            # Check asymptotic bound (Definition 2): lim sup 𝔼[V_n] ≤ L < ∞
                            has_finite_bound = (
                                not np.isnan(asymptotic_bound)
                                and asymptotic_bound < 1000.0
                            )

                            # Store result
                            results.append(
                                {
                                    "algorithm": opt_name,
                                    "rho": rho,
                                    "A": A,
                                    "dim": dim,
                                    "seed": seed,
                                    "lyapunov": lyapunov_val,
                                    "normalized_lyapunov": normalized_lyapunov,
                                    "asymptotic_bound": asymptotic_bound,
                                    "tracking_error": tracking_error,
                                    "is_stabilized": is_stabilized,
                                    "has_finite_bound": has_finite_bound,
                                    "runtime": result.runtime,
                                    "status": result.status,
                                }
                            )

                            # Save individual result to JSON
                            result_filename = (
                                f"rho{rho:.1f}_A{A:.3f}_dim{dim}_"
                                f"{opt_name}_seed{seed}.json"
                            )
                            result.save_to_json(str(output_path / result_filename))

                            # Progress update
                            elapsed = time.time() - start_time
                            eta = elapsed / run_counter * (total_runs - run_counter)
                            print(
                                f"[{run_counter}/{total_runs}] ρ={rho:.1f}, A={A:.3f}, "
                                f"dim={dim}, algo={opt_name:25s}, seed={seed} "
                                f"→ Lyapunov={lyapunov_val:.2f}, "
                                f"stabilized={is_stabilized}, "
                                f"runtime={result.runtime:.2f}s "
                                f"(ETA: {eta/60:.1f} min)"
                            )

                        except Exception as e:
                            print(
                                f"❌ Failed run {run_counter}/{total_runs} "
                                f"(ρ={rho}, A={A}, dim={dim}, {opt_name}, seed={seed}): {e}"
                            )
                            continue

                        # Periodic checkpoint (every 50 runs)
                        if run_counter % 50 == 0:
                            checkpoint_file = (
                                checkpoints_dir / f"checkpoint_{run_counter}.json"
                            )
                            with open(checkpoint_file, "w") as f:
                                json.dump(results, f, indent=2)
                            print(f"💾 Checkpoint saved: {checkpoint_file}")

    # Final aggregation
    results_df = pd.DataFrame(results)

    # Save full results
    results_df.to_csv(output_path / "full_results.csv", index=False)

    # Save aggregated statistics
    agg_stats = (
        results_df.groupby(["algorithm", "rho", "A"])
        .agg(
            {
                "lyapunov": ["mean", "std", "min", "max"],
                "normalized_lyapunov": ["mean", "std"],
                "asymptotic_bound": ["mean", "std"],
                "tracking_error": ["mean", "std"],
                "is_stabilized": "mean",  # Fraction stabilized
                "has_finite_bound": "mean",  # Fraction with finite bound
                "runtime": ["mean", "std"],
            }
        )
        .round(4)
    )
    agg_stats.to_csv(output_path / "aggregated_statistics.csv")

    # Save experiment metadata
    metadata = {
        "experiment_id": f"sthrd_lyapunov_{time.strftime('%Y%m%d_%H%M%S')}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runs": run_counter,
        "rho_values": rho_values,
        "A_values": A_values,
        "dimensions": dimensions,
        "algorithms": [name for name, _ in optimizers_config],
        "seeds": seeds,
        "T": T,
        "elapsed_time_seconds": time.time() - start_time,
        "output_directory": str(output_path),
        "spsa_lr": 0.005,  # Explicitly recorded for publication reproducibility
        "heavy_ball_lr": 0.1,
        "adam_lr": 0.001,
    }
    with open(output_path / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Experiment suite completed!")
    print(f"   Results saved to: {output_path}")
    print(f"   Total runtime: {metadata['elapsed_time_seconds']/60:.1f} minutes")
    print(f"   Successful runs: {len(results)} / {run_counter}")

    return {
        "results_df": results_df,
        "aggregated_stats": agg_stats,
        "metadata": metadata,
        "output_path": output_path,
    }


# ============================================================================
# 4. COMMAND-LINE EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("WIND BENCHMARK: Theoretically Grounded Stabilization Analysis")
    print("=" * 80)
    print()
    print("Validating Definitions 1-2 for 25 optimization algorithms:")
    print("  Definition 1 (Stabilization): ∃ C > 0 such that 𝔼[V_n(x̂_n)] ≤ C ∀ n")
    print("  Definition 2 (Asymptotic Bound): L = lim sup_{n→∞} 𝔼[V_n(x̂_n)] < ∞")
    print()
    print("Where V_n(x) = ‖x - θ_n‖_{ρ+1}^{ρ+1} is the Lyapunov function")
    print("for ρ-Hölder smooth gradients with drift magnitude A.")
    print()
    print("⚠️  CRITICAL PARAMETER: SPSA family uses lr=0.005 for stability")
    print()

    # Run experiment
    results = run_full_experiment_suite(
        output_dir="results/lyapunov", seeds=[42, 43, 44, 45, 46], T=500
    )

    # Summary statistics
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    df = results["results_df"]

    # Overall stabilization rate
    total_stabilized = df["is_stabilized"].sum()
    total_runs = len(df)
    stabilization_rate = total_stabilized / total_runs * 100

    print(
        f"\nOverall stabilization rate: {stabilization_rate:.1f}% "
        f"({total_stabilized}/{total_runs} runs)"
    )

    # Best algorithms by ρ regime
    print("\nTop 3 algorithms by Hölder exponent ρ:")
    for rho in [1.0, 0.5, 0.2]:
        rho_df = df[df["rho"] == rho]
        algo_stats = (
            rho_df.groupby("algorithm")["normalized_lyapunov"]
            .mean()
            .sort_values()
            .head(3)
        )
        print(f"\n  ρ = {rho:.1f}:")
        for rank, (algo, val) in enumerate(algo_stats.items(), 1):
            print(f"    {rank}. {algo:25s} → normalized Lyapunov = {val:.3f}")

    # Drift robustness
    print("\nDrift robustness (normalized Lyapunov across A ∈ [0.001, 1.0]):")
    algo_drift_robustness = (
        df.groupby("algorithm")["normalized_lyapunov"].std().sort_values().head(5)
    )
    for rank, (algo, std_val) in enumerate(algo_drift_robustness.items(), 1):
        print(f"  {rank}. {algo:25s} → std = {std_val:.3f} (lower = more robust)")

    print("\n" + "=" * 80)
    print("✅ All results saved to ./results/lyapunov/")
    print("   Key files:")
    print("     • full_results.csv          - Raw results for all runs")
    print("     • aggregated_statistics.csv - Mean/std per (algo, ρ, A)")
    print("     • experiment_metadata.json  - Complete experiment configuration")
    print("     • *.json                    - Individual run results with ground truth")
    print("=" * 80)
