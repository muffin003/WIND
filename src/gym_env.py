"""
Gymnasium adapter for WIND.

Wraps a WIND ``DynamicEnvironment`` + ``Oracle`` as a Gymnasium environment, so
that learned / RL policies can be trained to track a drifting optimum under the
same dynamics, geometries and noise models used by the benchmark.

The WIND core is untouched: this module is self-contained and only imported on
demand. ``gymnasium`` is an OPTIONAL dependency (``pip install gymnasium``).

MDP interpretation (a POMDP — theta_t is hidden from the agent):
    action      : an absolute point/proposal (``action_mode="absolute"``) or an
                  increment (``action_mode="delta"``). Euclidean actions are
                  clipped, simplex actions are projected, and Stiefel/Grassmann
                  actions are projected or retracted according to the selected
                  geometry.
    observation : the agent's current position x_t concatenated with the noisy
                  value and/or gradient returned by the oracle (NEVER theta_t).
    reward      : -(L_t(x_t) - L_t(theta_t)) = -instantaneous regret  (``"neg_regret"``)
                  or a domain-specific negative tracking distance
                  (``"neg_error"``).
    transition  : theta_{t+1} = drift(theta_t, t, x_t). With AdaptiveDrift the
                  transition depends on the action (a genuine control problem);
                  with the other drifts the optimum evolves exogenously (tracking).
    episode     : truncated after ``T`` steps (no terminal state by default).

The privileged comparator (theta_t / optimum value) is used ONLY to compute the
reward inside the wrapper and is exposed in ``info`` for logging — never inside the
observation, preserving WIND's information barrier.
"""

from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without gymnasium
    raise ImportError(
        "WindGymEnv requires the optional dependency 'gymnasium'. "
        "Install it with: pip install gymnasium"
    ) from exc

from .core import (
    DynamicEnvironment,
    GrassmannLandscape,
    SimplexLandscape,
    StiefelLandscape,
    make_environment,
)
from .manifold import (
    principal_angle_distance,
    project_to_stiefel,
    retract,
    tangent_project,
)
from .oracle import Oracle, ZeroOrderOracle


class WindGymEnv(gym.Env):
    """Gymnasium wrapper around a WIND environment + oracle.

    Args:
        environment: A configured ``DynamicEnvironment``.
        oracle: An ``Oracle`` bound to ``environment``. Defaults to a
            ``ZeroOrderOracle`` (classic value-only RL feedback).
        T: Episode horizon (number of steps before ``truncated``).
        action_mode: ``"absolute"`` (action = next x) or ``"delta"`` (x += action).
        x_bounds: ``(low, high)`` box bounds for the position (and absolute action).
        max_step: Per-coordinate bound on the increment in ``"delta"`` mode.
        reward: ``"neg_regret"`` (default) or ``"neg_error"``.
        x0: Initial position. Defaults to zeros.
        terminate_on_diverge: If set, end the episode (terminated=True) once the
            tracking error exceeds this value.
        geometry: ``"auto"`` (default), ``"euclidean"``, ``"simplex"``,
            ``"stiefel"``, or ``"grassmann"``. Auto-detection uses the configured
            landscape. The Euclidean path preserves the original clipping
            behavior; constrained modes enforce feasible actions by projection or
            retraction. Stiefel error tracks frames, whereas Grassmann error tracks
            subspaces through their principal angles.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: DynamicEnvironment,
        oracle: Optional[Oracle] = None,
        T: int = 200,
        action_mode: str = "absolute",
        x_bounds: Tuple[float, float] = (-10.0, 10.0),
        max_step: float = 1.0,
        reward: str = "neg_regret",
        x0: Optional[np.ndarray] = None,
        terminate_on_diverge: Optional[float] = None,
        geometry: str = "auto",
    ):
        super().__init__()
        if action_mode not in ("absolute", "delta"):
            raise ValueError("action_mode must be 'absolute' or 'delta'")
        if reward not in ("neg_regret", "neg_error"):
            raise ValueError("reward must be 'neg_regret' or 'neg_error'")
        if geometry not in (
            "auto",
            "euclidean",
            "simplex",
            "stiefel",
            "grassmann",
        ):
            raise ValueError(
                "geometry must be 'auto', 'euclidean', 'simplex', 'stiefel', "
                "or 'grassmann'"
            )

        self.env = environment
        self.oracle = oracle if oracle is not None else ZeroOrderOracle(environment)
        self.T = int(T)
        self.action_mode = action_mode
        self.low, self.high = float(x_bounds[0]), float(x_bounds[1])
        self.max_step = float(max_step)
        self.reward_kind = reward
        self.terminate_on_diverge = terminate_on_diverge

        self.dim = environment.dim
        self.geometry = self._resolve_geometry(geometry)
        self._stiefel_shape: Optional[Tuple[int, int]] = None
        if self.geometry in ("stiefel", "grassmann"):
            landscape = self.env.landscape
            expected_type = (
                StiefelLandscape if self.geometry == "stiefel" else GrassmannLandscape
            )
            if not isinstance(landscape, expected_type):
                raise ValueError(
                    f"geometry='{self.geometry}' requires a "
                    f"{expected_type.__name__} so d and r can be determined"
                )
            self._stiefel_shape = (landscape.d, landscape.r)
            if landscape.d * landscape.r != self.dim:
                raise ValueError(
                    "environment.dim must equal d*r for Stiefel/Grassmann "
                    "geometry: "
                    f"got dim={self.dim}, d={landscape.d}, r={landscape.r}"
                )
            if self.env.bounds is not None:
                raise ValueError(
                    "Stiefel/Grassmann geometry requires environment bounds=None "
                    "because coordinate clipping breaks orthonormality"
                )

        raw_x0 = np.zeros(self.dim) if x0 is None else np.asarray(x0, float)
        if raw_x0.size != self.dim:
            raise ValueError(
                f"x0 must contain {self.dim} entries, got shape {raw_x0.shape}"
            )
        self._x0 = self._project_point(raw_x0.reshape(self.dim))
        self._x = self._x0.copy()
        self._t = 0

        self._validate_latent_point(
            self.env.get_current_theta(for_analysis=True), context="initial theta"
        )

        # Probe once to discover which fields the oracle exposes (value/grad),
        # then restore clean state.
        self._has_value, self._has_grad = self._probe_oracle()
        self.env.reset()
        self.oracle.reset()

        # --- Spaces ---
        if action_mode == "absolute":
            self.action_space = spaces.Box(
                low=self.low, high=self.high, shape=(self.dim,), dtype=np.float32
            )
        else:  # delta
            self.action_space = spaces.Box(
                low=-self.max_step,
                high=self.max_step,
                shape=(self.dim,),
                dtype=np.float32,
            )

        obs_dim = (
            self.dim
            + (1 if self._has_value else 0)
            + (self.dim if self._has_grad else 0)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------ helpers
    def _resolve_geometry(self, geometry: str) -> str:
        if geometry != "auto":
            return geometry
        if isinstance(self.env.landscape, SimplexLandscape):
            return "simplex"
        if isinstance(self.env.landscape, StiefelLandscape):
            return "stiefel"
        if isinstance(self.env.landscape, GrassmannLandscape):
            return "grassmann"
        return "euclidean"

    def _project_point(self, point: np.ndarray) -> np.ndarray:
        """Map an ambient point to the configured decision domain."""
        point = np.asarray(point, dtype=float).reshape(self.dim)
        if self.geometry == "simplex":
            return SimplexLandscape.project(point)
        if self.geometry in ("stiefel", "grassmann"):
            assert self._stiefel_shape is not None
            d, r = self._stiefel_shape
            return project_to_stiefel(point.reshape(d, r)).reshape(-1)
        return point.copy()

    def _constraint_violation(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float).reshape(self.dim)
        if self.geometry == "simplex":
            return float(
                max(abs(float(np.sum(point)) - 1.0), max(0.0, -float(np.min(point))))
            )
        if self.geometry in ("stiefel", "grassmann"):
            assert self._stiefel_shape is not None
            d, r = self._stiefel_shape
            X = point.reshape(d, r)
            return float(np.linalg.norm(X.T @ X - np.eye(r), ord="fro"))
        return 0.0

    def _validate_latent_point(self, theta: np.ndarray, context: str) -> None:
        violation = self._constraint_violation(theta)
        if violation > 1e-7:
            raise ValueError(
                f"{context} is outside the {self.geometry} domain "
                f"(constraint violation={violation:.3e}). Configure a feasible "
                "initial_theta and a domain-preserving drift."
            )

    def _apply_action(self, action: np.ndarray) -> np.ndarray:
        """Interpret an action without changing Euclidean-mode semantics."""
        action = np.asarray(action, dtype=float).reshape(self.dim)

        if self.geometry == "euclidean":
            if self.action_mode == "absolute":
                return np.clip(action, self.low, self.high)
            step = np.clip(action, -self.max_step, self.max_step)
            return np.clip(self._x + step, self.low, self.high)

        if self.action_mode == "absolute":
            proposal = np.clip(action, self.low, self.high)
            return self._project_point(proposal)

        ambient_step = np.clip(action, -self.max_step, self.max_step)
        if self.geometry == "simplex":
            return self._project_point(self._x + ambient_step)

        assert self._stiefel_shape is not None
        d, r = self._stiefel_shape
        X = self._x.reshape(d, r)
        A = ambient_step.reshape(d, r)
        tangent_step = tangent_project(X, A)
        return retract(X, tangent_step).reshape(-1)

    def _distance(self, x: np.ndarray, theta: np.ndarray) -> float:
        if self.geometry == "grassmann":
            assert self._stiefel_shape is not None
            d, r = self._stiefel_shape
            return principal_angle_distance(x.reshape(d, r), theta.reshape(d, r))
        # Flattened Euclidean distance is the Frobenius distance for Stiefel
        # frames and the standard ambient distance for vector/simplex points.
        return float(np.linalg.norm(x - theta))

    def _probe_oracle(self) -> Tuple[bool, bool]:
        self.env.reset()
        self.oracle.reset()
        self.oracle.start_step(0)
        obs = self.oracle.query(self._x0.copy())
        self.oracle.end_step()
        return obs.value is not None, obs.grad is not None

    def _build_obs(self, x: np.ndarray, observation) -> np.ndarray:
        parts = [x.astype(np.float32)]
        if self._has_value:
            val = 0.0 if observation.value is None else observation.value
            parts.append(np.array([val], dtype=np.float32))
        if self._has_grad:
            g = (
                np.zeros(self.dim, dtype=np.float32)
                if observation.grad is None
                else observation.grad.astype(np.float32)
            )
            parts.append(g)
        return np.concatenate(parts)

    def _query(self, x: np.ndarray):
        """Query the oracle at x under the current theta_t (no advance)."""
        self.oracle.start_step(self.env.t)
        observation = self.oracle.query(x)
        theta = self.oracle.get_current_theta()
        self.oracle.end_step()
        self._validate_latent_point(theta, context=f"theta at t={self.env.t}")
        return observation, theta

    def _reward(self, x: np.ndarray, theta: np.ndarray) -> float:
        if self.reward_kind == "neg_error":
            return -self._distance(x, theta)
        # neg_regret: -(L_t(x) - L_t(theta)) using the TRUE (clean) loss.
        regret = self.env.landscape.loss(x, theta) - self.env.landscape.loss(
            theta, theta
        )
        return -float(regret)

    # -------------------------------------------------------------------- gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.env.reset()
        self.oracle.reset()  # restores component RNG state -> reproducible episodes
        self._t = 0
        self._x = self._x0.copy()

        observation, theta = self._query(self._x)
        obs = self._build_obs(self._x, observation)
        info = {
            "theta": theta.copy(),
            "error": self._distance(self._x, theta),
            "geometry": self.geometry,
            "constraint_violation": self._constraint_violation(self._x),
        }
        return obs, info

    def step(self, action: np.ndarray):
        x = self._apply_action(action)
        self._x = x

        # Evaluate at x under the current theta_t, then advance the environment.
        observation, theta = self._query(x)
        reward = self._reward(x, theta)
        self.env.step(action=x)  # theta_{t+1} = drift(theta_t, t, x)
        self._t += 1

        error = self._distance(x, theta)
        truncated = self._t >= self.T
        terminated = bool(
            self.terminate_on_diverge is not None and error > self.terminate_on_diverge
        )

        obs = self._build_obs(x, observation)
        info = {
            "theta": theta.copy(),
            "error": error,
            "t": self._t,
            "geometry": self.geometry,
            "constraint_violation": self._constraint_violation(x),
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        oracle: str = "zero-order",
        seed: int = 42,
        **kwargs,
    ) -> "WindGymEnv":
        """Build a WindGymEnv from a WIND environment config dict.

        Args:
            config: passed to ``core.make_environment``.
            oracle: ``"zero-order"`` (default) or ``"first-order"``.
            seed: reproducibility seed for the environment / oracle.
            **kwargs: forwarded to ``WindGymEnv`` (T, action_mode, reward, ...).
        """
        from .oracle import make_oracle

        env = make_environment(config, seed=seed)
        oracle_obj = make_oracle(oracle, env, seed=seed)
        return cls(env, oracle_obj, **kwargs)
