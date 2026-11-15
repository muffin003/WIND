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
    reward      : negative observed value (``"neg_value"``), or explicit zero
                  reward for gradient-only feedback (``"zero"``).
    transition  : theta_{t+1} = drift(theta_t, t, x_t). With AdaptiveDrift the
                  transition depends on the action (a genuine control problem);
                  with the other drifts the optimum evolves exogenously (tracking).
    episode     : truncated after ``T`` steps (no terminal state by default).

True states and clean metrics are sent only to an optional evaluator-owned
GymEvaluationSession. They never enter observation, reward, or info.
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

from wind_benchmark.core.environment import (
    DynamicEnvironment,
    GrassmannLandscape,
    SimplexLandscape,
    StiefelLandscape,
    make_environment,
)
from wind_benchmark.core.geometry import (
    principal_angle_distance,
    project_to_stiefel,
    retract,
    tangent_project,
)
from wind_benchmark.core.oracle import Oracle, ZeroOrderOracle
from wind_benchmark.integrations.gym_evaluation import (
    GymEvaluationSession,
    GymEvaluationStep,
)
from wind_benchmark.core.privacy import (
    TruthGuard,
    _internal,
    _bind,
    _bind_environment,
    _owner_for_session,
    _Lease,
    _validate_component,
    _user_call,
)
from wind_benchmark.core.randomness import (
    reseed_environment,
    reseed_oracle,
    stream_seed,
)


class WindGymEnv(TruthGuard, gym.Env):
    """Gymnasium wrapper around a WIND environment + oracle.

    Args:
        environment: A configured ``DynamicEnvironment``.
        oracle: An ``Oracle`` bound to ``environment``. Defaults to a
            ``ZeroOrderOracle`` (classic value-only RL feedback).
        T: Episode horizon (number of steps before ``truncated``).
        action_mode: ``"absolute"`` (action = next x) or ``"delta"`` (x += action).
        x_bounds: ``(low, high)`` box bounds for the position (and absolute action).
        max_step: Per-coordinate bound on the increment in ``"delta"`` mode.
        reward: ``"neg_value"`` (default) or ``"zero"`` for gradient-only tasks.
        x0: Initial position. Defaults to zeros.
        terminate_on_diverge: Must be None; hidden-state termination is forbidden.
        evaluation_session: Optional trusted recorder, kept outside policy access.
        geometry: ``"auto"`` (default), ``"euclidean"``, ``"simplex"``,
            ``"stiefel"``, or ``"grassmann"``. Auto-detection uses the configured
            landscape. The Euclidean path preserves the original clipping
            behavior; constrained modes enforce feasible actions by projection or
            retraction. Stiefel error tracks frames, whereas Grassmann error tracks
            subspaces through their principal angles.
    """

    metadata = {"render_modes": []}
    _public_during_run = frozenset(
        {
            "reset",
            "step",
            "close",
            "render",
            "unwrapped",
            "spec",
            "metadata",
            "render_mode",
            "action_space",
            "observation_space",
            "np_random",
            "np_random_seed",
            "_np_random",
            "_np_random_seed",
            "dim",
            "geometry",
            "T",
            "action_mode",
            "low",
            "high",
            "max_step",
            "reward_kind",
            "from_config",
        }
    )

    @_internal
    def __init__(
        self,
        environment: DynamicEnvironment,
        oracle: Optional[Oracle] = None,
        T: int = 200,
        action_mode: str = "absolute",
        x_bounds: Tuple[float, float] = (-10.0, 10.0),
        max_step: float = 1.0,
        reward: str = "neg_value",
        x0: Optional[np.ndarray] = None,
        terminate_on_diverge: Optional[float] = None,
        geometry: str = "auto",
        evaluation_session: Optional[GymEvaluationSession] = None,
    ):
        if any(type(value) is not str for value in (action_mode, reward, geometry)):
            raise ValueError("Action, reward and geometry modes must be strings")
        if type(T) not in (int, np.int32, np.int64):
            raise ValueError("T must be an integer")
        for component in (environment, oracle):
            _validate_component(component)
        for component in (environment.drift, environment.landscape):
            _validate_component(component)
        if oracle is not None:
            _validate_component(oracle.value_noise)
            _validate_component(oracle.grad_noise)
        super().__init__()
        if action_mode not in ("absolute", "delta"):
            raise ValueError("action_mode must be 'absolute' or 'delta'")
        if reward not in ("neg_value", "zero"):
            raise ValueError(
                "reward must be 'neg_value' or 'zero'; clean rewards are evaluation-only"
            )
        if terminate_on_diverge is not None:
            raise ValueError("terminate_on_diverge leaks hidden-state information")
        if T <= 0:
            raise ValueError("T must be positive")
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

        self.__environment = environment
        self.__oracle = oracle if oracle is not None else ZeroOrderOracle(environment)
        self.T = _user_call(int, T)
        self.action_mode = action_mode
        self.low, self.high = _user_call(
            lambda: (float(x_bounds[0]), float(x_bounds[1]))
        )
        self.max_step = _user_call(float, max_step)
        self.reward_kind = reward
        self.__evaluation_session = evaluation_session
        self.__episode_id = None
        self.__active = False
        self.__has_reset = False
        self.__closed = False
        self.__registered = False
        self.__owner = (
            evaluation_session._GymEvaluationSession__owner
            if evaluation_session is not None
            else _owner_for_session()
        )

        self.dim = environment.dim
        self.geometry = self._resolve_geometry(geometry)
        self._stiefel_shape: Optional[Tuple[int, int]] = None
        if self.geometry in ("stiefel", "grassmann"):
            landscape = self.__environment.landscape
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
            if self.__environment.bounds is not None:
                raise ValueError(
                    "Stiefel/Grassmann geometry requires environment bounds=None "
                    "because coordinate clipping breaks orthonormality"
                )

        raw_x0 = np.zeros(self.dim) if x0 is None else _user_call(np.asarray, x0, float)
        if raw_x0.size != self.dim:
            raise ValueError(
                f"x0 must contain {self.dim} entries, got shape {raw_x0.shape}"
            )
        self._x0 = self._project_point(raw_x0.reshape(self.dim))
        self._x = self._x0.copy()
        self._t = 0

        self._validate_latent_point(
            self.__environment.get_current_theta(for_analysis=True),
            context="initial theta",
        )

        # Probe once to discover which fields the oracle exposes (value/grad),
        # then restore clean state.
        self._has_value, self._has_grad = self._probe_oracle()
        if reward == "neg_value" and not self._has_value:
            raise ValueError(
                "neg_value requires value feedback; use reward='zero' for gradient-only tasks"
            )
        self.__environment.reset()
        self.__oracle.reset()

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
        if isinstance(self.__environment.landscape, SimplexLandscape):
            return "simplex"
        if isinstance(self.__environment.landscape, StiefelLandscape):
            return "stiefel"
        if isinstance(self.__environment.landscape, GrassmannLandscape):
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
                f"Latent state is outside the configured domain. Configure a feasible "
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
        self.__environment.reset()
        self.__oracle.reset()
        self.__oracle.start_step(0)
        obs = self.__oracle.query(self._x0.copy())
        self.__oracle.end_step()
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
        """Evaluate privately under one locked state."""
        self.__oracle.start_step(self.__environment.t)
        try:
            observation = self.__oracle.query(x)
            record = self.__oracle._take_evaluation()
        finally:
            self.__oracle._last_evaluation = None
            self.__oracle.end_step()
        self._validate_latent_point(record.theta, context="current state")
        return observation, record

    def _record(self, record):
        if self.__evaluation_session is not None:
            self.__evaluation_session._append(
                self.__episode_id,
                GymEvaluationStep(
                    record,
                    self._distance(record.x, record.theta),
                    self._constraint_violation(record.x),
                    self._t,
                ),
            )

    def _public_info(self):
        return {
            "t": self._t,
            "geometry": self.geometry,
            "constraint_violation": self._constraint_violation(self._x),
            "query_count": self.__oracle.n_queries,
            "value_queries": self.__oracle.n_value_queries,
            "gradient_queries": self.__oracle.n_grad_queries,
        }

    def _abort(self):
        self.__active = False
        if self.__evaluation_session is not None:
            self.__evaluation_session.abort()
        if self.__registered:
            self.__owner._end(False, count=False)
            self.__registered = False
        elif isinstance(self.__owner, _Lease):
            self.__owner._finish(False)

    # -------------------------------------------------------------------- gym API
    @_internal
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None and type(seed) not in (
            int,
            np.int32,
            np.int64,
            np.uint32,
            np.uint64,
        ):
            raise ValueError("seed must be an integer")
        if options is not None and (
            type(options) is not dict
            or set(options) - {"replay"}
            or type(options.get("replay", False)) is not bool
        ):
            raise ValueError("options accepts only a boolean replay flag")
        if self.__closed or self.__owner.state != "running":
            raise RuntimeError("Environment is closed")
        _bind_environment(self.__environment, self.__oracle, self.__owner)
        _bind(self, self.__owner)
        super().reset(seed=seed)
        replay = bool((options or {}).get("replay", False))
        if replay and seed is not None:
            raise ValueError("Use either seed for a new seeded episode or replay=True")
        if self.__evaluation_session is not None:
            self.__evaluation_session._check_open()
            if self.__active:
                self._abort()
                raise RuntimeError("Cannot reset an unfinished recorded episode")
        try:
            if self.__registered:
                self.__owner._end(True, count=False)
                self.__registered = False
            if not isinstance(self.__owner, _Lease):
                self.__owner._start()
                self.__registered = True
            # First unseeded reset honors the supplied task. Subsequent unseeded
            # episodes draw new streams; explicit seeds reproduce full episodes.
            if not replay and (seed is not None or self.__has_reset):
                episode_seed = (
                    int(seed)
                    if seed is not None
                    else int(self.np_random.integers(0, 2**32))
                )
                reseed_environment(
                    self.__environment, stream_seed(episode_seed, "environment")
                )
                reseed_oracle(self.__oracle, stream_seed(episode_seed, "oracle"))
            self.__environment.reset()
            self.__oracle.reset()
            self._t = 0
            self._x = self._x0.copy()
            if self.__evaluation_session is not None:
                self.__episode_id = self.__evaluation_session._begin()
            observation, record = self._query(self._x)
            self._record(record)
            self.__active = True
            self.__has_reset = True
            return self._build_obs(self._x, observation), self._public_info()
        except Exception:
            self._abort()
            raise

    @_internal
    def step(self, action: np.ndarray):
        action = _user_call(np.asarray, action, dtype=float)
        if self.__closed or self.__owner.state != "running":
            raise RuntimeError("Environment is closed")
        if self.__evaluation_session is not None:
            self.__evaluation_session._check_open()
        if not self.__active:
            raise RuntimeError("Call reset() before stepping a new episode")
        try:
            x = self._apply_action(action)
            self._x = x
            observation, record = self._query(x)
            obs = self._build_obs(x, observation)
            if self.reward_kind == "neg_value":
                if observation.value is None:
                    raise RuntimeError("Value feedback is unavailable for this reward")
                # Use exactly the value visible in the observation (float32).
                reward = -float(obs[self.dim])
            else:
                reward = 0.0
            self.__environment.step(action=x)
            self._t += 1
            self._record(record)
            truncated = self._t >= self.T
            if truncated:
                self.__active = False
                if self.__registered:
                    self.__owner._end(True, count=False)
                    self.__registered = False
                if self.__evaluation_session is not None:
                    self.__evaluation_session._complete(self.__episode_id)
            return obs, reward, False, truncated, self._public_info()
        except Exception:
            self._abort()
            raise

    @_internal
    def close(self):
        if self.__closed:
            return
        if self.__active:
            self._abort()
        elif (
            isinstance(self.__owner, _Lease)
            and self.__evaluation_session is None
            and self.__owner.state == "running"
        ):
            self.__owner._finish(True)
        self.__closed = True

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
        from wind_benchmark.core.oracle import make_oracle

        env = make_environment(config, seed=seed)
        oracle_obj = make_oracle(oracle, env, seed=seed)
        return cls(env, oracle_obj, **kwargs)
