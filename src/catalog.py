"""Canonical optimizer catalog and backward-compatible name resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Type
import warnings

from .benchmark import OptimizerProtocol
from .experiment import (
    AMSGrad,
    Adam,
    AdaptiveLR,
    CoupledL2Adam,
    DiminishingGaussianSecant,
    DistanceScaledExploration,
    EliteCovarianceSearch,
    FiniteDiffCentral,
    GaussianTemporalSecant,
    HeavyBall,
    LookaheadNesterov,
    MomentumTemporalSecant,
    NormalizedRandomSecant,
    OnePointTemporalSecant,
    ProxSGD,
    QuadraticInterpolation,
    RDA,
    RandomSearch,
    SGD,
    SGDPolyak,
    SMD,
    ScaledTemporalSecant,
    ShrinkingTemporalSecant,
    SignSGD,
    SignedGaussianTemporalSecant,
)


@dataclass(frozen=True)
class OptimizerSpec:
    """Stable identity and interaction metadata for one optimizer."""

    canonical_name: str
    optimizer_class: Type[OptimizerProtocol]
    oracle_type: str
    implementation_id: str
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    alias_default_parameters: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    query_cost_per_step: int = 1
    update_schedule: str = "one-oracle-call-per-outer-step"
    memory_complexity: str = "unknown"

    def instantiate(
        self,
        parameters: Optional[Mapping[str, Any]] = None,
        *,
        requested_name: Optional[str] = None,
        warn_alias: bool = True,
    ) -> OptimizerProtocol:
        """Create the canonical implementation and preserve alias provenance."""
        requested = requested_name or self.canonical_name
        if requested != self.canonical_name and warn_alias:
            warnings.warn(
                f"{requested} is a legacy alias for {self.canonical_name}; "
                "new results use the canonical name.",
                FutureWarning,
                stacklevel=3,
            )
        kwargs = dict(
            self.alias_default_parameters.get(requested, self.default_parameters)
        )
        if parameters:
            kwargs.update(parameters)
        optimizer = self.optimizer_class(**kwargs)
        return attach_optimizer_metadata(optimizer, self, requested_name=requested)


OPTIMIZER_SPECS = (
    OptimizerSpec(
        "SGD", SGD, "first-order", "sgd_v1", {"lr": 0.1}, memory_complexity="O(d)"
    ),
    OptimizerSpec(
        "SGD_Polyak",
        SGDPolyak,
        "first-order",
        "returned_iterate_averaging_v1",
        {"lr": 0.1},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "HeavyBall",
        HeavyBall,
        "first-order",
        "heavy_ball_v1",
        {"lr": 0.1, "beta": 0.9},
        aliases=("Nesterov",),
        alias_default_parameters={"Nesterov": {"lr": 0.05, "beta": 0.9}},
        update_schedule="one-gradient-per-outer-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "LookaheadNesterov",
        LookaheadNesterov,
        "first-order",
        "lookahead_nesterov_v1",
        {"lr": 0.05, "beta": 0.9},
        update_schedule="one-gradient-at-extrapolated-point-per-outer-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "Adam",
        Adam,
        "first-order",
        "adam_v1",
        {"lr": 0.001, "beta1": 0.9, "beta2": 0.999},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "CoupledL2Adam",
        CoupledL2Adam,
        "first-order",
        "coupled_l2_adam_v1",
        {"lr": 0.001, "beta1": 0.9, "beta2": 0.999, "weight_decay": 0.01},
        aliases=("AdamW",),
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "AMSGrad",
        AMSGrad,
        "first-order",
        "amsgrad_v1",
        {"lr": 0.001, "beta1": 0.9, "beta2": 0.999},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "SMD",
        SMD,
        "first-order",
        "simplex_mirror_descent_v1",
        {"lr": 0.1},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "RDA",
        RDA,
        "first-order",
        "regularized_dual_averaging_v1",
        {"lr": 0.1, "lambda_reg": 0.01},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "ProxSGD",
        ProxSGD,
        "first-order",
        "prox_sgd_v1",
        {"lr": 0.1, "lambda_reg": 0.01},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "AdaptiveLR",
        AdaptiveLR,
        "first-order",
        "gradient_norm_adaptive_lr_v1",
        {"lr0": 0.1},
        memory_complexity="O(1)",
    ),
    OptimizerSpec(
        "SignSGD",
        SignSGD,
        "first-order",
        "sign_sgd_v1",
        {"lr": 0.05},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "RandomSearch",
        RandomSearch,
        "zero-order",
        "incumbent_random_search_v1",
        {"lr": 0.1, "scale": 0.5},
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "OnePointTemporalSecant",
        OnePointTemporalSecant,
        "zero-order",
        "one_point_temporal_secant_v1",
        {"lr": 0.005, "perturb": 0.1},
        aliases=("OnePointSPSA",),
        update_schedule="one-new-value-per-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "FiniteDiffCentral",
        FiniteDiffCentral,
        "zero-order",
        "central_coordinate_difference_v1",
        {"lr": 0.02, "h": 1e-4},
        update_schedule="2d-outer-step-coordinate-buffer",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "NormalizedRandomSecant",
        NormalizedRandomSecant,
        "zero-order",
        "normalized_random_secant_v1",
        {"lr": 0.02, "h": 1e-4},
        aliases=("FDSA",),
        update_schedule="one-new-value-per-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "ScaledTemporalSecant",
        ScaledTemporalSecant,
        "zero-order",
        "scaled_temporal_secant_v1",
        {"lr": 0.005, "perturb": 0.1},
        aliases=("SPSA",),
        update_schedule="one-new-value-per-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "GaussianTemporalSecant",
        GaussianTemporalSecant,
        "zero-order",
        "gaussian_temporal_secant_v1",
        {"lr": 0.005, "mu": 0.01},
        aliases=("ZOSGD",),
        update_schedule="one-new-value-per-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "SignedGaussianTemporalSecant",
        SignedGaussianTemporalSecant,
        "zero-order",
        "signed_gaussian_temporal_secant_v1",
        {"lr": 0.005, "mu": 0.01},
        aliases=("ZOSignSGD",),
        update_schedule="one-new-value-per-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "QuadraticInterpolation",
        QuadraticInterpolation,
        "zero-order",
        "directional_quadratic_interpolation_v1",
        {"lr": 0.1},
        update_schedule="three-outer-step-directional-cycle",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "ShrinkingTemporalSecant",
        ShrinkingTemporalSecant,
        "zero-order",
        "shrinking_temporal_secant_v1",
        {"lr": 0.005, "cn": 0.1},
        aliases=("KieferWolfowitz",),
        update_schedule="one-new-value-per-shrinking-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "DiminishingGaussianSecant",
        DiminishingGaussianSecant,
        "zero-order",
        "diminishing_gaussian_secant_v1",
        {"lr": 0.005},
        aliases=("NedicSubgradient",),
        update_schedule="one-new-value-per-diminishing-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "MomentumTemporalSecant",
        MomentumTemporalSecant,
        "zero-order",
        "momentum_temporal_secant_v1",
        {"lr": 0.005, "perturb": 0.1, "beta": 0.9},
        aliases=("AcceleratedSPSA",),
        update_schedule="one-new-value-per-momentum-temporal-secant-step",
        memory_complexity="O(d)",
    ),
    OptimizerSpec(
        "EliteCovarianceSearch",
        EliteCovarianceSearch,
        "zero-order",
        "elite_covariance_search_v1",
        {"sigma": 0.5},
        aliases=("CMAES",),
        update_schedule="one-candidate-per-outer-step-population-cycle",
        memory_complexity="O(d^2)",
    ),
    OptimizerSpec(
        "DistanceScaledExploration",
        DistanceScaledExploration,
        "zero-order",
        "distance_scaled_exploration_v1",
        {"beta": 2.0},
        aliases=("GPUCB",),
        update_schedule="one-value-per-random-exploration-step",
        memory_complexity="O(td)",
    ),
)

OPTIMIZERS_BY_NAME = {spec.canonical_name: spec for spec in OPTIMIZER_SPECS}
OPTIMIZER_ALIASES = {
    alias: spec.canonical_name for spec in OPTIMIZER_SPECS for alias in spec.aliases
}


def attach_optimizer_metadata(
    optimizer: OptimizerProtocol,
    spec: OptimizerSpec,
    *,
    requested_name: Optional[str] = None,
) -> OptimizerProtocol:
    """Attach explicit reproducibility metadata without changing the update rule."""
    requested = requested_name or spec.canonical_name
    optimizer.name = spec.canonical_name
    optimizer.canonical_name = spec.canonical_name
    optimizer.requested_alias = requested if requested != spec.canonical_name else None
    optimizer.implementation_id = spec.implementation_id
    optimizer.oracle_type = spec.oracle_type
    optimizer.query_cost_per_step = spec.query_cost_per_step
    optimizer.update_schedule = spec.update_schedule
    optimizer.memory_complexity = spec.memory_complexity
    return optimizer


# Make capability metadata available even when a canonical class is instantiated
# directly rather than through ``create_optimizer``.
for _spec in OPTIMIZER_SPECS:
    _spec.optimizer_class.canonical_name = _spec.canonical_name
    _spec.optimizer_class.implementation_id = _spec.implementation_id
    _spec.optimizer_class.oracle_type = _spec.oracle_type
    _spec.optimizer_class.query_cost_per_step = _spec.query_cost_per_step
    _spec.optimizer_class.update_schedule = _spec.update_schedule
    _spec.optimizer_class.memory_complexity = _spec.memory_complexity


def canonical_optimizer_names() -> tuple[str, ...]:
    return tuple(spec.canonical_name for spec in OPTIMIZER_SPECS)


def accepted_optimizer_names() -> frozenset[str]:
    return frozenset((*OPTIMIZERS_BY_NAME, *OPTIMIZER_ALIASES))


def resolve_optimizer_spec(name: str) -> OptimizerSpec:
    canonical_name = OPTIMIZER_ALIASES.get(name, name)
    try:
        return OPTIMIZERS_BY_NAME[canonical_name]
    except KeyError as exc:
        raise KeyError(f"Unknown optimizer: {name}") from exc


def create_optimizer(
    name: str,
    parameters: Optional[Mapping[str, Any]] = None,
    *,
    warn_alias: bool = True,
) -> OptimizerProtocol:
    spec = resolve_optimizer_spec(name)
    return spec.instantiate(parameters, requested_name=name, warn_alias=warn_alias)
