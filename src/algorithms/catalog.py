"""The 22 source-specified algorithms in catalog_v2. No alias resolution."""

from dataclasses import dataclass
from typing import Any, Mapping
import inspect
from wind_benchmark.algorithms import optimizers as optimizers

CATALOG_VERSION = "catalog_v2"

# Immutable identities: these strings are opaque tokens, NOT accepted aliases.
_IDENTITIES = {
    "SGD": ("wind060.SGD.v1", 1, "wind060.SGD.v1"),
    "PolyakRuppertSGD": (
        "wind060.PolyakRuppertSGD.v1",
        1,
        "wind060.PolyakRuppertSGD.v1",
    ),
    "HeavyBall": ("wind060.HeavyBall.v1", 1, "wind060.HeavyBall.v1"),
    "Nesterov": ("wind060.Nesterov.v1", 1, "wind060.Nesterov.v1"),
    "Adam": ("wind060.Adam.v1", 1, "wind060.Adam.v1"),
    "AdamW": ("wind060.AdamW.v1", 1, "wind060.AdamW.v1"),
    "AMSGrad": ("wind060.AMSGrad.v1", 1, "wind060.AMSGrad.v1"),
    "SignSGD": ("wind060.SignSGD.v1", 1, "wind060.SignSGD.v1"),
    "ProxSGD": ("wind060.ProxSGD.v1", 1, "wind060.ProxSGD.v1"),
    "RDA": ("wind060.RDA.v1", 1, "wind060.RDA.v1"),
    "SPSA": ("wind.SPSA.normalized_tracking.v2", 2, "wind.SPSA.normalized_tracking.v2"),
    "KieferWolfowitz": (
        "wind060.CoordinateFiniteDifferenceSGD.v1",
        1,
        "wind060.CoordinateFiniteDifferenceSGD.v1",
    ),
    "GaussianZOSGD": ("wind060.GaussianZOSGD.v1", 1, "wind060.GaussianZOSGD.v1"),
    "SphericalZOSGD": ("wind060.SphericalZOSGD.v1", 1, "wind060.SphericalZOSGD.v1"),
    "ExponentiatedGradient": (
        "wind060.ExponentiatedGradient.v1",
        1,
        "wind060.ExponentiatedGradient.v1",
    ),
    "ProjectedGradient": (
        "wind060.ProjectedGradient.v1",
        1,
        "wind060.ProjectedGradient.v1",
    ),
    "RiemannianSGD": ("wind060.RiemannianSGD.v1", 1, "wind060.RiemannianSGD.v1"),
    "AdaGrad": ("wind.AdaGrad.v1", 1, "wind.AdaGrad.v1"),
    "OnePointSPSA": ("wind.OnePointSPSA.v1", 1, "wind.OnePointSPSA.v1"),
    "ZOSignSGD": (
        "wind.ZOSignSGD.sphere_forward.v1",
        1,
        "wind.ZOSignSGD.sphere_forward.v1",
    ),
    "RandomSearch": (
        "wind.RandomSearch.fresh_comparison.v1",
        1,
        "wind.RandomSearch.fresh_comparison.v1",
    ),
    "AcceleratedSPSA": (
        "wind.AcceleratedSPSA.constant.v1",
        1,
        "wind.AcceleratedSPSA.constant.v1",
    ),
}


@dataclass(frozen=True)
class OptimizerSpec:
    canonical_name: str
    optimizer_class: type
    source: str
    variant: str

    @property
    def default_parameters(self):
        return {
            name: parameter.default
            for name, parameter in inspect.signature(
                self.optimizer_class
            ).parameters.items()
        }

    @property
    def implementation_id(self):
        return _IDENTITIES[self.canonical_name][0]

    @property
    def implementation_version(self):
        return _IDENTITIES[self.canonical_name][1]

    @property
    def rng_stream_id(self):
        return _IDENTITIES[self.canonical_name][2]

    @property
    def oracle_type(self):
        return self.optimizer_class.oracle_type

    @property
    def query_cost_per_step(self):
        return self.optimizer_class.query_cost_per_step

    @property
    def update_schedule(self):
        return self.optimizer_class.update_schedule

    @property
    def memory_complexity(self):
        return self.optimizer_class.memory_complexity

    @property
    def geometries(self):
        return self.optimizer_class.geometries

    def instantiate(self, parameters: Mapping[str, Any] | None = None):
        return self.optimizer_class(**dict(parameters or {}))


_SOURCES = (
    (
        "SGD",
        "https://doi.org/10.1214/aoms/1177729586",
        "constant-step stochastic gradient",
    ),
    (
        "PolyakRuppertSGD",
        "https://doi.org/10.1137/0330046",
        "raw SGD lr*t^-0.75; mean of updated iterates; no burn-in",
    ),
    (
        "HeavyBall",
        "https://doi.org/10.1016/0041-5553(64)90137-5",
        "constant momentum; zero initial velocity",
    ),
    (
        "Nesterov",
        "https://proceedings.mlr.press/v28/sutskever13.pdf",
        "constant-momentum lookahead, equations 3-4",
    ),
    ("Adam", "https://arxiv.org/abs/1412.6980", "Algorithm 1; both bias corrections"),
    (
        "AdamW",
        "https://arxiv.org/abs/1711.05101",
        "Algorithm 2; decoupled decay; constant schedule",
    ),
    (
        "AMSGrad",
        "https://arxiv.org/abs/1904.09237",
        "Algorithm 2 raw moments; constant lr; epsilon outside sqrt",
    ),
    (
        "SignSGD",
        "https://proceedings.mlr.press/v80/bernstein18a.html",
        "single-worker sign of supplied stochastic gradient",
    ),
    (
        "ProxSGD",
        "https://doi.org/10.1137/080716542",
        "Euclidean forward-backward step with exact L1 prox",
    ),
    (
        "RDA",
        "https://www.jmlr.org/papers/v11/xiao10a.html",
        "Algorithm 1; quadratic prox; beta_t=sqrt(t)/lr; x0=0; L1",
    ),
    (
        "SPSA",
        "https://lib.physcon.ru/file?id=f7df197d2483",
        "Eq.7 SPSA component; normalized Bernoulli; no d factor; minus-plus; one pair",
    ),
    (
        "KieferWolfowitz",
        "https://doi.org/10.1214/aoms/1177729392",
        "central coordinate differences; constant lr/h; 2d queries",
    ),
    (
        "GaussianZOSGD",
        "https://doi.org/10.1007/s10208-015-9296-2",
        "symmetric Gaussian smoothing estimator; m-pair mean; no d multiplier",
    ),
    (
        "SphericalZOSGD",
        "https://www.jmlr.org/papers/v18/16-632.html",
        "symmetric sphere estimator; d multiplier; m independent pairs",
    ),
    (
        "ExponentiatedGradient",
        "https://www.tau.ac.il/~becka/3.pdf",
        "entropic mirror descent; invariant zero support; persistent log-space weights",
    ),
    (
        "ProjectedGradient",
        "https://icml.cc/Conferences/2008/papers/361.pdf",
        "Euclidean gradient step and exact simplex projection",
    ),
    (
        "RiemannianSGD",
        "https://sites.uclouvain.be/absil/amsbook/",
        "embedded Euclidean tangent projection and polar retraction",
    ),
    (
        "AdaGrad",
        "https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf",
        "diagonal accumulated squares; epsilon outside sqrt; no forgetting",
    ),
    (
        "OnePointSPSA",
        "https://www.jhuapl.edu/SPSA/PDF-SPSA/automatica97_one_measSPSA.pdf",
        "one fresh measurement; unnormalized Bernoulli; no baseline",
    ),
    (
        "ZOSignSGD",
        "https://openreview.net/references/pdf?id=SJJUQwctQ",
        "Eq.3 sphere forward differences; fresh shared baseline; sign after mean",
    ),
    (
        "RandomSearch",
        "https://www.mathnet.ru/eng/at11288",
        "fresh-comparison tracking adaptation; Gaussian sigma/sqrt(d); randomized query order",
    ),
    (
        "AcceleratedSPSA",
        "https://lib.physcon.ru/file?id=f7df197d2483",
        "Eq.8 structure with constant coefficients; normalized Bernoulli; no theorem claim",
    ),
)

OPTIMIZER_SPECS = tuple(
    OptimizerSpec(name, getattr(optimizers, name), source, variant)
    for name, source, variant in _SOURCES
)
OPTIMIZERS_BY_NAME = {spec.canonical_name: spec for spec in OPTIMIZER_SPECS}
for _spec in OPTIMIZER_SPECS:
    _cls = _spec.optimizer_class
    _cls.canonical_name = _spec.canonical_name
    _cls.implementation_id = _spec.implementation_id
    _cls.implementation_version = _spec.implementation_version
    _cls.rng_stream_id = _spec.rng_stream_id
    _cls.source = _spec.source
    _cls.variant = _spec.variant


def canonical_optimizer_names():
    return tuple(OPTIMIZERS_BY_NAME)


def accepted_optimizer_names():
    return frozenset(OPTIMIZERS_BY_NAME)


def resolve_optimizer_spec(name):
    try:
        return OPTIMIZERS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown algorithm {name!r}. WIND 0.7 accepts only current identifiers: "
            + ", ".join(OPTIMIZERS_BY_NAME)
        ) from exc


def create_optimizer(name, parameters=None):
    return resolve_optimizer_spec(name).instantiate(parameters)
