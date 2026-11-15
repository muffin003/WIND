import pytest
from wind_benchmark.algorithms.protocol import TaskSpec
from wind_benchmark.core.runner import OptimizerInfo
from wind_benchmark.algorithms.catalog import (
    OPTIMIZER_SPECS,
    accepted_optimizer_names,
    canonical_optimizer_names,
    create_optimizer,
    resolve_optimizer_spec,
)


def test_catalog_has_only_current_identifiers():
    names = canonical_optimizer_names()
    assert len(names) == len(set(names)) == 22
    assert accepted_optimizer_names() == frozenset(names)
    for retired in (
        "SGD_Polyak",
        "LookaheadNesterov",
        "CoupledL2Adam",
        "ScaledTemporalSecant",
        "CMAES",
        "GPUCB",
        "SMD",
    ):
        with pytest.raises(KeyError):
            resolve_optimizer_spec(retired)


@pytest.mark.parametrize("spec", OPTIMIZER_SPECS, ids=lambda s: s.canonical_name)
def test_catalog_metadata_identifies_formula_not_alias(spec):
    algorithm = spec.instantiate()
    info = OptimizerInfo.from_optimizer(algorithm)
    assert info.name == spec.canonical_name
    assert info.hyperparameters == spec.default_parameters
    assert info.source.startswith("https://")
    assert info.variant and info.implementation_version == spec.implementation_version
    assert "alias" not in str(info.to_dict())
    assert not hasattr(algorithm, "step")
    assert not hasattr(algorithm, "requested_alias")


def test_standard_names_are_distinct_real_implementations():
    for name in ("Nesterov", "AdamW", "SPSA"):
        assert type(create_optimizer(name)).__name__ == name
    assert create_optimizer("HeavyBall").parameters == {"lr": 0.1, "beta": 0.9}
    assert create_optimizer("Nesterov").parameters == {"lr": 0.05, "beta": 0.9}
