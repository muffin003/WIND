"""Evaluator-only, JSON-safe construction snapshots for built-in components.

No pickle or arbitrary import is used when restoring untrusted result files.
Custom components are explicitly marked non-restorable rather than pretending
that an incomplete signature can reproduce them.
"""

from copy import deepcopy
import inspect
import math
import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"array", "dtype"}:
            return np.array(value["array"], dtype=value["dtype"])
        if set(value) == {"nonfinite"}:
            return float(value["nonfinite"])
        return {k: decode(v) for k, v in value.items()}
    return value


def component_snapshot(component):
    if component is None:
        return None
    from wind_benchmark.core import environment as core

    cls = type(component)
    supported = (
        cls.__module__ == core.__name__ and getattr(core, cls.__name__, None) is cls
    )
    params, missing = {}, []
    aliases = {"condition_number": "kappa", "seed": "_seed", "path": "_path"}
    for name, parameter in inspect.signature(cls).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        attr = name if hasattr(component, name) else aliases.get(name, name)
        if hasattr(component, attr):
            params[name] = deepcopy(getattr(component, attr))
        elif name == "seed":
            params[name] = getattr(component, "_resolved_seed", 0)
        elif parameter.default is not parameter.empty:
            params[name] = parameter.default
        else:
            missing.append(name)
    state = {}
    for name in ("A", "_M", "_rng_state"):
        if hasattr(component, name):
            state[name] = deepcopy(getattr(component, name))
    return encode(
        {
            "type": cls.__name__,
            "parameters": params,
            "state": state,
            "restorable": supported and not missing,
            "missing": missing,
        }
    )


def restore_component(snapshot):
    if snapshot is None:
        return None
    from wind_benchmark.core import environment as core

    data = decode(snapshot)
    allowed = {
        cls.__name__: cls
        for registry in (
            core._DRIFT_REGISTRY,
            core._LANDSCAPE_REGISTRY,
            core._NOISE_REGISTRY,
        )
        for cls in registry.values()
    }
    allowed["PrescribedDrift"] = core.PrescribedDrift
    if not data["restorable"] or data["type"] not in allowed:
        raise ValueError(f"No built-in reconstruction for {data['type']}")
    component = allowed[data["type"]](**data["parameters"])
    if set(data["state"]) - {"A", "_M", "_rng_state"}:
        raise ValueError("Unsupported component state fields")
    for name, value in data["state"].items():
        setattr(component, name, value)
    if hasattr(component, "reset"):
        component.reset()
    return component


def environment_snapshot(env):
    # Capture a lazily realized, possibly unseeded P-norm orientation once.
    # Otherwise an uninitialized matrix would be redrawn when restoring.
    from wind_benchmark.core.environment import PNormLandscape

    if type(env.landscape) is PNormLandscape:
        env.landscape._lazy_M(env.dim)
    oracle = getattr(env, "_registered_oracle", None)
    oracle_data = None
    if oracle is not None:
        oracle_data = {
            "type": type(oracle).__name__,
            "value_noise": component_snapshot(oracle.value_noise),
            "grad_noise": component_snapshot(oracle.grad_noise),
            "seed": getattr(oracle, "_resolved_seed", 0),
            "parameters": {},
        }
        for name in ("blind_value", "schedule", "recorded_thetas"):
            if hasattr(oracle, name):
                oracle_data["parameters"][name] = encode(getattr(oracle, name))
        if hasattr(oracle, "landscape"):
            oracle_data["landscape"] = component_snapshot(oracle.landscape)
    return {
        "version": "environment_snapshot_v2",
        "dim": env.dim,
        "bounds": encode(env.bounds),
        "initial_theta": encode(env._initial_theta),
        "drift": component_snapshot(env.drift),
        "landscape": component_snapshot(env.landscape),
        "oracle": oracle_data,
    }


def snapshot_restorable(snapshot):
    if not snapshot:
        return False
    components = [snapshot["drift"], snapshot["landscape"]]
    oracle = snapshot.get("oracle")
    if oracle:
        if oracle["type"] not in {
            "FirstOrderOracle",
            "ZeroOrderOracle",
            "HybridOracle",
            "ScheduledOracle",
            "OfflineOracle",
        }:
            return False
        components.extend(
            [oracle["value_noise"], oracle["grad_noise"], oracle.get("landscape")]
        )
    return all(item is None or item["restorable"] for item in components)


def restore_environment(snapshot):
    from wind_benchmark.core.environment import DynamicEnvironment
    from wind_benchmark.core import oracle as module

    if snapshot["version"] != "environment_snapshot_v2":
        raise ValueError("Unsupported environment snapshot version")
    env = DynamicEnvironment(
        snapshot["dim"],
        restore_component(snapshot["drift"]),
        restore_component(snapshot["landscape"]),
        decode(snapshot["initial_theta"]),
        decode(snapshot["bounds"]),
    )
    data = snapshot["oracle"]
    if data is None:
        return env, None
    allowed = {
        name: getattr(module, name)
        for name in (
            "FirstOrderOracle",
            "ZeroOrderOracle",
            "HybridOracle",
            "ScheduledOracle",
            "OfflineOracle",
        )
    }
    if data["type"] not in allowed:
        raise ValueError("Custom oracle needs an explicit reconstruction factory")
    kwargs = decode(data["parameters"])
    if "landscape" in data:
        kwargs["landscape"] = restore_component(data["landscape"])
    oracle = allowed[data["type"]](
        env,
        value_noise=restore_component(data["value_noise"]),
        grad_noise=restore_component(data["grad_noise"]),
        seed=data["seed"],
        **kwargs,
    )
    return env, oracle
