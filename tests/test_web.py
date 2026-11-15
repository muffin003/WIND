import pytest

from wind_benchmark.interfaces.web import (
    ConfigurationError,
    _resolve_roots,
    capability_catalog,
    resolve_output_dir,
    validate_config,
)
from wind_benchmark.interfaces.web_runner import (
    WorkbenchConfigurationError,
    run_workbench,
    validate_workbench_config,
)


def test_source_assets_and_outputs_keep_the_repository_root(tmp_path):
    module = tmp_path / "checkout" / "src" / "interfaces" / "web.py"
    root, assets = _resolve_roots(module, tmp_path / "other")
    assert root == tmp_path / "checkout"
    assert assets == root / "web"


def test_installed_assets_do_not_send_outputs_to_site_packages(tmp_path):
    package = tmp_path / "site-packages" / "wind_benchmark"
    (package / "_web").mkdir(parents=True)
    work = tmp_path / "work"
    root, assets = _resolve_roots(package / "interfaces" / "web.py", work)
    assert root == work
    assert assets == package / "_web"


def valid_config():
    return {
        "schema_version": 3,
        "output_dir": "results/web_test",
        "seeds": [42, 43],
        "steps": 20,
        "rho_values": [1.0, 0.5],
        "drift_values": [0.01, 0.1],
        "dimensions": [5],
        "optimizers": ["SGD", "SPSA"],
    }


def test_validate_config_accepts_cli_schema():
    assert validate_config(valid_config()) == valid_config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("steps", 0),
        ("seeds", [-1]),
        ("rho_values", [0.7]),
        ("drift_values", [0.0]),
        ("dimensions", [2.5]),
        ("optimizers", ["UnknownMethod"]),
    ],
)
def test_validate_config_rejects_invalid_values(field, value):
    config = valid_config()
    config[field] = value
    with pytest.raises(ConfigurationError):
        validate_config(config)


def test_output_directory_cannot_escape_project():
    with pytest.raises(ConfigurationError):
        resolve_output_dir("../outside")


def workbench_config():
    return {
        "schema_version": 3,
        "environment": {
            "dim": 5,
            "x_bounds": [-10, 10],
            "landscape": {"type": "quadratic", "condition_number": 5},
            "drift": {"type": "random_walk", "sigma": 0.02},
        },
        "oracle": {
            "type": "first-order",
            "blind_value": True,
            "value_noise": {"type": "gaussian", "sigma": 0.01},
            "grad_noise": {"type": "gaussian", "sigma": 0.01},
        },
        "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
        "runner": {
            "steps": 20,
            "seeds": [42],
            "tail_fraction": 0.2,
            "output_dir": "results/workbench_test",
        },
        "metrics": ["tracking_error", "dynamic_regret"],
    }


def test_web_rda_requires_explicit_shared_zero_start_and_runs(tmp_path):
    config = workbench_config()
    config["optimizer"] = {"name": "RDA", "params": {}}
    with pytest.raises(WorkbenchConfigurationError, match="runner.initial_x"):
        validate_workbench_config(config)
    config["runner"].update(initial_x=[0.0] * 5, steps=3)
    assert validate_workbench_config(config) == config
    result = run_workbench(config, tmp_path)
    assert result["runs"][0]["optimizer"] == "RDA"


def test_validate_workbench_config_accepts_general_benchmark():
    config = workbench_config()
    assert validate_workbench_config(config) is config


def test_validate_workbench_config_accepts_empty_optimizer_draft():
    config = workbench_config()
    config.pop("optimizer")
    config["optimizers"] = []
    assert validate_workbench_config(config) is config


def test_validate_workbench_config_rejects_unknown_landscape():
    config = workbench_config()
    config["environment"]["landscape"]["type"] = "unknown"
    with pytest.raises(WorkbenchConfigurationError):
        validate_workbench_config(config)


def test_validate_workbench_config_rejects_incompatible_feedback():
    config = workbench_config()
    config["oracle"]["type"] = "zero-order"
    with pytest.raises(WorkbenchConfigurationError):
        validate_workbench_config(config)


def test_validate_workbench_config_rejects_csv_without_trajectory():
    config = workbench_config()
    config["runner"].update(export_csv=True, record_trajectory=False)
    with pytest.raises(WorkbenchConfigurationError):
        validate_workbench_config(config)


def test_capability_catalog_exposes_full_workbench():
    catalog = capability_catalog()
    assert len(catalog["landscapes"]) == 8
    assert len(catalog["drifts"]) == 8
    assert len(catalog["noises"]) == 7
    assert len(catalog["oracles"]) == 5
    assert len(catalog["optimizers"]) == 22
    optimizer_names = {item["name"] for item in catalog["optimizers"]}
    assert "CoupledL2Adam" not in optimizer_names
    assert "PolyakRuppertSGD" in optimizer_names
    assert "AdamW" in optimizer_names
    assert "Nesterov" in optimizer_names


def test_workbench_reports_correct_adamw_without_alias(tmp_path):
    config = workbench_config()
    config["optimizer"] = {"name": "AdamW", "params": {"lr": 0.001}}
    config["runner"]["steps"] = 2
    config["runner"]["output_dir"] = "results"

    summary = run_workbench(config, tmp_path)

    assert summary["optimizer"] == "AdamW"
    assert "requested_aliases" not in summary
    assert "requested_alias" not in summary["runs"][0]
    assert summary["runs"][0]["implementation_id"] == "wind060.AdamW.v1"
    assert (tmp_path / "results" / "AdamW_seed42.json").is_file()


def test_workbench_runner_executes_general_configuration(tmp_path):
    config = workbench_config()
    config["runner"]["steps"] = 2
    config["runner"]["output_dir"] = "results"
    config["runner"]["export_csv"] = True

    summary = run_workbench(config, tmp_path)

    assert summary["status"] == "SUCCESS"
    assert summary["optimizer"] == "SGD"
    assert len(summary["runs"]) == 1
    assert summary["runs"][0]["status"] == "SUCCESS"
    assert (tmp_path / "results" / "SGD_seed42.json").is_file()
    assert (tmp_path / "results" / "SGD_seed42.csv").is_file()
    assert (tmp_path / "results" / "workbench_summary.json").is_file()


def test_workbench_runner_executes_multiple_optimizers(tmp_path):
    config = workbench_config()
    config.pop("optimizer")
    config["optimizers"] = [
        {"name": "SGD", "params": {"lr": 0.1}},
        {"name": "Adam", "params": {"lr": 0.001}},
    ]
    config["runner"]["steps"] = 2
    config["runner"]["output_dir"] = "results"

    summary = run_workbench(config, tmp_path)

    assert summary["optimizers"] == ["SGD", "Adam"]
    assert len(summary["runs"]) == 2
    assert {run["optimizer"] for run in summary["runs"]} == {"SGD", "Adam"}
    assert (tmp_path / "results" / "SGD_seed42.json").is_file()
    assert (tmp_path / "results" / "Adam_seed42.json").is_file()


def test_workbench_runner_rejects_empty_optimizer_selection(tmp_path):
    config = workbench_config()
    config.pop("optimizer")
    config["optimizers"] = []
    with pytest.raises(WorkbenchConfigurationError, match="at least one"):
        run_workbench(config, tmp_path)


def test_workbench_runner_wires_extended_coefficients(tmp_path):
    config = workbench_config()
    config["environment"]["drift"] = {
        "type": "adaptive",
        "alpha": 0.04,
        "threshold": 3.0,
        "mode": "evasion",
    }
    config["environment"]["initial_theta"] = [0.2] * 5
    config["oracle"]["value_noise"] = {
        "type": "heavy_tailed",
        "alpha": 1.7,
        "scale": 0.02,
    }
    config["oracle"]["grad_noise"] = {
        "type": "correlated",
        "sigma": 0.01,
        "phi": 0.6,
    }
    config["optimizer"] = {
        "name": "Adam",
        "params": {"lr": 0.001, "beta1": 0.8, "beta2": 0.95, "eps": 1e-7},
    }
    config["runner"].update(
        steps=2,
        output_dir="results",
        tracking_norm="mahalanobis",
        normalize_tracking=True,
        jump_threshold=0.5,
        recovery_epsilon=0.2,
        oracle_ttr=2.0,
        rho=0.7,
        initial_x=[-0.1] * 5,
    )
    config["metrics"] = [
        "tracking_error",
        "time_to_recovery",
        "adaptivity",
        "lyapunov",
        "asymptotic_bound",
    ]

    summary = run_workbench(config, tmp_path)

    assert summary["status"] == "SUCCESS"
    assert (tmp_path / "results" / "Adam_seed42.json").is_file()
