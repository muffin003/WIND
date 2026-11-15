"""Process scheduling must not change the experiment or its information barrier."""

from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import signal
import subprocess
import sys
import sysconfig
import time

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from wind_benchmark.experiments.canonical import validate_result
from wind_benchmark.experiments.parallel import (
    Execution,
    WorkerError,
    execution_options,
)
from wind_benchmark.experiments.standard import (
    Method,
    Profile,
    Task,
    catalog_method,
    default_tasks,
    main,
    run_suite,
)
from wind_benchmark.core.privacy import EvaluationSession, GroundTruthAccessError
from wind_benchmark.core.resources import resident_bytes
from parallel_fixtures import (
    crashed_factory,
    inspected_factory,
    ordinary_factory,
    delayed_factory,
    exception_factory,
    initialization_error_factory,
)

PROFILE = Profile("parallel-test", 41, (20001, 20002), 100, 5)
RESOURCE = {
    "runtime_seconds",
    "rss_baseline_bytes",
    "sampled_peak_rss_bytes",
    "sampled_rss_increase_bytes",
}


def test_method_order_and_completion_order_do_not_change_scores(tmp_path):
    method = Method(
        "Custom",
        "first-order",
        ("euclidean",),
        ordinary_factory,
        rng_key="fixed-custom",
    )
    methods = [catalog_method("SGD"), catalog_method("SPSA"), method]
    a = run_suite(
        tmp_path / "a",
        PROFILE,
        methods,
        default_tasks()[:2],
        workers=1,
        plots=False,
        progress=False,
    )
    b = run_suite(
        tmp_path / "b",
        PROFILE,
        [replace(method, factory=delayed_factory), *methods[:2][::-1]],
        default_tasks()[:2],
        workers=4,
        plots=False,
        progress=False,
    )
    for key in ("runs", "summary", "paired"):
        columns = ["task", "algorithm"] + (["seed"] if key == "runs" else [])
        left, right = (
            r[key]
            .drop(columns=list(RESOURCE), errors="ignore")
            .sort_values(columns)
            .reset_index(drop=True)
            for r in (a, b)
        )
        pd.testing.assert_frame_equal(left, right, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("entry", ["module", "file", "console"])
def test_cli_spawn_bootstrap(entry, tmp_path):
    root = Path(__file__).resolve().parents[1]
    if entry == "module":
        command = [sys.executable, "-m", "wind_benchmark.experiments.standard"]
    elif entry == "file":
        command = [sys.executable, str(root / "src/experiments/standard.py")]
    else:
        command = [
            str(
                # Base Windows installs keep console scripts in Scripts/,
                # while python.exe lives one directory above it.
                Path(sysconfig.get_path("scripts"))
                / ("wind-canonical.exe" if os.name == "nt" else "wind-canonical")
            )
        ]
    output = tmp_path / entry
    result = subprocess.run(
        [
            *command,
            "--profile",
            "smoke",
            "--workers",
            "2",
            "--algorithms",
            "ProjectedGradient",
            "RiemannianSGD",
            "--tasks",
            "simplex_moving_gaussian",
            "grassmann_moving_correlated",
            "--output-dir",
            str(output),
            "--no-plots",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_status(output)["status"] == "complete"


def test_cli_console_does_not_assume_python_and_scripts_share_directory(
    monkeypatch, tmp_path
):
    # Reproduce the installation-layout distinction even inside a local venv.
    monkeypatch.setattr(sys, "executable", str(tmp_path / "base-python" / "python.exe"))
    test_cli_spawn_bootstrap("console", tmp_path)


def test_series_rss_is_peak_of_concurrent_sums(monkeypatch):
    import wind_benchmark.core.resources as resources

    phase = [0]

    def rss(pid=None):
        if pid is None:
            phase[0] += 1
            return 10
        return 100 if pid == (1 if phase[0] % 2 else 2) else 0

    monkeypatch.setattr(resources, "resident_bytes", rss)
    sampler = resources.SeriesResourceSampler()
    sampler.add(1)
    sampler.add(2)
    time.sleep(0.08)
    assert sampler.finish() == 110


def read_status(output):
    return json.loads((output / "execution_status.json").read_text())


def compare_outputs(left, right):
    for filename in (
        "runs.csv",
        "summary.csv",
        "paired_comparisons.csv",
        "metrics_summary.csv",
        "curves.csv",
        "recovery_events.csv",
        "trace_index.csv",
    ):
        a, b = (pd.read_csv(p / filename) for p in (left, right))
        if filename == "metrics_summary.csv":
            a, b = (f[~f.metric.isin(RESOURCE)].reset_index(drop=True) for f in (a, b))
        a, b = (f.drop(columns=list(RESOURCE), errors="ignore") for f in (a, b))
        pd.testing.assert_frame_equal(a, b, atol=1e-12, rtol=1e-10)
    for filename in [
        "trajectories.npz",
        *(str(p.relative_to(left)) for p in (left / "recommendations").glob("*.npz")),
    ]:
        with (
            np.load(left / filename, allow_pickle=False) as a,
            np.load(right / filename, allow_pickle=False) as b,
        ):
            assert set(a.files) == set(b.files)
            for key in a.files:
                if a[key].dtype.kind in "US":
                    np.testing.assert_array_equal(a[key], b[key])
                else:
                    np.testing.assert_allclose(a[key], b[key], atol=1e-12, rtol=1e-10)
    a, b = (json.loads((p / "manifest.json").read_text()) for p in (left, right))
    assert a["resolved_runs"] == b["resolved_runs"]
    assert a["specification_sha256"] == b["specification_sha256"]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_bad_workers_do_not_create_files(tmp_path, workers):
    output = tmp_path / "no"
    with pytest.raises(ValueError, match="positive integer"):
        run_suite(output, PROFILE, workers=workers, plots=False)
    assert not output.exists()


def test_worker_options_and_dry_run(tmp_path, capsys):
    assert execution_options(4, 0)["workers_effective"] == 0
    assert execution_options(4, 2)["workers_effective"] == 2
    with pytest.warns(UserWarning, match="exceeds"):
        execution_options((os.cpu_count() or 1) + 1, 100000)
    output = tmp_path / "none"
    assert main(["--dry-run", "--workers", "4", "--output-dir", str(output)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["execution"]["workers_effective"] == 4
    assert data["compatible_runs"] == 22450
    assert not output.exists()


def test_lambda_rejected_only_for_parallel(tmp_path):
    method = Method(
        "custom", "first-order", ("euclidean",), lambda t, s: ordinary_factory(t, s)
    )
    with pytest.raises(ValueError, match="top-level factory"):
        run_suite(tmp_path / "invalid", PROFILE, [method], workers=2, plots=False)
    assert not (tmp_path / "invalid").exists()
    result = run_suite(
        tmp_path / "serial",
        PROFILE,
        [method],
        [Task("clean", "reference")],
        plots=False,
    )
    assert set(result["runs"].status) == {"SUCCESS"}


def test_all_methods_serial_and_spawn_identical(tmp_path):
    # Covers every method, geometry and partial coordinate batch without a full run.
    tasks = [
        default_tasks()[0],
        default_tasks()[1],
        *(t for t in default_tasks() if t.name.endswith("moving_correlated")),
    ]
    for workers in (1, 2, 4):
        output = tmp_path / str(workers)
        before = pickle.dumps((np.random.get_state(), random.getstate()))
        pools = threadpool_info()
        result = run_suite(
            output, PROFILE, tasks=tasks, workers=workers, plots=False, progress=False
        )
        assert pickle.dumps((np.random.get_state(), random.getstate())) == before
        assert threadpool_info() == pools
        assert read_status(output)["status"] == "complete"
        assert not set(result["runs"].status) & {"ERROR", "ERROR_INTERFACE"}
        assert result["manifest"]["execution"]["workers_requested"] == workers
        if workers != 1:
            compare_outputs(tmp_path / "1", output)
    assert not mp.active_children()


def test_outer_session_holds_results_and_abort(tmp_path):
    for abort in (False, True):
        output = tmp_path / str(abort)
        try:
            with EvaluationSession(expected_runs=2):
                pending = run_suite(
                    output,
                    PROFILE,
                    [catalog_method("SGD")],
                    [Task("clean", "reference")],
                    workers=2,
                    plots=False,
                    progress=False,
                )
                assert read_status(output)["status"] == "awaiting_release"
                assert not (output / "runs.csv").exists()
                with pytest.raises(GroundTruthAccessError):
                    pending.result()
                if abort:
                    raise RuntimeError("test abort")
        except RuntimeError:
            assert abort
        if abort:
            assert read_status(output)["status"] == "failed"
            assert not (output / "manifest.json").exists()
            with pytest.raises(GroundTruthAccessError):
                pending.result()
        else:
            assert read_status(output)["status"] == "complete"
            assert len(pending.result()["runs"]) == 2


def test_reused_workers_keep_retained_objects_sealed(tmp_path):
    method = Method(
        "Inspect", "first-order", ("euclidean",), inspected_factory, rng_key="inspect"
    )
    profile = replace(PROFILE, seeds=tuple(range(20001, 20009)))
    result = run_suite(
        tmp_path / "guard",
        profile,
        [method],
        [Task("clean", "reference")],
        workers=2,
        plots=False,
        progress=False,
    )
    assert set(result["runs"].status) == {"SUCCESS"}, result["runs"].reason.tolist()


@pytest.mark.parametrize(
    "factory,status",
    [(exception_factory, "ERROR"), (initialization_error_factory, "ERROR_INTERFACE")],
)
def test_user_exception_formatter_has_no_evaluator_authority(tmp_path, factory, status):
    method = Method(
        "Exception",
        "first-order",
        ("euclidean",),
        factory,
        rng_key="exception",
    )
    result = run_suite(
        tmp_path / "exception",
        replace(PROFILE, seeds=tuple(range(20001, 20007))),
        [method],
        [Task("clean", "reference")],
        workers=2,
        plots=False,
        progress=False,
    )
    assert set(result["runs"].status) == {status}
    assert all("remained unprivileged" in value for value in result["runs"].reason)


def test_worker_crash_aborts_without_export(tmp_path):
    method = Method(
        "Crash", "first-order", ("euclidean",), crashed_factory, rng_key="crash"
    )
    output = tmp_path / "crash"
    with pytest.raises(WorkerError):
        run_suite(
            output,
            PROFILE,
            [method],
            [Task("clean", "reference")],
            workers=2,
            plots=False,
        )
    assert read_status(output)["status"] == "failed"
    assert not (output / "runs.csv").exists()
    assert not mp.active_children()


def test_keyboard_interrupt_stops_owned_workers(tmp_path, monkeypatch):
    original = Execution.collected

    def interrupt(self):
        original(self)
        raise KeyboardInterrupt()

    monkeypatch.setattr(Execution, "collected", interrupt)
    output = tmp_path / "cancel"
    with pytest.raises(KeyboardInterrupt):
        run_suite(
            output,
            PROFILE,
            [catalog_method("SGD")],
            default_tasks()[:2],
            workers=2,
            plots=False,
        )
    assert read_status(output)["status"] == "stopped"
    assert not (output / "runs.csv").exists()
    assert not mp.active_children()


def test_export_failure_is_not_session_abort(tmp_path, monkeypatch):
    output = tmp_path / "export"
    original = pd.DataFrame.to_csv

    def fail(frame, path, *args, **kwargs):
        if Path(path).name == "runs.csv":
            raise OSError("disk failure")
        return original(frame, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
    session = EvaluationSession()
    with pytest.raises(OSError, match="disk failure"), session:
        pending = run_suite(
            output,
            PROFILE,
            [catalog_method("SGD")],
            [Task("clean", "reference")],
            workers=2,
            plots=False,
        )
    assert session.state == "released"
    assert session.export_status == "failed"
    assert len(pending.result()["runs"]) == 2
    assert read_status(output)["status"] == "export_failed"
    assert not validate_result(output)["valid"]


def test_failed_abort_status_does_not_strand_outer_session(tmp_path, monkeypatch):
    original = Execution.failed

    def broken_status(self, stopped):
        original(self, stopped)
        raise OSError("cannot record abort")

    monkeypatch.setattr(Execution, "failed", broken_status)
    session = EvaluationSession()
    with pytest.raises(RuntimeError, match="outer failure"), session:
        run_suite(
            tmp_path / "abort-status",
            PROFILE,
            [catalog_method("SGD")],
            [Task("clean", "reference")],
            plots=False,
            progress=False,
        )
        raise RuntimeError("outer failure")
    assert session.state == "aborted"
    with EvaluationSession() as fresh:
        pass
    assert fresh.state == "released"


def test_validator_detects_damaged_and_unfinished_export(tmp_path):
    output = tmp_path / "valid"
    run_suite(
        output,
        PROFILE,
        [catalog_method("SGD")],
        [Task("clean", "reference")],
        workers=2,
        plots=False,
    )
    (output / "summary.csv").write_text("damaged")
    assert any(
        "Damaged export: summary.csv" in e for e in validate_result(output)["errors"]
    )
    status = read_status(output)
    status["status"] = "stopped"
    (output / "execution_status.json").write_text(json.dumps(status))
    assert "Execution/export is not complete" in validate_result(output)["errors"]


@pytest.mark.parametrize("after_ready", [False, True])
def test_parent_death_stops_workers(tmp_path, after_ready):
    output = tmp_path / "orphan"
    diagnostic_path = tmp_path / "subprocess.log"
    diagnostic = diagnostic_path.open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("parallel_probe.py")),
            "slow",
            str(output),
        ],
        stdout=subprocess.DEVNULL,
        stderr=diagnostic,
    )
    children = []
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (output / "execution_status.json").exists():
                status = read_status(output)
                children = status["worker_pids"]
                if len(children) == 2 and (
                    not after_ready or len(status.get("ready_worker_pids", [])) == 2
                ):
                    break
            if process.poll() is not None:
                pytest.fail(diagnostic_path.read_text(errors="replace"))
            time.sleep(0.05)
        assert len(children) == 2
        # Windows venv python.exe may be a launcher, not the coordinator itself.
        coordinator_pid = read_status(output)["coordinator_pid"]
        os.kill(coordinator_pid, signal.SIGTERM)
        process.wait(timeout=10)
        deadline = time.monotonic() + (12 if after_ready else 30)
        alive = list(children)
        while alive and time.monotonic() < deadline:
            alive = []
            for pid in children:
                try:
                    if resident_bytes(pid) > 0:
                        alive.append(pid)
                except OSError:
                    pass
            time.sleep(0.1)
        assert not alive
        assert not (output / "runs.csv").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        for pid in children:
            try:
                if resident_bytes(pid) > 0:
                    os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        diagnostic.close()
