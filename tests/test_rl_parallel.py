"""The same replicated RL design, independent of process count."""

import hashlib
import json
import multiprocessing as mp
import pickle
import random
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")
import torch
from wind_benchmark.experiments import rl, parallel
from wind_benchmark.core.privacy import EvaluationSession, GroundTruthAccessError
from parallel_fixtures import rl_crash_worker, rl_inspection_worker

PROFILE = replace(
    rl.PROFILES["smoke"], fine_tune_episodes=10, fine_tune_checkpoints=(0, 10)
)


@pytest.fixture(scope="module")
def completed_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rl_parallel")
    for workers in (1, 2):
        output = root / str(workers)
        before = pickle.dumps((np.random.get_state(), random.getstate()))
        torch_state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        with pytest.MonkeyPatch.context() as patch:
            if workers == 2:
                patch.setattr(parallel, "_rl_worker", rl_inspection_worker)
            with EvaluationSession(expected_runs=12) as session:
                pending = rl.run_experiment(PROFILE, output, workers=workers)
                assert session.state == "running"
                assert (
                    json.loads((output / "execution_status.json").read_text())["status"]
                    == "awaiting_release"
                )
                assert not (output / "evaluation_runs.csv").exists()
                assert not (output / "evaluation_histories.npz").exists()
                with pytest.raises(GroundTruthAccessError):
                    pending.result()
            assert pending.result() == output
        assert session.state == "released"
        assert pickle.dumps((np.random.get_state(), random.getstate())) == before
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert torch.get_num_threads() == threads
        assert not mp.active_children()
    return root


def test_rl_process_count_preserves_design_and_all_numerical_exports(completed_runs):
    a, b = completed_runs / "1", completed_runs / "2"
    for name in ("evaluation_runs", "interaction_costs", "summary", "paired_contrasts"):
        pd.testing.assert_frame_equal(
            pd.read_csv(a / f"{name}.csv"),
            pd.read_csv(b / f"{name}.csv"),
            check_exact=True,
        )
    with (
        np.load(a / "evaluation_histories.npz") as left,
        np.load(b / "evaluation_histories.npz") as right,
    ):
        assert left.files == right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
    frame = pd.read_csv(b / "evaluation_runs.csv")
    assert len(frame) == 48
    assert frame.trace_key.tolist() == [f"episode_{i:05d}" for i in range(1, 49)]
    assert set(frame.status) == {"SUCCESS"}
    costs = pd.read_csv(b / "interaction_costs.csv")
    # 8 pretrained chains; all 12 chains adapt; 48 held-out episodes.
    assert costs.queries.sum() == (8 * 10 + 12 * 10 + 48) * 101


def test_rl_policy_files_and_release_inventory(completed_runs):
    from stable_baselines3 import PPO

    a, b = completed_runs / "1", completed_runs / "2"
    for original in (a / "policies").iterdir():
        other = b / "policies" / original.name
        if original.suffix == ".npz":
            with np.load(original) as left, np.load(other) as right:
                for key in left.files:
                    np.testing.assert_array_equal(left[key], right[key])
        else:
            left, right = (PPO.load(p, device="cpu") for p in (original, other))
            assert left.num_timesteps == right.num_timesteps
            for key, value in left.policy.state_dict().items():
                assert torch.equal(value, right.policy.state_dict()[key])
    manifest = json.loads((b / "manifest.json").read_text())
    assert manifest["protocol"] == "rl_transfer_v3"
    assert manifest["execution"]["workers_requested"] == 2
    assert manifest["execution"]["start_method"] == "spawn"
    assert manifest["ppo"]["n_envs"] == 1
    assert json.loads((b / "execution_status.json").read_text())["status"] == "complete"
    for name, digest in json.loads((b / "artifact_hashes.json").read_text()).items():
        assert hashlib.sha256((b / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_rl_invalid_workers_create_no_output(tmp_path, workers):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="positive integer"):
        rl.run_experiment(PROFILE, output, workers=workers)
    assert not output.exists()


def test_rl_dry_run_creates_no_output(tmp_path, capsys):
    output = tmp_path / "not-created"
    assert rl.main(["--dry-run", "--workers", "8", "--output-dir", str(output)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["chains"] == 60 and data["evaluation_episodes"] == 9000
    assert data["execution"]["workers_effective"] == 8
    assert not output.exists()


def test_rl_refuses_to_overwrite(completed_runs):
    with pytest.raises(FileExistsError):
        rl.run_experiment(PROFILE, completed_runs / "1", workers=2)


def test_rl_worker_failure_aborts_and_never_exports(tmp_path, monkeypatch):
    output = tmp_path / "crashed"
    monkeypatch.setattr(parallel, "_rl_worker", rl_crash_worker)
    with pytest.raises(parallel.WorkerError):
        with EvaluationSession() as session:
            rl.run_experiment(PROFILE, output, workers=2)
    assert session.state == "aborted"
    assert (
        json.loads((output / "execution_status.json").read_text())["status"] == "failed"
    )
    assert not (output / "evaluation_runs.csv").exists()
    assert not (output / "manifest.json").exists()
    assert not mp.active_children()


def test_rl_cancellation_stops_workers_and_seals_series(tmp_path, monkeypatch):
    output = tmp_path / "cancelled"

    def interrupt(self, index, counts):
        raise KeyboardInterrupt("controlled RL cancellation")

    monkeypatch.setattr(parallel.Execution, "_record_progress", interrupt)
    with pytest.raises(KeyboardInterrupt):
        with EvaluationSession() as session:
            rl.run_experiment(PROFILE, output, workers=2)
    assert session.state == "aborted"
    assert (
        json.loads((output / "execution_status.json").read_text())["status"]
        == "stopped"
    )
    assert not (output / "evaluation_histories.npz").exists()
    assert not mp.active_children()
