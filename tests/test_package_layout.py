"""Nested source layout preserves API boundaries and complete provenance."""

import hashlib
from pathlib import Path
import subprocess
import sys

import wind_benchmark
from wind_benchmark.core.privacy import _trusted_file
from wind_benchmark.core.runner import BenchmarkRunner
from wind_benchmark.reporting.common import source_hashes


def test_root_api_and_nested_source_fingerprints():
    assert wind_benchmark.BenchmarkRunner is BenchmarkRunner
    root = Path(wind_benchmark.__file__).resolve().parent
    hashes = source_hashes()
    for name in (
        "core/environment.py",
        "core/privacy.py",
        "algorithms/optimizers.py",
        "experiments/standard.py",
        "experiments/parallel.py",
        "integrations/gym_env.py",
        "interfaces/web.py",
        "reporting/common.py",
    ):
        assert hashes[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert "__init__.py" in hashes
    assert "experiments/__init__.py" in hashes
    assert not any("results/" in name or "__pycache__" in name for name in hashes)


def test_trusted_modules_remain_an_exact_path_allowlist(tmp_path):
    root = Path(wind_benchmark.__file__).resolve().parent
    assert _trusted_file(str(root / "core/runner.py"))
    assert _trusted_file(str(root / "experiments/parallel.py"))
    for path in (
        root / "runner.py",
        root / "algorithms/optimizers.py",
        root / "reporting/common.py",
        root / "interfaces/web.py",
        tmp_path / "core/runner.py",
    ):
        assert not _trusted_file(str(path))


def test_catalog_does_not_import_optional_integrations_or_experiments():
    script = (
        "import sys; import wind_benchmark.algorithms.catalog; "
        "assert 'gymnasium' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert not any(n.startswith('wind_benchmark.experiments.') "
        "for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
