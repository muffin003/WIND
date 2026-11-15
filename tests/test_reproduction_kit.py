"""The public recipe must reconstruct source bytes, never start calculations."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def recipe(tmp_path):
    path = Path(__file__).resolve().parents[1] / "reproducibility/0.7.0/prepare.py"
    spec = importlib.util.spec_from_file_location("wind_reproduction_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path / "checkout"
    module.HERE = module.ROOT / "reproducibility/0.7.0"
    module.HERE.mkdir(parents=True)
    (module.ROOT / "src").mkdir()
    for name in ("web", "docs"):
        (module.ROOT / name).mkdir()
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "LICENSE",
        "README.md",
        "requirements.txt",
        "experiment.example.json",
    ):
        (module.ROOT / name).write_text("fixture", encoding="utf-8")
    data = b"value = 1\r\n"
    normalized = data.replace(b"\r\n", b"\n")
    (module.ROOT / "src/module.py").write_bytes(normalized)
    records = {
        "module.py": {
            "sha256_original": hashlib.sha256(data).hexdigest(),
            "sha256_lf": hashlib.sha256(normalized).hexdigest(),
            "line_endings": "CRLF",
        }
    }
    (module.HERE / "provenance.json").write_text(
        json.dumps({"builds": {"rl": {"source": records}}}), encoding="utf-8"
    )
    return module, records


def test_reproduction_restores_original_bytes(recipe, tmp_path):
    module, _ = recipe
    output = module.prepare("rl", tmp_path / "prepared")
    assert (output / "src/module.py").read_bytes() == b"value = 1\r\n"
    assert (module.ROOT / "src/module.py").read_bytes() == b"value = 1\n"


def test_reproduction_rejects_changed_source_before_writing(recipe, tmp_path):
    module, _ = recipe
    (module.ROOT / "src/module.py").write_bytes(b"changed\n")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Source changed"):
        module.prepare("rl", output)
    assert not output.exists()


def test_reproduction_never_overwrites(recipe, tmp_path):
    module, _ = recipe
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "user.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.prepare("rl", output)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_reproduction_uses_archival_override(recipe, tmp_path):
    module, records = recipe
    records["module.py"]["override"] = True
    folder = module.HERE / "canonical-paper-overrides"
    folder.mkdir()
    (folder / "module.py.txt").write_bytes(b"value = 1\n")
    (module.ROOT / "src/module.py").write_bytes(b"active = 2\n")
    (module.HERE / "provenance.json").write_text(
        json.dumps({"builds": {"canonical-paper": {"source": records}}}),
        encoding="utf-8",
    )
    output = module.prepare("canonical-paper", tmp_path / "prepared")
    assert (output / "src/module.py").read_bytes() == b"value = 1\r\n"
    assert (module.ROOT / "src/module.py").read_bytes() == b"active = 2\n"
