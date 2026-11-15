"""Reconstruct a source snapshot; never install packages or run experiments."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def prepare(profile, output):
    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    specification = provenance["builds"][profile]
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    source = {}
    for name, record in specification["source"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source path: {name}")
        origin = (
            HERE / "canonical-paper-overrides" / (name + ".txt")
            if record.get("override")
            else ROOT / "src" / name
        )
        normalized = origin.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != record["sha256_lf"]:
            raise ValueError(f"Source changed since the recorded build: {name}")
        original = (
            normalized.replace(b"\n", b"\r\n")
            if record["line_endings"] == "CRLF"
            else normalized
        )
        if hashlib.sha256(original).hexdigest() != record["sha256_original"]:
            raise ValueError(f"Source byte reconstruction failed: {name}")
        source[name] = original
    # Validate every source file before creating any output.
    output.mkdir(parents=True, exist_ok=False)
    for name, data in source.items():
        target = output / "src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name in (
        "pyproject.toml",
        "MANIFEST.in",
        "LICENSE",
        "README.md",
        "requirements.txt",
        "experiment.example.json",
    ):
        shutil.copyfile(ROOT / name, output / name)
    for name in ("web", "docs"):
        shutil.copytree(
            ROOT / name,
            output / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    print(f"Verified {len(source)} original Python source hashes: {profile}")
    print(f"Prepared source only: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("canonical-paper", "rl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.profile, args.output)


if __name__ == "__main__":
    main()
