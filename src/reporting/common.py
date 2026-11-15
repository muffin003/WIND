"""Strict JSON and console-safe reporting; no evaluator authority is granted here."""

import builtins
import hashlib
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import sys
import uuid
import numpy as np


def source_hashes():
    """Fingerprint the complete nested package, with unambiguous relative paths."""
    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if not {"__pycache__", ".pytest_cache", "results"}.intersection(
            path.relative_to(root).parts
        )
    }


def json_safe(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: json_safe(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value, *, prepared=False):
    """Atomic new-file export. Existing experiment artifacts are never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(
            value if prepared else json_safe(value),
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    partial.rename(path)
    return path


def safe_print(*args, **kwargs):
    stream = kwargs.get("file") or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    args = tuple(
        str(arg).encode(encoding, errors="backslashreplace").decode(encoding)
        for arg in args
    )
    return builtins.print(*args, **kwargs)
