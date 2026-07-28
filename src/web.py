"""Local web interface for configuring and running WIND experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .web_runner import (
    DRIFTS,
    LANDSCAPES,
    METRICS,
    NOISES,
    OPTIMIZERS,
    ORACLES,
    WorkbenchConfigurationError,
    optimizer_configs_from_payload,
    validate_workbench_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = PROJECT_ROOT / "web"
MAX_REQUEST_BYTES = 1_000_000
PROGRESS_PATTERN = re.compile(r"\[(\d+)/(\d+)\]")

KNOWN_OPTIMIZERS = {
    "SGD",
    "SGD_Polyak",
    "HeavyBall",
    "Nesterov",
    "Adam",
    "AdamW",
    "AMSGrad",
    "SMD",
    "RDA",
    "ProxSGD",
    "AdaptiveLR",
    "SignSGD",
    "RandomSearch",
    "OnePointSPSA",
    "FiniteDiffCentral",
    "FDSA",
    "SPSA",
    "ZOSGD",
    "ZOSignSGD",
    "QuadraticInterpolation",
    "KieferWolfowitz",
    "NedicSubgradient",
    "AcceleratedSPSA",
    "CMAES",
    "GPUCB",
}

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
}


class ConfigurationError(ValueError):
    """Raised when a submitted experiment configuration is invalid."""


@dataclass
class ExperimentJob:
    job_id: str
    config: Dict[str, Any]
    config_path: Path
    output_dir: Path
    module: str = "wind_benchmark"
    status: str = "starting"
    current: int = 0
    total: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    return_code: Optional[int] = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=160))
    process: Optional[subprocess.Popen[str]] = field(default=None, repr=False)
    cancel_requested: bool = False

    def snapshot(self) -> Dict[str, Any]:
        elapsed_to = self.finished_at if self.finished_at is not None else time.time()
        return {
            "id": self.job_id,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "progress": round(self.current / self.total * 100, 1) if self.total else 0,
            "elapsed_seconds": round(elapsed_to - self.started_at, 1),
            "output_dir": self.config.get("output_dir")
            or self.config.get("runner", {}).get("output_dir"),
            "return_code": self.return_code,
            "logs": list(self.logs)[-18:],
        }


JOB_LOCK = threading.Lock()
CURRENT_JOB: Optional[ExperimentJob] = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_list(config: Dict[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{key} must be a non-empty list")
    return value


def resolve_output_dir(output_dir: str) -> Path:
    """Resolve an output directory and keep browser-triggered writes in the project."""
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ConfigurationError("output_dir must be a non-empty string")

    supplied = Path(output_dir.strip())
    resolved = (
        (PROJECT_ROOT / supplied).resolve()
        if not supplied.is_absolute()
        else supplied.resolve()
    )
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ConfigurationError(
            "output_dir must stay inside the project directory"
        ) from exc
    return resolved


def validate_config(payload: Any) -> Dict[str, Any]:
    """Validate and normalize the same seven fields accepted by the WIND CLI."""
    if not isinstance(payload, dict):
        raise ConfigurationError("Request body must contain a JSON object")

    allowed = {
        "output_dir",
        "seeds",
        "steps",
        "rho_values",
        "drift_values",
        "dimensions",
        "optimizers",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown configuration field(s): {sorted(unknown)}")

    output_dir = payload.get("output_dir")
    resolve_output_dir(output_dir)

    steps = payload.get("steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ConfigurationError("steps must be a positive integer")

    seeds = _require_list(payload, "seeds")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in seeds
    ):
        raise ConfigurationError("seeds must contain non-negative integers")

    rho_values = _require_list(payload, "rho_values")
    if not all(
        _is_number(value) and float(value) in {1.0, 0.5, 0.2} for value in rho_values
    ):
        raise ConfigurationError("rho_values supports only 1.0, 0.5 and 0.2")

    drift_values = _require_list(payload, "drift_values")
    if not all(_is_number(value) and value > 0 for value in drift_values):
        raise ConfigurationError("drift_values must contain positive numbers")

    dimensions = _require_list(payload, "dimensions")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in dimensions
    ):
        raise ConfigurationError("dimensions must contain positive integers")

    optimizers = payload.get("optimizers")
    if optimizers is not None:
        if not isinstance(optimizers, list) or not optimizers:
            raise ConfigurationError("optimizers must be null or a non-empty list")
        if not all(isinstance(value, str) for value in optimizers):
            raise ConfigurationError("optimizers must contain method names")
        unknown_optimizers = set(optimizers) - KNOWN_OPTIMIZERS
        if unknown_optimizers:
            raise ConfigurationError(
                f"Unknown optimizer(s): {sorted(unknown_optimizers)}"
            )

    return {
        "output_dir": output_dir.strip().replace("\\", "/"),
        "seeds": list(dict.fromkeys(seeds)),
        "steps": steps,
        "rho_values": [float(value) for value in dict.fromkeys(rho_values)],
        "drift_values": [float(value) for value in dict.fromkeys(drift_values)],
        "dimensions": list(dict.fromkeys(dimensions)),
        "optimizers": (
            list(dict.fromkeys(optimizers)) if optimizers is not None else None
        ),
    }


def _run_job(job: ExperimentJob) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    # The benchmark emits Unicode status markers. Force UTF-8 for the child
    # process so Windows legacy console encodings cannot abort an otherwise
    # successful run while its output is being captured by the web server.
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    command = [
        sys.executable,
        "-m",
        job.module,
        "--config",
        str(job.config_path),
    ]

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with JOB_LOCK:
            job.process = process
            job.status = "running"

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            with JOB_LOCK:
                if line:
                    job.logs.append(line)
                match = PROGRESS_PATTERN.search(line)
                if match:
                    job.current = int(match.group(1))
                    job.total = int(match.group(2))

        return_code = process.wait()
        with JOB_LOCK:
            job.return_code = return_code
            if job.cancel_requested:
                job.status = "cancelled"
            else:
                job.status = "completed" if return_code == 0 else "failed"
            job.finished_at = time.time()
    except Exception as exc:  # pragma: no cover - defensive process boundary
        with JOB_LOCK:
            job.logs.append(f"Failed to start experiment: {exc}")
            job.status = "failed"
            job.finished_at = time.time()
            job.return_code = -1


def start_job(config: Dict[str, Any]) -> ExperimentJob:
    global CURRENT_JOB

    with JOB_LOCK:
        if CURRENT_JOB is not None and CURRENT_JOB.status in {"starting", "running"}:
            raise ConfigurationError("Another experiment is already running")

        output_dir = resolve_output_dir(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "experiment_config.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        CURRENT_JOB = ExperimentJob(
            job_id=uuid.uuid4().hex[:12],
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            total=(
                len(config["seeds"])
                * len(config["rho_values"])
                * len(config["drift_values"])
                * len(config["dimensions"])
                * len(config["optimizers"] or KNOWN_OPTIMIZERS)
            ),
        )
        job = CURRENT_JOB

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def start_workbench_job(config: Dict[str, Any]) -> ExperimentJob:
    """Start a general benchmark run configured by the workbench UI."""
    global CURRENT_JOB

    try:
        validate_workbench_config(config)
    except WorkbenchConfigurationError as exc:
        raise ConfigurationError(str(exc)) from exc

    optimizer_configs = optimizer_configs_from_payload(config)
    if not optimizer_configs:
        raise ConfigurationError("Select at least one optimizer")

    output_dir = resolve_output_dir(config["runner"]["output_dir"])
    if config["oracle"].get("type") == "offline":
        recorded_path = config["oracle"].get("recorded_path")
        if not isinstance(recorded_path, str) or not recorded_path.strip():
            raise ConfigurationError("Offline oracle requires recorded_path")
        resolved_recording = (PROJECT_ROOT / recorded_path).resolve()
        try:
            resolved_recording.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ConfigurationError(
                "Offline replay path must stay inside the project directory"
            ) from exc
        if not resolved_recording.is_file():
            raise ConfigurationError("Offline replay file was not found")

    with JOB_LOCK:
        if CURRENT_JOB is not None and CURRENT_JOB.status in {"starting", "running"}:
            raise ConfigurationError("Another benchmark run is already running")

        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "workbench_config.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        seeds = list(dict.fromkeys(config["runner"]["seeds"]))
        CURRENT_JOB = ExperimentJob(
            job_id=uuid.uuid4().hex[:12],
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            module="wind_benchmark.web_runner",
            total=len(seeds) * len(optimizer_configs),
        )
        job = CURRENT_JOB

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def capability_catalog() -> Dict[str, Any]:
    return {
        "landscapes": sorted(LANDSCAPES),
        "drifts": sorted(DRIFTS),
        "noises": sorted(NOISES),
        "oracles": sorted(ORACLES),
        "metrics": sorted(METRICS),
        "optimizers": [
            {"name": name, "order": oracle_type}
            for name, (_, oracle_type) in OPTIMIZERS.items()
        ],
        "exports": ["json", "csv", "plotly"],
        "geometry": ["euclidean", "simplex", "stiefel", "grassmann"],
    }


def list_recent_results(limit: int = 30) -> list[Dict[str, Any]]:
    results_root = PROJECT_ROOT / "results"
    if not results_root.is_dir():
        return []
    files = sorted(
        results_root.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: max(1, min(limit, 100))]
    items = []
    for path in files:
        item: Dict[str, Any] = {
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "name": path.name,
            "size": path.stat().st_size,
            "modified": path.stat().st_mtime,
            "kind": "result",
        }
        if path.name in {
            "manifest.json",
            "experiment_metadata.json",
            "workbench_summary.json",
        }:
            item["kind"] = "summary"
        if path.stat().st_size <= 5_000_000:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    optimizer = data.get("optimizer_info", {})
                    metadata = data.get("metadata", {})
                    item["optimizer"] = optimizer.get("name") or data.get("optimizer")
                    item["status"] = data.get("status")
                    item["steps"] = metadata.get("total_steps")
                    item["dimension"] = metadata.get("dimension")
                    item["metrics"] = data.get("final_metrics", {})
            except (OSError, json.JSONDecodeError):
                item["kind"] = "data"
        items.append(item)
    return items


def cancel_job(job_id: str) -> ExperimentJob:
    with JOB_LOCK:
        if CURRENT_JOB is None or CURRENT_JOB.job_id != job_id:
            raise ConfigurationError("Experiment job not found")
        if CURRENT_JOB.status not in {"starting", "running"}:
            return CURRENT_JOB
        CURRENT_JOB.cancel_requested = True
        process = CURRENT_JOB.process
        if process is not None:
            process.terminate()
        return CURRENT_JOB


class WindWebHandler(BaseHTTPRequestHandler):
    server_version = "WINDLab/0.2"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    def _send_json(
        self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ConfigurationError("Invalid Content-Length") from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ConfigurationError("Invalid request body size")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigurationError("Request body must be valid UTF-8 JSON") from exc

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            with JOB_LOCK:
                job = CURRENT_JOB.snapshot() if CURRENT_JOB is not None else None
            self._send_json({"connected": True, "version": "0.2.0", "job": job})
            return

        if path == "/api/catalog":
            self._send_json(capability_catalog())
            return

        if path == "/api/results":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["30"])[0])
            except ValueError:
                limit = 30
            self._send_json({"results": list_recent_results(limit)})
            return

        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/").strip("/")
            with JOB_LOCK:
                if CURRENT_JOB is None or CURRENT_JOB.job_id != job_id:
                    self._send_json(
                        {"error": "Experiment job not found"}, HTTPStatus.NOT_FOUND
                    )
                    return
                snapshot = CURRENT_JOB.snapshot()
            self._send_json({"job": snapshot})
            return

        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        try:
            if path == "/api/run":
                config = validate_config(self._read_json())
                job = start_job(config)
                self._send_json({"job": job.snapshot()}, HTTPStatus.ACCEPTED)
                return

            if path == "/api/benchmark/run":
                config = self._read_json()
                job = start_workbench_job(config)
                self._send_json({"job": job.snapshot()}, HTTPStatus.ACCEPTED)
                return

            cancel_match = re.fullmatch(r"/api/jobs/([a-f0-9]+)/cancel", path)
            if cancel_match:
                job = cancel_job(cancel_match.group(1))
                self._send_json({"job": job.snapshot()}, HTTPStatus.ACCEPTED)
                return

            self._send_json({"error": "Endpoint not found"}, HTTPStatus.NOT_FOUND)
        except ConfigurationError as exc:
            status = (
                HTTPStatus.CONFLICT
                if "already running" in str(exc)
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json({"error": str(exc)}, status)

    def _serve_static(self, requested_path: str) -> None:
        relative = (
            "index.html"
            if requested_path == "/"
            else unquote(requested_path.lstrip("/"))
        )
        asset = (WEB_ROOT / relative).resolve()
        try:
            asset.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if not asset.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = asset.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            MIME_TYPES.get(asset.suffix.lower(), "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wind-benchmark-ui",
        description="Open the local WIND Benchmark workbench.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Local interface host")
    parser.add_argument("--port", default=8765, type=int, help="Local interface port")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser automatically"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not WEB_ROOT.is_dir():
        raise SystemExit(f"Web interface assets were not found: {WEB_ROOT}")

    server = ThreadingHTTPServer((args.host, args.port), WindWebHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"WIND Lab is available at {url}")
    print("Press Ctrl+C to stop the local interface.")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWIND Lab stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
