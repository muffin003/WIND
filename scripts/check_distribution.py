"""Check release contents and smoke-test an installed (non-editable) wheel."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import tarfile
import threading
import urllib.request
import zipfile

ASSETS = {"index.html", "app.js", "guide-content.js", "styles.css"}


def check_archives(directory):
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise AssertionError("Expect exactly one wheel and one source archive")
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = set(archive.namelist())
    with tarfile.open(sources[0]) as archive:
        source_names = {"/".join(Path(n).parts[1:]) for n in archive.getnames()}
    for names in (wheel_names, source_names):
        forbidden = {"local", "legacy", "results", ".git", "__pycache__", "tmp"}
        assert not [
            n
            for n in names
            if forbidden.intersection(Path(n).parts)
            or any(part.startswith(".venv") for part in Path(n).parts)
            or Path(n).suffix in {".tex", ".pyc", ".pyo"}
        ], "Private/runtime files must not be distributed"
    assert {f"wind_benchmark/_web/{name}" for name in ASSETS} <= wheel_names
    assert {f"web/{name}" for name in ASSETS} <= source_names
    assert "tests/fixtures/paper_migration_reference.json" in source_names
    assert "src/algorithms/optimizers.py" in source_names
    assert "docs/releases/0.5.0.md" in source_names
    print(json.dumps({"wheel": wheels[0].name, "sdist": sources[0].name}))


def check_installed():
    import numpy as np
    import wind_benchmark
    from wind_benchmark.interfaces import web as web
    from wind_benchmark.core.runner import BenchmarkRunner
    from wind_benchmark.algorithms.catalog import create_optimizer
    from wind_benchmark.core.environment import make_environment
    from wind_benchmark.core.metrics import MetricsCollection, TrackingErrorMetric
    from wind_benchmark.core.oracle import FirstOrderOracle

    package = Path(wind_benchmark.__file__).resolve().parent
    assert package.name == "wind_benchmark", "Smoke must use an installed wheel"
    assert wind_benchmark.__version__ == importlib.metadata.version("wind-benchmark")
    assert web.WEB_ROOT == package / "_web"
    assert web.PROJECT_ROOT == Path.cwd().resolve()
    assert ASSETS <= {p.name for p in web.WEB_ROOT.iterdir()}
    env = make_environment(
        {"dim": 2, "landscape": {"type": "quadratic"}, "drift": {"type": "stationary"}},
        seed=3,
    )
    oracle = FirstOrderOracle(env)
    result = BenchmarkRunner(
        env, oracle, MetricsCollection([TrackingErrorMetric()])
    ).run(create_optimizer("SGD"), T=4, x0=np.ones(2), seed=7)
    assert oracle.n_queries == 4 and result.final_metrics
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.WindWebHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for path in ("/", "/app.js", "/guide-content.js", "/styles.css"):
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}{path}", timeout=10
            ) as response:
                assert response.status == 200 and response.read()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=10)
    print(
        json.dumps({"installed": str(package), "version": wind_benchmark.__version__})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--installed-smoke", action="store_true")
    args = parser.parse_args()
    if not args.dist and not args.installed_smoke:
        parser.error("Select --dist and/or --installed-smoke")
    if args.dist:
        check_archives(args.dist)
    if args.installed_smoke:
        check_installed()
