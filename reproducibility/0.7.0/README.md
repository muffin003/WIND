# Reproducing the WIND 0.7.0 experiments

This kit contains **inputs, not published reference scores**: the source build
identities, version-pinned dependencies, task/profile specifications, random-stream
identities and commands. New result arrays and trained policies are not included.
The older files already tracked under `results/` are historical, not 0.7.0 scores.

## Two source builds

The completed canonical and five paper studies used an earlier 0.7.0 source
build. The final RL study used the current 0.7.0 build, including the ensemble-grid
fix and the parallel RL executor. Package version alone does not distinguish them.

`provenance.json` records both builds and their experiment settings. `prepare.py`
reconstructs the exact 37 Python source files for either build and verifies their
original SHA-256 hashes, including original line endings. Four older files are
stored as plain-text overrides under `canonical-paper-overrides/`; they are
archival reproduction inputs, not active modules or compatibility aliases.
Preparation never installs dependencies, starts an experiment, or overwrites an
existing destination. Keep the current checkout unchanged while preparing a build.

The reconstructed Python sources match the run manifests. Packaging metadata,
web assets and documentation come from this checkout; rebuilding a wheel is not
expected to reproduce the original ZIP bytes or wheel SHA-256. Original wheel
hashes identify the runs, while source hashes verify the reconstruction.

## Environment

The recorded runs used Windows x86-64, Python **3.12.14**, CPU execution and one
BLAS/OpenMP thread per worker. The two `*.lock.txt` files contain recorded package
version pins, not a cross-platform wheel lock. Floating-point results need not be
bitwise identical across operating systems, CPU libraries or dependency builds.
Use separate environments for the two builds. Run the commands from the checkout
root; activate with `source .venv-reproduce-*/bin/activate` on Linux instead of the
PowerShell activation command below.

### Canonical and paper

```powershell
py -3.12 -m venv .venv-reproduce-canonical
.\.venv-reproduce-canonical\Scripts\Activate.ps1
python reproducibility/0.7.0/prepare.py --profile canonical-paper --output local/reproduce_070/canonical-paper
python -m pip install -r reproducibility/0.7.0/canonical-paper.lock.txt
python -m pip install setuptools==84.2.1 wheel==0.48.0
python -m pip install --no-deps --no-build-isolation ./local/reproduce_070/canonical-paper
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

The explicit setuptools/wheel installation supplies packaging tools for the
fresh environment; these are not numerical experiment dependencies.

Inspect the canonical design without computation:

```powershell
python -m wind_benchmark.experiments.standard --profile standard --workers 8 --output-dir results/wind_tracking_v1/reproduction_001 --no-plots --dry-run
```

The following commands start **full, potentially long-running experiments**.
Choose unused output names. No resume or overwrite is implied.

```powershell
python -m wind_benchmark.experiments.standard --profile standard --workers 8 --output-dir results/wind_tracking_v1/reproduction_001 --no-plots
python -m wind_benchmark.experiments.canonical results/wind_tracking_v1/reproduction_001
python -m wind_benchmark.experiments.paper --profile paper --studies all --output-dir results/paper_tracking_v4 --run-name reproduction_001 --no-plots
```

- Canonical: `wind_tracking_v1`, `catalog_v2`, all 32 tasks and 22 methods;
  evaluation seeds 10001–10050, 4000 queries per compatible run; no tuning.
  The design has 22,450 compatible runs. Development seeds are 20001–20020.
- Paper: `paper_tracking_v4`, evaluation seeds 1001–1030; calibration 1500 steps,
  memory 2000 steps, zeroth/dimension 10,000 queries, geometry 1000 steps.
  Dimension uses 5, 10, 20, 50, 100 and 200. All five studies run together so the
  dimension-20 development selection is reused between zeroth and dimension.
- Development selection: 12 candidates per ZO configuration/dimension, seeds
  9001–9005, 2000 queries. The grid is fixed in the source and documented in
  [paper_tracking_v4.md](../../docs/paper_tracking_v4.md).
  Preserve `TUNING_FAILED`: do not expand OnePointSPSA's search or substitute
  defaults when reproducing this protocol.
- Canonical scores use queried points; paper scores use recommended iterates.
  Keep the distinct timing, aggregation and recovery definitions.

Full definitions: [canonical](../../docs/default_run.md),
[algorithms](../../docs/algorithms_070.md), [metrics](../../docs/metrics_v2.md).

### RL

Use a new shell/environment after the canonical/paper setup:

```powershell
py -3.12 -m venv .venv-reproduce-rl
.\.venv-reproduce-rl\Scripts\Activate.ps1
python reproducibility/0.7.0/prepare.py --profile rl --output local/reproduce_070/rl
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
python -m pip install -r reproducibility/0.7.0/rl.lock.txt
python -m pip install --no-deps --no-build-isolation ./local/reproduce_070/rl
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m wind_benchmark.experiments.rl --profile paper --seed 20261112 --workers 8 --output-dir results/rl_transfer_v3/reproduction_001 --no-plots --dry-run
```

After checking the design, remove `--dry-run` to start the full run. The protocol
is `rl_transfer_v3`: Q-learning and external Stable-Baselines3 2.9.2 PPO; three
pretraining conditions; ten training seeds; thirty held-out workloads; checkpoints
0, 10, 50, 100 and 200. The complete design has 9000 evaluation episodes and 300
policy checkpoints. Pretraining, adaptation and evaluation consume respectively
12,880,000, 3,612,000 and 2,709,000 value queries (19,201,000 in total).
See [RL definitions](../../docs/rl_transfer_v3.md).

## Checking a reproduction

The canonical validator requires a complete export, including local trajectories;
the recipe is not a substitute for those files. Preserve every new manifest,
status, seed, selected parameter, failure/censoring record and source fingerprint.
Paper's design contains 8910 evaluation records, including tuning-failure records
that do not represent executed evaluation trajectories. RL completion requires
all 9000 episodes, 300 checkpoints and the separate phase budgets above.

Timing, PID, paths, memory use and timestamps are execution metadata, not expected
numerical matches. New runs should be compared by task, method, seed, protocol and
status, with the declared metric aggregation and uncertainty. The full tests check
finite-step formulas, privacy, exports and parallel determinism; this publication
does not run the full experiments again or claim a fresh cross-platform replication.
