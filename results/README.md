# Experiment Results

This directory contains versioned outputs from WIND benchmark runs.

- `lyapunov/` is the default destination of the experiment suite.
- Short validation runs and private before/after checks belong under `local/`,
  not among versioned publication results.
- `wind_comprehensive_analysis/` contains the current publication figures.

Previously tracked outputs are retained as historical material. New 0.7.0
canonical, paper and RL outputs are kept locally and excluded from this Git
publication. Their absence is deliberate, not an incomplete experiment export.
See the [reproduction kit](../reproducibility/0.7.0/README.md) for source
snapshots, dependency pins, profiles, seeds and commands to generate new outputs.
Private validation runs remain under `local/`, never in published result data.
