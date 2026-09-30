# BCES-IoV: implementation and reproducibility code

This repository contains the Python implementation, experiment entry points,
configuration, SUMO network definition, and tests for the BCES-IoV study. It
is a code-only release: it does **not** contain third-party datasets, trained
checkpoints, simulation/analysis outputs, preregistration artifacts, or the
manuscript. Consequently, a fresh clone can run the synthetic smoke tests, but
cannot independently reconstruct the reported numerical tables without the
specified source data and frozen study artifacts.

## Contents

- `bces/`: protocol, geometry, regret oracle, models, simulation, baselines,
  network accounting, and evaluation modules.
- `scripts/`: environment checks, acquisition/manifest utilities, training,
  calibration, controlled experiments, frozen analyses, and external-data
  validation entry points.
- `configs/`, `schemas/`, `sumo/`: study configurations, schema, and the
  controlled SUMO network.
- `tests/`: unit, integration, and smoke tests.
- `pyproject.toml`, `uv.lock`, `Makefile`: Python environment and common tasks.

## Environment and code verification

The project targets Python 3.11 and pins SUMO to version 1.27.1. With
[`uv`](https://docs.astral.sh/uv/) installed:

```powershell
uv sync --extra phase0 --frozen
uv run --no-sync python scripts/01_verify_environment.py
uv run --no-sync python -m pytest -q tests/unit tests/smoke
uv run --no-sync python scripts/02_smoke_pipeline.py
```

The self-contained selection above passed 330 tests in the release staging
copy. The integration suite also includes tests that require external data or
frozen output artifacts and is not expected to pass in a code-only clone.

## Reproducing the reported analyses

The numbered scripts preserve the original pipeline order. The confirmatory
closed-loop analysis entry point is `scripts/32_analyze_closed_loop_v2.py`.
It verifies SHA-256 digests for registered branch files before computing
scenario-clustered results. It requires the original
`outputs/study_b/closed_loop_confirmation_v2/` preregistration and branch
artifacts; this repository intentionally does not supply them. Do not create
new seed sets and call them a replication of the registered result.

The scripts include historical development and negative-result probes to
retain analytical provenance. Their presence does not mean that all probes
were confirmatory or that all observed improvements replicated. Controlled
simulation outcomes and external trajectory proxies have different evidence
roles and should not be pooled.

The source-data manifests under `configs/data/` identify selection and source
contracts but contain no raw trajectories. Dataset acquisition is subject to
each provider's terms. See the associated article's Data Availability
statement for the data sources and request route for derived artifacts.

## Scope and licensing

This release does not convey third-party dataset redistribution rights. It
also does not include a software license until the rights holders approve one.
