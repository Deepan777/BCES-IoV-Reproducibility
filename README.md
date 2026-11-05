# Experimental implementation and generated data

Python 3.11 is required. Recorded package versions are in `operations/environment.json`; SUMO is version 1.27.1.

## Setup and reproduce

```text
python -m pip install -r requirements-reproduction.txt
python reproduce.py materialize
python reproduce.py verify --materialized
python reproduce.py analyze all
python -m pytest -q tests
```

The materialization command restores generated experiment files to their expected paths. It verifies their SHA-256 digests and refuses to overwrite differing files. The compressed data packs preserve the original per-case bytes. Running the analyses writes new results beneath `reproduced/`; the saved reference results remain available for comparison.

Individual analyses can be run with `python reproduce.py analyze primary`, `broad`, `driving`, `source-query`, `robustness`, or `support`. The `primary` command covers independent calibration and evaluation. Source-query and receiver-local driving are separate studies; the earlier failed source-query run is retained separately and is not combined with the repaired run.

## Implementation

- `bces/`: receiver protocol, geometry, planners, models, simulation and statistics.
- `experiments/`: fixed classifiers, factorial training, operating-point selection, receiver-local driving, sensing errors, premise audits and external-view evaluation.
- `scripts/`: controlled simulation generation, original training/selection, source-query driving and external-data adapters.
- `configs/`, `sumo/`: experimental configurations and the controlled road network.
- `tests/`: causality, policy-feature consistency, transport and driving-contract checks.
- `data/generated/`: generated scenario inputs, frozen checkpoints, calibration/evaluation records, traffic/planning outcomes, ablation predictions and derived external metrics.
- `data/reproduction_manifest.json`: checksums and materialization paths.

The experiment sources and registered inputs retain their scientific bytes. `reproduce.py` relocates their working paths in memory. To invoke an original entry point, use `python reproduce.py run experiments/premise_and_runtime.py runtime`, replacing the script and arguments as needed. Training entry points include `scripts/13_train_decision_surfaces.py` and `experiments/development_study.py`. Repeated analyses use the supplied fixed operating points and scenario assignments.

## External input data

Derived external evaluation metrics are included. Regenerating them from observations also requires the provider's data: [UrbanIng-V2X](https://doi.org/10.7910/DVN/A9LPY7), [V2X-Real](https://mobility-lab.seas.ucla.edu/v2x-real/) and [FLUID](https://doi.org/10.6084/m9.figshare.29974954.v2). The adapters specify the expected local paths under `data/raw/` and `data/external/`. Provider sensor/trajectory files are obtained directly from those sources.
