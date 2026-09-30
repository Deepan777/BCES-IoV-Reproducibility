#!/usr/bin/env python3
"""Fit scenario-clustered conservative shrinkage on calibration only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.calibration import (
    CalibrationTarget,
    calibration_curve,
    select_shrinkage,
)
from bces.models.dataset import SurfaceDataset
from bces.models.surfacenet import SurfaceNetLite
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "calibration" / "surfacenet_calibration_v2.yaml"


def _grid(specification: dict) -> tuple[float, ...]:
    start, stop, step = map(float, (specification["start"], specification["stop"], specification["step"]))
    return tuple(round(value, 10) for value in np.arange(start, stop + step / 2, step))


def _predict(
    dataset: SurfaceDataset, checkpoint: Path, device: torch.device,
) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SurfaceNetLite()
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    rows = []
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            offsets, _ = model(moved)
            rows.append(offsets.cpu().numpy().astype(np.float64))
    return np.concatenate(rows)


def _primary_seed(phase6: dict) -> int:
    row = min(
        phase6["seeds"],
        key=lambda item: (
            item["best_metrics"]["unsafe_accept_rate"],
            -item["best_metrics"]["valid_coverage"],
            item["best_metrics"]["total_loss"],
        ),
    )
    return int(row["seed"])


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["partition"] != "calibration" or config["test_partition_access"] != "prohibited":
        raise RuntimeError("invalid calibration partition contract")
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("calibration_v2 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("calibration requires a clean committed revision")
    dataset = SurfaceDataset(ROOT / config["phase4_root"], "calibration")
    phase6_root = ROOT / config["phase6_root"]
    phase6 = json.loads((phase6_root / "run_manifest.json").read_text(encoding="utf-8"))
    target = CalibrationTarget(
        unsafe_accept_upper=float(config["unsafe_accept_upper_target"]),
        confidence=float(config["confidence"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        refresh_guard_band=float(config["refresh_guard_band"]),
        minimum_accepted=int(config["minimum_accepted_per_curve"]),
    )
    grid = _grid(config["shrinkage_grid"])
    drift = dataset.normalized_drift.numpy().astype(np.float64)
    valid = dataset.valid.numpy().astype(bool)
    behaviors = dataset.reference_tensors["identifiers"][dataset.reference_indices, 0].numpy().astype(int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = []
    for seed_index, seed_row in enumerate(phase6["seeds"]):
        seed = int(seed_row["seed"])
        checkpoint = phase6_root / f"seed_{seed}" / "best.pt"
        if sha256_file(checkpoint) != seed_row["best_checkpoint_sha256"]:
            raise ValueError("Phase-6 checkpoint hash mismatch")
        offsets = _predict(dataset, checkpoint, device)
        curves, selections = {}, {}
        for behavior_id in map(int, config["behavior_ids"]):
            indices = np.flatnonzero(behaviors == behavior_id)
            cluster_count = len({dataset.scenario_ids[index] for index in indices})
            pooled = len(indices) < int(config["pool_behavior_below_points"]) or cluster_count < int(config["pool_behavior_below_clusters"])
            if pooled:
                indices = np.arange(len(dataset))
            clusters = tuple(dataset.scenario_ids[index] for index in indices)
            curve = calibration_curve(
                offsets[indices], drift[indices], valid[indices], clusters, grid, target,
                seed=int(config["bootstrap_seed"]) + 10000 * seed_index + 100 * behavior_id,
            )
            selection = select_shrinkage(curve, target)
            selection.update(
                {
                    "behavior_id": behavior_id, "risk_class": 1,
                    "pooled_behaviors": pooled, "point_count": len(indices),
                    "cluster_count": len(set(clusters)),
                    "directional_shrinkage": [selection["shrinkage"]] * 16,
                }
            )
            curves[str(behavior_id)] = curve
            selections[str(behavior_id)] = selection
        seed_results.append(
            {
                "seed": seed, "checkpoint_sha256": seed_row["best_checkpoint_sha256"],
                "selections": selections, "risk_coverage_curves": curves,
                "all_behavior_targets_met": all(row["target_met"] for row in selections.values()),
                "all_behavior_coverage_nonzero": all(row["accepted"] > 0 for row in selections.values()),
            }
        )
        print(
            f"calibrated seed={seed} target_cells="
            f"{sum(row['target_met'] for row in selections.values())}/5",
            flush=True,
        )
    primary_seed = _primary_seed(phase6)
    provisional = {
        "schema_version": 1, "phase": 7, "status": "PASS",
        "partition": "calibration", "test_partition_accessed": False,
        "target": {
            "unsafe_accept_upper": target.unsafe_accept_upper,
            "confidence": target.confidence,
            "bootstrap_replicates": target.bootstrap_replicates,
            "refresh_guard_band": target.refresh_guard_band,
            "minimum_accepted": target.minimum_accepted,
        },
        "calibration_data": dataset.summary, "primary_seed": primary_seed,
        "seed_results": seed_results, "config_sha256": sha256_file(CONFIG),
        "phase6_manifest_sha256": phase6["manifest_sha256"],
        "git": state, "completed_utc": utc_now(),
    }
    calibration_hash = canonical_json_hash(provisional)
    provisional["calibration_id"] = max(1, int(calibration_hash[:2], 16))
    provisional["calibration_sha256"] = canonical_json_hash(provisional)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "calibration.json", provisional)
    print(json.dumps(provisional, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
