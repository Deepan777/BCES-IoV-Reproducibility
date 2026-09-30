#!/usr/bin/env python3
"""Audit fixed-polytope adequacy against oracle labels and a flexible MLP."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.calibration import quantize_numpy
from bces.models.dataset import SurfaceDataset
from bces.models.surfacenet import SurfaceNetLite, normal_tensor
from bces.models.validity_mlp import ValidityMLPLite
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/geometry_audit_v1.yaml"


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _predict(
    surface: SurfaceNetLite,
    mlp: ValidityMLPLite,
    dataset: SurfaceDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    offsets, probabilities = [], []
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    surface.eval()
    mlp.eval()
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                predicted, _ = surface(batch)
                logits = mlp(batch)
            offsets.append(predicted.float().cpu().numpy().astype(np.float64))
            probabilities.append(
                torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
            )
    return np.concatenate(offsets), np.concatenate(probabilities)


def _classification(accepted: np.ndarray, valid: np.ndarray) -> dict:
    accepted = np.asarray(accepted, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    unsafe = accepted & ~valid
    safe_reject = ~accepted & valid
    accepted_count = int(accepted.sum())
    return {
        "accepted": accepted_count,
        "unsafe_accepted": int(unsafe.sum()),
        "invalid_oracle_points_included": int(unsafe.sum()),
        "valid_oracle_points_excluded": int(safe_reject.sum()),
        "unsafe_accept_rate": (
            int(unsafe.sum()) / accepted_count if accepted_count else None
        ),
        "coverage": accepted_count / len(valid),
        "valid_coverage": int((accepted & valid).sum()) / int(valid.sum()),
        "safe_reject_rate": int(safe_reject.sum()) / int(valid.sum()),
    }


def _boundary_audit(
    path: Path,
    partition: str,
    reference_offsets: dict[str, np.ndarray],
    shrinkages: dict[int, float],
    reference_behaviors: dict[str, int],
) -> dict:
    normals = normal_tensor().numpy().astype(np.float64)
    errors = []
    boundaries = transitions = reentries = infeasible = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != partition:
                continue
            boundaries += 1
            probes = sorted(row["probes"], key=lambda item: item["radius"])
            labels = [
                bool(item["result"]["valid"])
                for item in probes
                if item["result"]["feasible"]
            ]
            infeasible += sum(not item["result"]["feasible"] for item in probes)
            first_invalid = next(
                (index for index, value in enumerate(labels) if not value), None
            )
            if first_invalid is not None and any(labels[first_invalid + 1 :]):
                reentries += 1
            if row["status"] != "transition":
                continue
            transitions += 1
            reference_id = row["reference_id"]
            behavior = reference_behaviors[reference_id]
            offsets = np.maximum(
                0.0,
                reference_offsets[reference_id] - shrinkages[behavior],
            )[None, :]
            calibrated = quantize_numpy(offsets)[0]
            direction = np.asarray(row["direction"], dtype=np.float64)
            products = normals @ direction
            positive = products > 1e-12
            predicted_radius = float(
                np.min(calibrated[positive] / products[positive])
            )
            errors.append(predicted_radius - float(row["valid_radius"]))
    error_array = np.asarray(errors, dtype=np.float64)
    return {
        "boundary_count": boundaries,
        "transition_boundary_count": transitions,
        "infeasible_probe_count": infeasible,
        "directional_monotonicity_violations": reentries,
        "directional_reentry_nonconvex_evidence": reentries,
        "boundary_signed_error_mean": float(error_array.mean()),
        "boundary_mae": float(np.abs(error_array).mean()),
        "boundary_rmse": float(np.sqrt(np.mean(error_array**2))),
        "boundary_underestimate_fraction": float((error_array < 0).mean()),
        "boundary_overestimate_fraction": float((error_array > 0).mean()),
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("geometry_audit_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("geometry audit requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("geometry audit requires CUDA")
    device = torch.device("cuda")
    dataset = SurfaceDataset(
        ROOT / config["artifact_root"], config["partition"]
    )
    surface_checkpoint = (
        ROOT
        / config["surface_root"]
        / f"seed_{config['surface_seed']}"
        / "best.pt"
    )
    mlp_checkpoint = (
        ROOT
        / config["validity_mlp_root"]
        / f"seed_{config['validity_mlp_seed']}"
        / "best.pt"
    )
    surface = SurfaceNetLite().to(device)
    surface.load_state_dict(
        torch.load(
            surface_checkpoint, map_location=device, weights_only=True
        )["model_state_dict"]
    )
    mlp = ValidityMLPLite().to(device)
    mlp.load_state_dict(
        torch.load(
            mlp_checkpoint, map_location=device, weights_only=True
        )["model_state_dict"]
    )
    offsets, probabilities = _predict(surface, mlp, dataset, device)
    valid = dataset.valid.numpy().astype(bool)
    drift = dataset.normalized_drift.numpy().astype(np.float64)
    point_behaviors = (
        dataset.reference_tensors["identifiers"][
            dataset.reference_indices, 0
        ]
        .numpy()
        .astype(int)
    )
    surface_calibration_path = ROOT / config["surface_calibration"]
    surface_calibration = json.loads(
        surface_calibration_path.read_text(encoding="utf-8")
    )
    surface_seed = next(
        row
        for row in surface_calibration["seed_results"]
        if int(row["seed"]) == int(config["surface_seed"])
    )
    surface_shrinkages = {
        int(behavior): float(selection["shrinkage"])
        for behavior, selection in surface_seed["selections"].items()
    }
    mlp_manifest_path = (
        ROOT / config["validity_mlp_root"] / "run_manifest.json"
    )
    mlp_manifest = json.loads(mlp_manifest_path.read_text(encoding="utf-8"))
    mlp_seed = next(
        row
        for row in mlp_manifest["seeds"]
        if int(row["seed"]) == int(config["validity_mlp_seed"])
    )
    mlp_thresholds = {
        int(behavior): float(result["selection"]["threshold"])
        for behavior, result in mlp_seed["calibration"].items()
    }
    normals = normal_tensor().numpy().astype(np.float64)
    uncalibrated_minimum = (offsets - drift @ normals.T).min(axis=1)
    shrinkage = np.asarray(
        [surface_shrinkages[behavior] for behavior in point_behaviors]
    )
    calibrated_offsets = quantize_numpy(
        np.maximum(0.0, offsets - shrinkage[:, None])
    )
    calibrated_minimum = (
        calibrated_offsets - drift @ normals.T
    ).min(axis=1)
    surface_uncalibrated = uncalibrated_minimum >= 0.0
    surface_accepted = (
        calibrated_minimum >= float(config["refresh_guard_band"])
    )
    mlp_threshold = np.asarray(
        [mlp_thresholds[behavior] for behavior in point_behaviors]
    )
    mlp_accepted = probabilities >= mlp_threshold
    reference_offsets = {}
    for point_index, reference_index in enumerate(
        dataset.reference_indices.tolist()
    ):
        reference_offsets.setdefault(
            dataset.reference_ids[reference_index], offsets[point_index]
        )
    reference_behaviors = {
        reference_id: int(
            dataset.reference_tensors["identifiers"][index, 0].item()
        )
        for index, reference_id in enumerate(dataset.reference_ids)
    }
    boundary = _boundary_audit(
        ROOT / config["artifact_root"] / "boundaries.jsonl.gz",
        config["partition"],
        reference_offsets,
        surface_shrinkages,
        reference_behaviors,
    )
    report = {
        "schema_version": 1,
        "phase": 8,
        "component": "geometry_adequacy_audit",
        "status": "PASS",
        "partition": config["partition"],
        "data": dataset.summary,
        "surface_uncalibrated": _classification(
            surface_uncalibrated, valid
        ),
        "surface_calibrated": _classification(surface_accepted, valid),
        "validity_mlp_calibrated": _classification(mlp_accepted, valid),
        "ranking": {
            "surface_auroc": float(
                roc_auc_score(valid, calibrated_minimum)
            ),
            "surface_auprc": float(
                average_precision_score(valid, calibrated_minimum)
            ),
            "validity_mlp_auroc": float(
                roc_auc_score(valid, probabilities)
            ),
            "validity_mlp_auprc": float(
                average_precision_score(valid, probabilities)
            ),
        },
        "surface_mlp_disagreement": {
            "count": int((surface_accepted != mlp_accepted).sum()),
            "fraction": float(
                (surface_accepted != mlp_accepted).mean()
            ),
        },
        "boundary_audit": boundary,
        "interpretation_flags": {
            "directional_nonconvexity_observed": (
                boundary["directional_reentry_nonconvex_evidence"] > 0
            ),
            "surface_target_met_on_calibration": all(
                selection["target_met"]
                for selection in surface_seed["selections"].values()
            ),
            "mlp_target_met_on_calibration": all(
                result["selection"]["target_met"]
                for result in mlp_seed["calibration"].values()
            ),
        },
        "test_partition_accessed": False,
        "config_sha256": sha256_file(CONFIG),
        "surface_checkpoint_sha256": sha256_file(surface_checkpoint),
        "surface_calibration_sha256": sha256_file(
            surface_calibration_path
        ),
        "validity_mlp_checkpoint_sha256": sha256_file(mlp_checkpoint),
        "validity_mlp_manifest_sha256": sha256_file(mlp_manifest_path),
        "git": state,
        "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "geometry_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
