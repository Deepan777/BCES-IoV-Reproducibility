#!/usr/bin/env python3
"""Evaluate registered cooperative-track corruptions on validation only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.corruptions import apply_corruption
from bces.models.calibration import quantize_numpy
from bces.models.dataset import SurfaceDataset
from bces.models.surfacenet import SurfaceNetLite, normal_tensor
from bces.models.validity_mlp import ValidityMLPLite
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/object_uncertainty_v1.yaml"


class CorruptedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        base: SurfaceDataset,
        name: str | None,
        magnitude: float = 0.0,
    ) -> None:
        self.base = base
        self.name = name
        self.magnitude = magnitude

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base[index]
        if self.name is None:
            return item
        reference_index = int(self.base.reference_indices[index])
        return apply_corruption(
            item,
            self.base.reference_ids[reference_index],
            self.name,
            self.magnitude,
        )


def _classification(accepted: np.ndarray, valid: np.ndarray) -> dict:
    unsafe = accepted & ~valid
    accepted_count = int(accepted.sum())
    return {
        "accepted": accepted_count,
        "unsafe_accepted": int(unsafe.sum()),
        "unsafe_accept_rate": (
            int(unsafe.sum()) / accepted_count if accepted_count else None
        ),
        "coverage": accepted_count / len(valid),
        "valid_coverage": int((accepted & valid).sum()) / int(valid.sum()),
    }


def _predict(
    surface: SurfaceNetLite,
    mlp: ValidityMLPLite,
    dataset: Dataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets, probabilities, drift = [], [], []
    surface.eval()
    mlp.eval()
    with torch.no_grad():
        for raw in DataLoader(
            dataset, batch_size=512, shuffle=False, num_workers=0
        ):
            batch = {key: value.to(device) for key, value in raw.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                predicted, _ = surface(batch)
                logits = mlp(batch)
            offsets.append(predicted.float().cpu().numpy().astype(np.float64))
            probabilities.append(
                torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64)
            )
            drift.append(
                batch["normalized_drift"].float().cpu().numpy().astype(np.float64)
            )
    return (
        np.concatenate(offsets),
        np.concatenate(probabilities),
        np.concatenate(drift),
    )


def _conditions(config: dict) -> list[tuple[str | None, float]]:
    result: list[tuple[str | None, float]] = [(None, 0.0)]
    for name, values in config["perturbations"].items():
        result.extend((name, float(value)) for value in values)
    return result


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("object_uncertainty_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("uncertainty stress requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("uncertainty stress requires CUDA")
    device = torch.device("cuda")
    base = SurfaceDataset(
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
    calibration_path = ROOT / config["surface_calibration"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    surface_seed = next(
        row
        for row in calibration["seed_results"]
        if int(row["seed"]) == int(config["surface_seed"])
    )
    shrinkages = {
        int(key): float(value["shrinkage"])
        for key, value in surface_seed["selections"].items()
    }
    mlp_manifest_path = (
        ROOT / config["validity_mlp_root"] / "run_manifest.json"
    )
    mlp_manifest = json.loads(
        mlp_manifest_path.read_text(encoding="utf-8")
    )
    mlp_seed = next(
        row
        for row in mlp_manifest["seeds"]
        if int(row["seed"]) == int(config["validity_mlp_seed"])
    )
    thresholds = {
        int(key): float(value["selection"]["threshold"])
        for key, value in mlp_seed["calibration"].items()
    }
    valid = base.valid.numpy().astype(bool)
    behaviors = (
        base.reference_tensors["identifiers"][base.reference_indices, 0]
        .numpy()
        .astype(int)
    )
    normal_array = normal_tensor().numpy().astype(np.float64)
    shrinkage = np.asarray([shrinkages[value] for value in behaviors])
    threshold = np.asarray([thresholds[value] for value in behaviors])
    results = []
    clean_surface = clean_mlp = None
    torch.cuda.reset_peak_memory_stats(device)
    for name, magnitude in _conditions(config):
        offsets, probabilities, drift = _predict(
            surface,
            mlp,
            CorruptedDataset(base, name, magnitude),
            device,
        )
        calibrated = quantize_numpy(
            np.maximum(0.0, offsets - shrinkage[:, None])
        )
        minimum = (calibrated - drift @ normal_array.T).min(axis=1)
        surface_accepted = (
            minimum >= float(config["refresh_guard_band"])
        )
        mlp_accepted = probabilities >= threshold
        if name is None:
            clean_surface = surface_accepted.copy()
            clean_mlp = mlp_accepted.copy()
        if clean_surface is None or clean_mlp is None:
            raise RuntimeError("clean condition must be evaluated first")
        results.append(
            {
                "condition": "clean" if name is None else name,
                "magnitude": magnitude,
                "surface": _classification(surface_accepted, valid),
                "validity_mlp": _classification(mlp_accepted, valid),
                "surface_decision_change_fraction": float(
                    (surface_accepted != clean_surface).mean()
                ),
                "validity_mlp_decision_change_fraction": float(
                    (mlp_accepted != clean_mlp).mean()
                ),
            }
        )
    report = {
        "schema_version": 1,
        "phase": 8,
        "component": "object_track_uncertainty",
        "status": "PASS",
        "partition": config["partition"],
        "data": base.summary,
        "corruption_semantics": config["corruption_semantics"],
        "conditions": results,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "test_partition_accessed": False,
        "config_sha256": sha256_file(CONFIG),
        "surface_checkpoint_sha256": sha256_file(surface_checkpoint),
        "surface_calibration_sha256": sha256_file(calibration_path),
        "validity_mlp_checkpoint_sha256": sha256_file(mlp_checkpoint),
        "validity_mlp_manifest_sha256": sha256_file(mlp_manifest_path),
        "git": state,
        "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "object_uncertainty.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "condition_count": len(results),
                "peak_cuda_allocated_bytes": report[
                    "peak_cuda_allocated_bytes"
                ],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
