#!/usr/bin/env python3
"""Independently verify the frozen learned scalar-TTL baseline artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.dataset import SurfaceDataset
from bces.models.surfacenet import (
    ScalarTTLLite,
    SurfaceNetLite,
    trainable_parameter_count,
)
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "evaluation" / "scalar_ttl_v1.yaml"
RUN = ROOT / "outputs" / "phase8" / "scalar_ttl_v1"


def _validation_counts(
    model: ScalarTTLLite, dataset: SurfaceDataset, device: torch.device,
) -> tuple[int, int, int, int]:
    valid_accept = unsafe_accept = valid_total = invalid_total = 0
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                ttl, _ = model(batch)
            accepted = ttl >= batch["normalized_drift"][:, 6]
            valid = batch["valid"] > 0.5
            valid_accept += int((accepted & valid).sum())
            unsafe_accept += int((accepted & ~valid).sum())
            valid_total += int(valid.sum())
            invalid_total += int((~valid).sum())
    return valid_accept, unsafe_accept, valid_total, invalid_total


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    expected_manifest_hash = manifest.pop("manifest_sha256")
    validation = SurfaceDataset(ROOT / config["artifact_root"], "validation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hashes = state_only = validation_metrics = curve_contract = True
    total_checkpoint_bytes = 0
    target_cells_met = 0
    expected_grid_count = round(
        (float(config["calibration"]["grid_stop"]) - float(config["calibration"]["grid_start"]))
        / float(config["calibration"]["grid_step"])
    ) + 1
    for row in manifest["seeds"]:
        seed_dir = RUN / f"seed_{row['seed']}"
        best_path, final_path = seed_dir / "best.pt", seed_dir / "final.pt"
        checkpoint_hashes &= sha256_file(best_path) == row["best_checkpoint_sha256"]
        checkpoint_hashes &= sha256_file(final_path) == row["final_checkpoint_sha256"]
        total_checkpoint_bytes += best_path.stat().st_size + final_path.stat().st_size
        for path in (best_path, final_path):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state_only &= "model_state_dict" in checkpoint and "optimizer_state_dict" not in checkpoint
        model = ScalarTTLLite().to(device)
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)["model_state_dict"]
        )
        valid_accept, unsafe_accept, valid_total, invalid_total = _validation_counts(
            model, validation, device
        )
        accepted = valid_accept + unsafe_accept
        metrics = row["best_metrics"]
        validation_metrics &= accepted == int(metrics["accepted_count"])
        validation_metrics &= abs(unsafe_accept / accepted - metrics["unsafe_accept_rate"]) < 1e-12
        validation_metrics &= abs(valid_accept / valid_total - metrics["valid_coverage"]) < 1e-12
        validation_metrics &= abs(
            unsafe_accept / invalid_total - metrics["invalid_accept_fraction"]
        ) < 1e-12
        target_cells_met += int(row["calibration_targets_met"])
        curve_contract &= set(map(int, row["calibration"])) == set(range(5))
        for result in row["calibration"].values():
            curve_contract &= len(result["curve"]) == expected_grid_count
            curve_contract &= any(
                point["shrinkage"] == result["selection"]["shrinkage"]
                and point["unsafe_accept_upper"] == result["selection"]["unsafe_accept_upper"]
                for point in result["curve"]
            )
    checks = [
        {
            "name": "manifest_and_config_hashes",
            "passed": canonical_json_hash(manifest) == expected_manifest_hash
            and manifest["config_sha256"] == sha256_file(CONFIG),
        },
        {
            "name": "equal_feature_parameter_budget",
            "passed": manifest["parameter_count"] == trainable_parameter_count(ScalarTTLLite())
            <= trainable_parameter_count(SurfaceNetLite()) <= int(config["max_parameters"]),
        },
        {
            "name": "five_registered_seeds",
            "passed": [row["seed"] for row in manifest["seeds"]] == list(config["seeds"]),
        },
        {
            "name": "independent_exact_uar_recomputation",
            "passed": validation_metrics and config["uar_denominator"] == "all_accepted_decisions",
        },
        {
            "name": "nondegenerate_checkpoint_selection",
            "passed": all(
                row["best_metrics"]["accepted_count"] >= int(config["minimum_validation_accepted"])
                for row in manifest["seeds"]
            ),
        },
        {
            "name": "checkpoint_hashes_sizes_and_state_only",
            "passed": checkpoint_hashes and state_only
            and total_checkpoint_bytes == manifest["checkpoint_bytes"]
            <= int(config["max_checkpoint_bytes"]),
        },
        {
            "name": "calibration_after_all_training",
            "passed": manifest["calibration_opened_after_all_training"]
            and config["calibration_partition_access"] == "calibration_only_after_training",
        },
        {
            "name": "behaviorwise_clustered_curve_contract",
            "passed": curve_contract
            and all(
                result["cluster_count"] > 1
                for row in manifest["seeds"]
                for result in row["calibration"].values()
            ),
        },
        {
            "name": "test_partition_locked",
            "passed": not manifest["test_partition_accessed"]
            and manifest["validation_data"]["other_split_records_deserialized"] == 0
            and manifest["calibration_data"]["other_split_records_deserialized"] == 0,
        },
        {
            "name": "resource_caps",
            "passed": 0 < manifest["peak_cuda_allocated_bytes"] <= int(config["max_peak_cuda_bytes"]),
        },
        {"name": "clean_generation_revision", "passed": manifest["git"]["dirty"] is False},
    ]
    report = {
        "schema_version": 1,
        "phase": 8,
        "component": "learned_scalar_ttl",
        "status": "PASS" if all(check["passed"] for check in checks) else "FAIL",
        "checks": checks,
        "primary_seed": manifest["primary_seed"],
        "target_cells_met": target_cells_met,
        "target_cells_total": len(manifest["seeds"]) * 5,
        "negative_scientific_result": target_cells_met == 0,
        "manifest_sha256": expected_manifest_hash,
        "git": git_state(ROOT),
        "timestamp_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(ROOT / "outputs" / "phase8" / "scalar_ttl_gate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
