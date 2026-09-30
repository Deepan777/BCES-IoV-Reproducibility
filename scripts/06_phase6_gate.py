#!/usr/bin/env python3
"""Verify immutable Phase-6 checkpoints, resource caps, and leakage guards."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.surfacenet import SurfaceNetLite, trainable_parameter_count
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "training" / "surfacenet_lite_v2.yaml"
RUN = ROOT / "outputs" / "phase6" / "surfacenet_v2"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256")
    manifest_ok = canonical_json_hash(manifest) == expected
    xml = ET.parse(ROOT / "outputs" / "phase6" / "phase6_test_report.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    checkpoint_hashes = True
    state_only = True
    for row in manifest["seeds"]:
        seed_dir = RUN / f"seed_{row['seed']}"
        best, final = seed_dir / "best.pt", seed_dir / "final.pt"
        checkpoint_hashes &= sha256_file(best) == row["best_checkpoint_sha256"]
        checkpoint_hashes &= sha256_file(final) == row["final_checkpoint_sha256"]
        for path in (best, final):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state_only &= "model_state_dict" in payload and "optimizer_state_dict" not in payload
    checks = [
        {"name": "phase6_tests_pass", "passed": tests["tests"] >= 5 and tests["failures"] == tests["errors"] == tests["skipped"] == 0},
        {"name": "manifest_and_config_hashes", "passed": manifest_ok and manifest["config_sha256"] == sha256_file(CONFIG)},
        {"name": "registered_parameter_limit", "passed": manifest["parameter_count"] == trainable_parameter_count(SurfaceNetLite()) <= int(config["max_parameters"])},
        {"name": "toy_overfit_gate", "passed": manifest["toy_overfit"]["passed"]},
        {"name": "five_sequential_seeds", "passed": [row["seed"] for row in manifest["seeds"]] == list(config["seeds"]) and len(manifest["seeds"]) == 5},
        {"name": "registered_uar_denominator", "passed": config["uar_denominator"] == "all_accepted_decisions" and all("unsafe_accept_rate" in row["best_metrics"] and "invalid_accept_fraction" in row["best_metrics"] for row in manifest["seeds"])},
        {"name": "nondegenerate_checkpoint_selection", "passed": all(row["best_metrics"]["accepted_count"] >= int(config["minimum_validation_accepted"]) for row in manifest["seeds"])},
        {"name": "finite_training_histories", "passed": all(not row["nan_or_inf_detected"] and row["epochs"] <= int(config["max_epochs"]) for row in manifest["seeds"])},
        {"name": "checkpoint_hashes_and_state_only", "passed": checkpoint_hashes and state_only},
        {"name": "checkpoint_allocation", "passed": manifest["checkpoint_bytes"] <= int(config["max_checkpoint_bytes"])},
        {"name": "measured_cuda_cap", "passed": 0 < manifest["peak_cuda_allocated_bytes"] <= int(config["max_peak_cuda_bytes"])},
        {"name": "calibration_and_test_locked", "passed": not manifest["calibration_partition_accessed"] and not manifest["test_partition_accessed"] and manifest["train_data"]["other_split_records_deserialized"] == manifest["validation_data"]["other_split_records_deserialized"] == 0},
        {"name": "clean_generation_revision", "passed": manifest["git"]["dirty"] is False},
    ]
    report = {
        "schema_version": 1, "phase": 6,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks": checks, "tests": tests,
        "parameter_count": manifest["parameter_count"],
        "peak_cuda_allocated_bytes": manifest["peak_cuda_allocated_bytes"],
        "checkpoint_bytes": manifest["checkpoint_bytes"],
        "git": git_state(ROOT), "timestamp_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(ROOT / "outputs" / "phase6" / "phase6_gate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
