#!/usr/bin/env python3
"""Verify Phase-5 SUMO same-world artifacts and tests."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.oracle.validity import ValidityThresholds
from bces.simulation.branching import recompute_pair_label
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

RUN = ROOT / "outputs" / "phase5" / "sumo_v3"


def _verified(path: Path, field: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path}")
    return payload


def main() -> int:
    manifest = _verified(RUN / "run_manifest.json", "manifest_sha256")
    audit = _verified(RUN / "audit.json", "audit_sha256")
    pairs_path = RUN / "pairs.jsonl"
    rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines()]
    config_path = ROOT / "configs" / "simulation" / "sumo_v3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values = config["label_thresholds"]
    thresholds = ValidityThresholds(values["max_cost_regret"], values["max_cached_risk"], values["max_trajectory_deviation_m"])
    recomputed = [recompute_pair_label(row, thresholds) for row in rows]
    xml = ET.parse(ROOT / "outputs" / "phase5" / "phase5_test_report_v3.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    forbidden_media = [path.as_posix() for path in RUN.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".avi"}]
    expected_splits = {str(key): int(value) for key, value in config["partitions"].items()}
    frozen_inputs = [row["reference"]["frozen_input"] for row in rows]
    frozen_contract_ok = all(
        row["reference"]["input_contract"] == "exact_reference_payload_and_frozen_policy_query_v2"
        and item["generated_at_ms"] == item["query_available_at_ms"] == item["context_available_at_ms"]
        and item["message_timestamp_ms"] <= item["generated_at_ms"]
        and item["query_provenance"] == "frozen_reference_policy"
        and len(item["batch"]["object_features"]) == 32
        and len(item["batch"]["object_mask"]) == 32
        and len(item["batch"]["reference_path"]) == 12
        and canonical_json_hash(item["batch"]) == item["feature_sha256"]
        for row, item in zip(rows, frozen_inputs)
    )
    drift_ok = all(
        len(tick["normalized_drift"]) == 7
        for row in rows for branch in (row["fresh"], row["cached"])
        for tick in branch["ticks"]
    )
    checks = [
        {"name": "phase5_tests_pass", "passed": tests["tests"] >= 3 and tests["failures"] == tests["errors"] == tests["skipped"] == 0},
        {"name": "pinned_sumo_and_artifact_hashes", "passed": manifest["sumo_version"] == config["sumo_version"] and sha256_file(pairs_path) == manifest["pairs_sha256"]},
        {"name": "six_registered_families_present", "passed": set(audit["family_counts"]) == set(config["families"])},
        {"name": "one_hundred_state_equivalent_pairs", "passed": len(rows) == audit["pair_count"] == audit["state_equivalent_count"] == 100 and all(row["state_equivalent"] for row in rows)},
        {"name": "deterministic_frozen_partitions", "passed": audit["split_counts"] == expected_splits and manifest["split_counts"] == expected_splits},
        {"name": "corrected_policy_bound_to_behavior", "passed": all(row["policy_hash"] == manifest["policy_hash"] and row["reference"]["frozen_input"]["batch"]["identifiers"][2] == manifest["policy_hash"] for row in rows)},
        {"name": "exact_frozen_reference_input_contract", "passed": frozen_contract_ok and audit["input_contract_count"] == 100},
        {"name": "seven_dimensional_training_scale_drift", "passed": drift_ok and bool(manifest["drift_scales_sha256"])},
        {"name": "simulation_sources_bound_to_manifest", "passed": manifest["branching_source_sha256"] == sha256_file(ROOT / "bces" / "simulation" / "branching.py") and manifest["scenario_builder_source_sha256"] == sha256_file(ROOT / "bces" / "simulation" / "scenario_builder.py")},
        {"name": "development_class_adequacy_without_test_tuning", "passed": audit["development_class_counts"]["invalid"] >= int(config["development_adequacy"]["minimum_non_test_invalid_pairs"]) and audit["training_class_counts"]["valid"] >= int(config["development_adequacy"]["minimum_train_valid_pairs"]) and audit["training_class_counts"]["invalid"] >= int(config["development_adequacy"]["minimum_train_invalid_pairs"]) and not audit["test_labels_used_for_adequacy"]},
        {"name": "every_label_independently_recomputes", "passed": not audit["label_recomputation_mismatches"] and all(bool(row["valid"]) == value for row, value in zip(rows, recomputed))},
        {"name": "compact_tick_logs_are_complete", "passed": all(len(row["fresh"]["ticks"]) == len(row["cached"]["ticks"]) == 15 for row in rows)},
        {"name": "no_rendered_media", "passed": manifest["rendered_frames"] == 0 and not forbidden_media},
    ]
    report = {
        "schema_version": 3, "phase": 5,
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks, "tests": tests, "pair_count": len(rows),
        "valid_count": sum(row["valid"] for row in rows),
        "invalid_count": sum(not row["valid"] for row in rows),
        "git": git_state(ROOT), "timestamp_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(ROOT / "outputs" / "phase5" / "phase5_gate_report_v3.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
