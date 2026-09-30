#!/usr/bin/env python3
"""Gate the repaired Phase-4 sender inputs, labels, and oracle boundaries."""

from __future__ import annotations

import gzip
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.adequacy import evaluate_adequacy
from bces.data.workflow import load_verified_manifest
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

RUN = ROOT / "outputs" / "phase4" / "observational_v2"


def verified(path: Path, field: str) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path}")
    return payload, expected


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main() -> int:
    manifest, manifest_sha = verified(RUN / "run_manifest.json", "manifest_sha256")
    audit, _ = verified(RUN / "boundary_audit.json", "audit_sha256")
    scales, scales_sha = verified(ROOT / "data/manifests/drift_scales_v2.json", "scales_sha256")
    references = {row["reference_id"]: row for row in rows(RUN / "references.jsonl.gz")}
    points = list(rows(RUN / "validity_points.jsonl.gz"))
    boundaries = list(rows(RUN / "boundaries.jsonl.gz"))
    current_policy = KinematicPlanner(PlannerConfig.from_yaml(ROOT / "configs/planner/kinematic.yaml")).policy_hash
    artifact_paths = {name: RUN / f"{name}.jsonl.gz" for name in ("references", "validity_points", "boundaries")}
    artifact_hashes_pass = all(
        sha256_file(path) == manifest["artifact_sha256"][name]
        for name, path in artifact_paths.items()
    )
    input_contract_pass = all(
        row["input_contract"] == "exact_reference_payload_and_frozen_policy_query_v2"
        and row["frozen_input"]["generated_at_ms"] >= max(
            row["frozen_input"]["query_available_at_ms"],
            row["frozen_input"]["context_available_at_ms"],
            row["frozen_input"]["message_timestamp_ms"],
        )
        and len(row["frozen_input"]["batch"]["object_features"]) == 32
        and all(len(item) == 10 for item in row["frozen_input"]["batch"]["object_features"])
        and len(row["frozen_input"]["batch"]["reference_path"]) == 12
        and row["frozen_input"]["batch"]["identifiers"][0] == row["behavior_id"]
        and row["frozen_input"]["batch"]["identifiers"][2] == current_policy
        for row in references.values()
    )
    point_contract_pass = all(
        row["reference_id"] in references
        and row["split"] == references[row["reference_id"]]["split"]
        and row["behavior"] == references[row["reference_id"]]["behavior"]
        and len(row["normalized_drift"]) == 7
        and "features" not in row and "diagnostic_receiver_features" not in row
        and row["policy_hash"] == current_policy
        for row in points
    )
    by_reference: dict[str, set[int]] = defaultdict(set)
    transition_valid = True
    for row in boundaries:
        by_reference[row["reference_id"]].add(int(row["direction_index"]))
        if row["status"] == "transition":
            transition_valid &= (
                not row["censored"] and row["valid_radius"] is not None
                and row["invalid_radius"] is not None
                and 0 <= row["valid_radius"] < row["invalid_radius"]
                and row["invalid_radius"] - row["valid_radius"] <= 0.0100001
            )
    status_counts = Counter(row["status"] for row in boundaries)
    boundary_contract_pass = (
        all(reference_id in references and indices == set(range(16)) for reference_id, indices in by_reference.items())
        and len({row["boundary_id"] for row in boundaries}) == len(boundaries)
        and transition_valid and status_counts["transition"] >= 100
    )
    manifest_records = []
    for record in load_verified_manifest(ROOT / "data/manifests/v2xtraj_primary_manifest_v1.json"):
        item = record.__dict__.copy()
        item["source_role"] = record.source_role.value
        manifest_records.append(item)
    split, _ = verified(ROOT / "data/manifests/v2xtraj_primary_split_v1.json", "split_sha256")
    adequacy = evaluate_adequacy(
        manifest_records, split_assignments=split["assignments"],
        validity_decisions_by_split_behavior=manifest["validity_decisions_by_split_behavior"],
    )
    report_xml = ET.parse(ROOT / "outputs/phase4/phase4_test_report.xml").getroot()
    suites = [report_xml] if report_xml.tag == "testsuite" else list(report_xml.findall("testsuite"))
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    state = git_state(ROOT)
    checks = [
        {"name": "phase4_tests_pass", "passed": tests["tests"] >= 20 and tests["failures"] == tests["errors"] == tests["skipped"] == 0},
        {"name": "full_clean_run_and_artifact_hashes", "passed": manifest["run_scope"] == "full" and manifest["status"] == "PASS" and manifest["git"]["dirty"] is False and artifact_hashes_pass},
        {"name": "policy_source_and_config_binding", "passed": manifest["policy_hash"] == current_policy and all(row["policy_hash"] == current_policy for row in references.values())},
        {"name": "training_only_drift_scale_fit", "passed": scales["partition"] == "train" and scales["test_partition_accessed"] is False and scales["git"]["dirty"] is False and manifest["drift_scales_sha256"] == scales_sha},
        {"name": "frozen_sender_payload_query_inputs", "passed": bool(references) and input_contract_pass},
        {"name": "later_receiver_state_excluded_from_model_inputs", "passed": bool(points) and point_contract_pass},
        {"name": "oracle_bracketed_directional_boundaries", "passed": bool(boundaries) and boundary_contract_pass},
        {"name": "independent_100_boundary_source_reload_audit", "passed": audit["passed"] and audit["boundary_count"] == 100 and not audit["mismatches"] and audit["test_partition_accessed"] is False},
        {"name": "confirmatory_data_adequacy", "passed": adequacy.passes and adequacy.status.value == "confirmatory"},
        {"name": "test_not_used_for_development", "passed": manifest["test_partition_used_for_development"] is False},
        {"name": "gate_tree_clean", "passed": state["dirty"] is False},
    ]
    report = {
        "schema_version": 2, "phase": 4,
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks, "tests": tests, "adequacy": adequacy.to_dict(),
        "counts": manifest["counts"], "boundary_status_counts": manifest["boundary_status_counts"],
        "observational_manifest_sha256": manifest_sha, "git": state, "timestamp_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(ROOT / "outputs/phase4/phase4_gate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
