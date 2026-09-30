#!/usr/bin/env python3
"""Derive the amended Phase-3 engineering gate from reproducible evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.adequacy import (
    MIN_BEHAVIORS, MIN_INTERSECTIONS, MIN_SCENARIOS, MIN_TEST_INTERSECTIONS,
    MIN_VALID_WINDOWS, MIN_VALIDITY_DECISIONS_PER_SPLIT_BEHAVIOR,
)
from bces.data.datasets import dataset_profile
from bces.data.manifest import FORBIDDEN_MODALITY_TOKENS, load_catalog
from bces.data.multidataset import load_multidataset_plan
from bces.data.workflow import load_verified_manifest
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash_without(payload: dict[str, Any], field: str) -> str:
    return canonical_json_hash({key: value for key, value in payload.items() if key != field})


def _numbered_paths(prefix: str) -> dict[int, Path]:
    manifests = ROOT / "data" / "manifests"
    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.json$")
    return {
        int(match.group(1)): path
        for path in manifests.glob(f"{prefix}_v*.json")
        if (match := pattern.match(path.name))
    }


def _verified_numbered_history(
    maximum_transfer_bytes: int,
) -> tuple[bool, bool, int, dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    plan_paths = _numbered_paths("v2xtraj_tranche")
    receipt_paths = _numbered_paths("v2xtraj_download_receipt")
    numbers = sorted(receipt_paths)
    contiguous = numbers == list(range(1, len(numbers) + 1))
    complete_set = bool(numbers) and set(plan_paths) == set(receipt_paths)
    verified = contiguous and complete_set
    continuation_verified = verified
    plans: dict[int, dict[str, Any]] = {}
    receipts: dict[int, dict[str, Any]] = {}
    seen_scenarios: set[str] = set()
    for number in numbers:
        plan = _json(plan_paths[number])
        receipt = _json(receipt_paths[number])
        plan_hash = plan.get("tranche_sha256")
        scenario_ids = {str(item["scenario_id"]) for item in plan.get("scenarios", ())}
        verified &= (
            plan_hash == _hash_without(plan, "tranche_sha256")
            and int(plan.get("tranche_number", 1 if number == 1 else -1)) == number
            and receipt.get("receipt_sha256") == _hash_without(receipt, "receipt_sha256")
            and int(receipt.get("tranche_number", 1 if number == 1 else -1)) == number
            and receipt.get("tranche_sha256") == plan_hash
            and receipt.get("status") == "complete"
            and receipt.get("scenario_count") == len(scenario_ids)
            and receipt.get("transfer_bytes", maximum_transfer_bytes + 1)
            <= maximum_transfer_bytes
            and not (seen_scenarios & scenario_ids)
        )
        seen_scenarios |= scenario_ids
        if number >= 2:
            continuation_path = (
                ROOT
                / "data"
                / "manifests"
                / f"v2xtraj_continuation_review_v{number - 1}.json"
            )
            if not continuation_path.is_file():
                continuation_verified = False
            else:
                continuation = _json(continuation_path)
                continuation_verified &= (
                    continuation.get("continuation_review_sha256")
                    == _hash_without(continuation, "continuation_review_sha256")
                    and continuation.get("source_plan_sha256") == plan_hash
                    and continuation.get("decision")
                    == "approved_for_one_additional_review_tranche"
                    and continuation.get("phase4_remains_prohibited") is True
                )
        plans[number] = plan
        receipts[number] = receipt
    return verified, continuation_verified, len(seen_scenarios), plans, receipts


def _test_evidence(path: Path) -> tuple[dict[str, int], set[str]]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    return summary, {case.attrib.get("name", "") for case in root.iter("testcase")}


def _payload_files() -> list[str]:
    paths = []
    for directory in (ROOT / "data" / "raw", ROOT / "data" / "processed"):
        if directory.exists():
            paths.extend(path.relative_to(ROOT).as_posix() for path in directory.rglob("*") if path.is_file() and path.name != ".gitkeep")
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-report", type=Path, default=ROOT / "outputs" / "phase3" / "phase3_test_report.xml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "phase3" / "phase3_gate_report.json")
    args = parser.parse_args()
    plan = load_multidataset_plan(ROOT / "configs" / "data" / "multi_dataset_plan.yaml")
    aggregate = _json(ROOT / "outputs" / "phase3" / "multidataset_adequacy_report.json")
    tests, test_names = _test_evidence(args.test_report)
    catalogs_match = True
    manifests_verify = True
    blocked_entries_empty = True
    allowlisted_paths: set[str] = set()
    for item in plan.datasets:
        catalog = load_catalog(ROOT / item.catalog)
        manifest_path = ROOT / "data" / "manifests" / f"{item.dataset_id}_manifest_v1.json"
        manifest = _json(manifest_path)
        catalogs_match &= catalog.dataset_id == item.dataset_id and catalog.evidence_role == item.evidence_role.value and catalog.sha256 == manifest.get("catalog_sha256")
        manifests_verify &= manifest.get("manifest_sha256") == _hash_without(manifest, "manifest_sha256") and manifest.get("dataset_id") == item.dataset_id and manifest.get("evidence_role") == item.evidence_role.value and len(load_verified_manifest(manifest_path)) == len(manifest.get("records", ()))
        blocked_entries_empty &= not (catalog.status.startswith("blocked") and catalog.entries)
        allowlisted_paths |= {f"data/raw/{item.dataset_id}/{entry.destination}" for entry in catalog.entries}
    probes = [_json(ROOT / "data" / "manifests" / name) for name in ("v2xtraj_source_probe_v1.json", "tumtraf_v2x_source_probe_v1.json", "interaction_source_probe_v1.json")]
    fetch = _json(ROOT / "outputs" / "phase3" / "fetch_decision.json")
    window_review = _json(ROOT / "data" / "manifests" / "v2xtraj_window_review_v1.json")
    review_summary = window_review.get("summary", {})
    budget_values = yaml.safe_load((ROOT / "configs" / "budget.yaml").read_text(encoding="utf-8")) or {}
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data", **budget_values)
    budget_report = build_budget_report(budget)
    history_verified, continuations_verified, history_scenarios, tranches, receipts = (
        _verified_numbered_history(budget.max_download_tranche_bytes)
    )
    initial_receipt = receipts.get(1, {})
    payload_files = _payload_files()
    forbidden = [path for path in payload_files if any(token in path.lower() for token in FORBIDDEN_MODALITY_TOKENS)]
    checks = [
        {"name": "phase3_tests_pass", "passed": tests["tests"] >= 35 and tests["failures"] == tests["errors"] == tests["skipped"] == 0},
        {"name": "scientific_amendment_is_registered", "passed": (ROOT / "docs" / "SCIENTIFIC_AMENDMENT_001.md").is_file() and plan.amendment_id == "SCIENTIFIC_AMENDMENT_001"},
        {"name": "exactly_one_primary_dataset", "passed": plan.primary_dataset_id == "v2xtraj_primary" and dataset_profile(plan.primary_dataset_id).may_unlock_phase4},
        {"name": "catalogs_match_frozen_evidence_roles", "passed": catalogs_match},
        {"name": "isolated_manifests_and_hashes_verify", "passed": manifests_verify},
        {"name": "primary_only_phase4_gate", "passed": aggregate.get("gate_rule") == "primary_dataset_only" and aggregate.get("primary_dataset_id") == "v2xtraj_primary" and aggregate.get("phase4_permitted") == aggregate.get("datasets", {}).get("v2xtraj_primary", {}).get("passes") and all(not details.get("may_unlock_phase4") for key, details in aggregate.get("datasets", {}).items() if key != "v2xtraj_primary")},
        {"name": "aggregate_report_hash_verifies", "passed": aggregate.get("report_sha256") == _hash_without(aggregate, "report_sha256")},
        {"name": "network_impairments_have_explicit_provenance", "passed": aggregate.get("network_impairments") == "replayed_or_simulated_not_observed_packet_logs"},
        {"name": "auxiliary_metadata_probes_consumed_no_payload", "passed": all(probe.get("payload_bytes_downloaded") == 0 for probe in probes[1:])},
        {"name": "selective_primary_tranche_is_complete_and_verified", "passed": history_verified and initial_receipt.get("scenario_count") == 100 and initial_receipt.get("intersection_count") == 8 and initial_receipt.get("extracted_bytes", budget.initial_data_review_bytes + 1) <= budget.initial_data_review_bytes and probes[0].get("payload_bytes_downloaded") == initial_receipt.get("transfer_bytes")},
        {"name": "all_post_checkpoint_continuations_are_bound_and_verified", "passed": len(tranches) >= 2 and continuations_verified},
        {"name": "window_review_is_verified_and_not_mislabeled_as_oracle_evidence", "passed": window_review.get("review_sha256") == _hash_without(window_review, "review_sha256") and review_summary.get("reviewed_scenarios") == history_scenarios and review_summary.get("valid_windows") == aggregate.get("artifacts", {}).get("v2xtraj_primary", {}).get("adequacy", {}).get("counts", {}).get("valid_windows") and review_summary.get("oracle_validity_decisions") == 0 and aggregate.get("artifacts", {}).get("v2xtraj_primary", {}).get("adequacy", {}).get("validity_decisions_by_split_behavior") == {}},
        {"name": "all_payload_is_allowlisted_and_no_forbidden_modality", "passed": set(payload_files) <= allowlisted_paths and not forbidden},
        {"name": "blocked_catalogs_have_no_speculative_files", "passed": blocked_entries_empty},
        {"name": "registered_thresholds_are_exact", "passed": (MIN_VALID_WINDOWS, MIN_SCENARIOS, MIN_INTERSECTIONS, MIN_BEHAVIORS, MIN_TEST_INTERSECTIONS, MIN_VALIDITY_DECISIONS_PER_SPLIT_BEHAVIOR) == (500, 30, 8, 3, 2, 50)},
        {"name": "leakage_and_primary_gate_tests_executed", "passed": "test_intersection_split_is_deterministic_and_disjoint" in test_names and "test_scenario_and_final_test_leakage_guards" in test_names and "test_auxiliary_success_cannot_unlock_phase4" in test_names},
        {"name": "workspace_budget_and_output_reserve_pass", "passed": budget_report["status"] in {"pass", "warning"} and budget_report["measurements"]["output_reserve_preserved"]},
        {"name": "blocker_and_data_policy_are_documented", "passed": (ROOT / "docs" / "BLOCKED.md").is_file() and (ROOT / "docs" / "DATA.md").is_file()},
    ]
    status = "PASS" if all(item["passed"] for item in checks) else "FAIL"
    report = {"schema_version": 2, "timestamp_utc": utc_now(), "phase": 3, "amendment_id": plan.amendment_id, "status": status, "data_adequacy_classification": aggregate.get("primary_adequacy_status"), "phase4_permitted": aggregate.get("phase4_permitted"), "checks": checks, "tests": tests, "payload_files": payload_files, "forbidden_payload_files": forbidden, "plan_sha256": plan.sha256, "aggregate_report_sha256": aggregate.get("report_sha256"), "budget": budget_report, "git": git_state(ROOT)}
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
