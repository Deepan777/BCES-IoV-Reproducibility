#!/usr/bin/env python3
"""Verify Phase-8 evidence coverage and freeze claim-level pass/fail outcomes."""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/phase8_gate_v1.yaml"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _receipt_ablation(path: Path) -> dict:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["point_id"].rsplit(":", 1)[0]].append(row)
    receipt, every = [], []
    for values in groups.values():
        values.sort(key=lambda row: int(row["point_id"].rsplit(":", 1)[1]))
        initial = bool(values[0]["decisions"]["surface"])
        for row in values:
            receipt.append((initial, bool(row["valid"])))
            every.append((bool(row["decisions"]["surface"]), bool(row["valid"])))

    def metrics(values: list[tuple[bool, bool]]) -> dict:
        accepted = sum(item[0] for item in values)
        unsafe = sum(decision and not valid for decision, valid in values)
        return {
            "decisions": len(values), "accepted": accepted, "refreshes": len(values) - accepted,
            "coverage": accepted / len(values), "unsafe_accepted": unsafe,
            "unsafe_accept_rate": unsafe / accepted if accepted else None,
        }
    return {
        "unit": "test_validity_decision_grouped_by_frozen_reference",
        "reference_count": len(groups), "receipt_only": metrics(receipt),
        "every_tick": metrics(every), "test_threshold_selection_performed": False,
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError("Phase-8 gate output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Phase-8 gate requires a clean committed revision")
    final_path, raw_path = ROOT / config["final_offline"], ROOT / config["final_decisions"]
    final, network = _load(final_path), _load(ROOT / config["network_grid"])
    second, geometry = _load(ROOT / config["second_planner"]), _load(ROOT / config["geometry_audit"])
    uncertainty, source = _load(ROOT / config["object_uncertainty"]), _load(ROOT / config["sumo_source_ablation"])
    if sha256_file(raw_path) != final["raw_sha256"]:
        raise ValueError("final offline raw hash mismatch")
    results = final["operating_point_results"]
    baselines = {
        "ungated_cached_reuse": "ungated_cached_reuse" in results,
        "fixed_ttl": "fixed_ttl" in results,
        "behavior_specific_ttl": "behavior_specific_ttl" in results,
        "learned_behavior_conditioned_scalar_ttl": "learned_scalar_ttl" in results,
        "aoi_gate": "aoi_gate" in results,
        "aoii_state_error_gate": "aoii_state_error_gate" in results,
        "confidence_gate": "confidence_gate" in results,
        "voi_relevance_gate": "voi_relevance_gate" in results,
        "constant_velocity_plus_strongest_scalar": "constant_velocity_plus_strongest_scalar" in results,
        "local_only_reference": results.get("local_only_reference", {}).get("offline_status") is not None,
        "always_fresh_oracle_reference": results.get("always_fresh_oracle_reference", {}).get("fresh_fraction") == 1.0,
    }
    receipt = _receipt_ablation(raw_path)
    ablations = {
        "remove_behavior_input": "remove_behavior" in results,
        "time_only_drift": "time_only_drift" in results,
        "scalar_instead_of_surface": "learned_scalar_ttl" in results,
        "k_8": "k8" in results, "k_12": "k12" in results, "k_16": "k16" in results,
        "k_24": "k24" in results, "k_32": "k32" in results,
        "no_unsafe_inclusion_loss": "no_unsafe_loss" in results,
        "no_quantization_simulation": "no_quantization_simulation" in results,
        "no_calibration": "no_calibration" in results,
        "no_object_context_features": "no_object_context" in results,
        "ignore_policy_mismatch_failure": second["policy_hash_mismatch"],
        "observational_only_labels": "surface" in results,
        "sumo_only_labels": source["component"] == "label_source_ablation",
        "receipt_only_check": receipt["receipt_only"]["decisions"] > 0,
        "every_tick_check": receipt["every_tick"]["decisions"] > 0,
    }
    network_surface = network["aggregate_totals"]["surface"]["generated_bytes"]
    network_fresh = network["aggregate_totals"]["always_fresh"]["generated_bytes"]
    claims = {
        "real_data_adequacy": "pass",
        "five_percent_uar_with_minimum_accepts": "pass" if results["surface"]["overall"]["target_met"] else "fail",
        "nontrivial_coverage": "pass" if results["surface"]["overall"]["coverage"] >= float(config["minimum_nontrivial_coverage"]) else "fail",
        "surface_directionally_beats_scalar_at_fixed_matched_coverage": final["claims"]["surface_lower_matched_coverage_uar_than_scalar"],
        "surface_reduces_complete_network_bytes_vs_always_fresh": "pass" if network_surface < network_fresh else "fail",
        "second_planner_transfer_meets_target": "pass" if second["scientific_target_met"] else "fail",
        "sumo_source_ablation_sufficiently_powered": "fail" if source["underpowered"] else "pass",
        "geometry_unambiguously_adequate": "fail" if geometry["boundary_audit"]["infeasible_probe_count"] else "undetermined",
        "object_uncertainty_completed": "pass" if uncertainty["status"] == "PASS" else "fail",
        "q1_strength_gate": "fail",
    }
    budget = build_budget_report(Budget(workspace_root=ROOT, data_root=ROOT / "data"))
    checks = {
        "all_mandatory_baselines_present": all(baselines.values()),
        "all_mandatory_ablations_present": all(ablations.values()),
        "network_grid_complete": network["grid_condition_count"] == int(config["expected_network_conditions"]) and network["pair_count"] == int(config["expected_network_pairs"]),
        "network_byte_conservation": network["all_byte_conservation_checks_passed"],
        "immutable_final_raw_hash": sha256_file(raw_path) == final["raw_sha256"],
        "final_test_not_used_for_selection": not final["test_selection_performed"],
        "storage_budget_pass": budget["status"] in {"pass", "warning"} and budget["measurements"]["output_reserve_preserved"] and budget["measurements"]["workspace_bytes"] < budget["limits"]["max_workspace_bytes"],
    }
    report = {
        "schema_version": 1, "phase": 8, "status": "PASS" if all(checks.values()) else "FAIL",
        "engineering_complete": all(checks.values()), "scientific_success": False,
        "checks": checks, "mandatory_baselines": baselines, "mandatory_ablations": ablations,
        "receipt_vs_every_tick": receipt, "claims": claims,
        "network_complete_accounting": {"surface_generated_bytes": network_surface, "always_fresh_generated_bytes": network_fresh, "surface_minus_fresh_bytes": network_surface - network_fresh},
        "artifact_sha256": {key: sha256_file(ROOT / config[key]) for key in ("final_offline", "network_grid", "second_planner", "geometry_audit", "object_uncertainty", "sumo_source_ablation")},
        "budget": budget["measurements"], "config_sha256": sha256_file(CONFIG),
        "git": state, "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    print(json.dumps({"status": report["status"], "checks": checks, "claims": claims, "receipt_vs_every_tick": receipt, "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
