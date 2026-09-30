#!/usr/bin/env python3
"""Verify Phase-10 statistics, uncertainty, and negative-result retention."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

STATISTICS = ROOT / "outputs/phase10/statistics_v1/statistics.json"
CONFIG = ROOT / "configs/evaluation/statistics_v1.yaml"
OUTPUT = ROOT / "outputs/phase10/phase10_gate.json"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("Phase-10 gate output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Phase-10 gate requires a clean committed revision")
    report = json.loads(STATISTICS.read_text(encoding="utf-8"))
    surface_scalar = report["offline_matched_coverage"]["comparisons"]["surface_minus_learned_scalar_ttl"]
    intervals = [surface_scalar["clustered_uar_difference"]]
    for condition in report["sumo_paired_comparisons"].values():
        for comparison in condition.values():
            intervals.extend(endpoint["bootstrap_ci"] for endpoint in comparison.values())
    checks = {
        "statistics_status_pass": report["status"] == "PASS",
        "all_intervals_rebuildable": bool(intervals) and all(row["replicates"] == 10000 and row["confidence"] == 0.95 for row in intervals),
        "paired_tests_present": all("paired_unsafe_contribution" in row for row in report["offline_matched_coverage"]["comparisons"].values()),
        "holm_families_complete": set(report["holm_adjusted_p_values"]) == {"offline_selective_risk", "sumo_communication", "sumo_driving"},
        "scenario_seed_units": report["offline_unit"] == "scenario_or_intersection_cluster" and report["sumo_unit"] == "matched_seed",
        "null_and_negative_retained": not report["conclusions"]["surface_uar_improvement_vs_scalar_ci_below_zero"] and not report["conclusions"]["closed_loop_safety_difference_identifiable"] and not report["conclusions"]["surface_communication_reduction_vs_fresh_ci_below_zero"],
        "q1_claim_rejected": not report["conclusions"]["q1_strength_conditions_met"] and not report["scientific_success"],
    }
    gate = {
        "schema_version": 1, "phase": 10, "status": "PASS" if all(checks.values()) else "FAIL",
        "engineering_complete": all(checks.values()), "scientific_success": False,
        "checks": checks, "statistics_sha256": sha256_file(STATISTICS),
        "config_sha256": sha256_file(CONFIG), "git": state, "completed_utc": utc_now(),
    }
    gate["report_sha256"] = canonical_json_hash(gate)
    write_json_atomic(OUTPUT, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

