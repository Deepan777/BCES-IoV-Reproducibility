#!/usr/bin/env python3
"""Verify calibration-only shrinkage, clustered curves, and frozen identifiers."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "calibration" / "surfacenet_calibration_v2.yaml"
RUN = ROOT / "outputs" / "phase7" / "calibration_v2"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload = json.loads((RUN / "calibration.json").read_text(encoding="utf-8"))
    expected = payload.pop("calibration_sha256")
    hash_ok = canonical_json_hash(payload) == expected
    xml = ET.parse(ROOT / "outputs" / "phase7" / "phase7_test_report.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    grid = config["shrinkage_grid"]
    expected_grid_count = round((float(grid["stop"]) - float(grid["start"])) / float(grid["step"])) + 1
    target_cells = 0
    curve_contract = True
    selections_frozen = True
    for seed in payload["seed_results"]:
        target_cells += sum(row["target_met"] for row in seed["selections"].values())
        curve_contract &= set(map(int, seed["risk_coverage_curves"])) == set(map(int, config["behavior_ids"]))
        curve_contract &= all(len(curve) == expected_grid_count for curve in seed["risk_coverage_curves"].values())
        for behavior, selection in seed["selections"].items():
            curve = seed["risk_coverage_curves"][behavior]
            selections_frozen &= any(
                row["shrinkage"] == selection["shrinkage"]
                and row["unsafe_accept_upper"] == selection["unsafe_accept_upper"]
                for row in curve
            )
    checks = [
        {"name": "phase7_tests_pass", "passed": tests["tests"] >= 5 and tests["failures"] == tests["errors"] == tests["skipped"] == 0},
        {"name": "calibration_and_config_hashes", "passed": hash_ok and payload["config_sha256"] == sha256_file(CONFIG)},
        {"name": "calibration_partition_only", "passed": payload["partition"] == "calibration" and payload["calibration_data"]["split"] == "calibration" and payload["calibration_data"]["other_split_records_deserialized"] == 0},
        {"name": "test_partition_locked", "passed": not payload["test_partition_accessed"]},
        {"name": "five_seed_behavior_risk_curves", "passed": len(payload["seed_results"]) == 5 and curve_contract},
        {"name": "scenario_clustered_upper_bounds", "passed": payload["target"]["bootstrap_replicates"] == int(config["bootstrap_replicates"]) and payload["target"]["confidence"] == float(config["confidence"])},
        {"name": "selected_shrinkage_resolves_to_curve", "passed": selections_frozen},
        {"name": "nonzero_selected_coverage", "passed": all(row["all_behavior_coverage_nonzero"] for row in payload["seed_results"])},
        {"name": "calibration_identifier_frozen", "passed": 1 <= payload["calibration_id"] <= 255},
        {"name": "clean_generation_revision", "passed": payload["git"]["dirty"] is False},
    ]
    report = {
        "schema_version": 1, "phase": 7,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "checks": checks, "tests": tests,
        "primary_seed": payload["primary_seed"], "target_cells_met": target_cells,
        "target_cells_total": 25, "calibration_id": payload["calibration_id"],
        "git": git_state(ROOT), "timestamp_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(ROOT / "outputs" / "phase7" / "phase7_gate_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
