#!/usr/bin/env python3
"""Freeze the Phase-9 variance and local sample-size pilot before confirmation."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/simulation/phase9_pilot_v1.yaml"


def _critical(row: dict, threshold: float) -> int:
    value = row["minimum_ttc_s"]
    return int(row["collision"] or (value is not None and float(value) < threshold))


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("phase9 pilot output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Phase-9 pilot requires a clean committed revision")
    phase8 = json.loads((ROOT / config["phase8_gate"]).read_text(encoding="utf-8"))
    if phase8["status"] != "PASS":
        raise RuntimeError("Phase 8 engineering gate has not passed")
    pair_path = ROOT / config["phase5_pairs"]
    phase5 = json.loads((ROOT / config["phase5_manifest"]).read_text(encoding="utf-8"))
    if sha256_file(pair_path) != phase5["pairs_sha256"]:
        raise ValueError("Phase-5 pair hash mismatch")
    allowed = set(config["allowed_partitions"])
    if "test" in allowed or config["test_partition_access"] != "prohibited":
        raise RuntimeError("pilot cannot use final SUMO test pairs")
    rows = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] in allowed]
    threshold = float(config["critical_ttc_s"])
    endpoints = {
        "collision": [int(row["cached"]["collision"]) - int(row["fresh"]["collision"]) for row in rows],
        "critical_event": [_critical(row["cached"], threshold) - _critical(row["fresh"], threshold) for row in rows],
        "communication_bytes": [row["cached"]["communication_bytes"] - row["fresh"]["communication_bytes"] for row in rows],
        "risk": [row["cached"]["risk"] - row["fresh"]["risk"] for row in rows],
        "route_progress_m": [row["cached"]["route_progress_m"] - row["fresh"]["route_progress_m"] for row in rows],
        "total_abs_jerk": [row["cached"]["total_abs_jerk"] - row["fresh"]["total_abs_jerk"] for row in rows],
    }
    summary = {}
    for name, values in endpoints.items():
        summary[name] = {
            "count": len(values), "mean_paired_difference_cached_minus_fresh": statistics.fmean(values),
            "sd_paired_difference": statistics.stdev(values),
            "nonzero_pairs": sum(value != 0 for value in values),
        }
    communication_sd = summary["communication_bytes"]["sd_paired_difference"]
    fresh_mean = statistics.fmean(row["fresh"]["communication_bytes"] for row in rows)
    minimum_effect = fresh_mean * float(config["minimum_relevant_communication_effect_fraction"])
    z_alpha, z_power = 1.959963984540054, 0.8416212335729143
    communication_required = max(2, math.ceil(((z_alpha + z_power) * communication_sd / minimum_effect) ** 2)) if minimum_effect else None
    critical_nonzero = summary["critical_event"]["nonzero_pairs"]
    minimum = int(config["minimum_confirmatory_seeds"])
    maximum = int(config["maximum_local_confirmatory_seeds"])
    recommendation = min(maximum, max(minimum, communication_required or minimum))
    report = {
        "schema_version": 1, "phase": 9, "component": "pilot_variance_and_power", "status": "PASS",
        "scientific_success": False, "pair_count": len(rows), "unit_of_inference": config["unit_of_inference"],
        "endpoint_summary": summary,
        "power_analysis": {
            "communication_minimum_effect_bytes": minimum_effect,
            "communication_required_pairs_normal_approximation": communication_required,
            "critical_event_power_status": "undetermined_due_insufficient_discordant_pairs" if critical_nonzero < 10 else "estimable",
            "critical_event_discordant_pairs": critical_nonzero,
            "local_resource_cap": maximum,
            "recommended_confirmatory_matched_seeds": recommendation,
            "recommendation_basis": "registered_minimum_and_local_cap_with_pilot_communication_variance",
        },
        "test_partition_accessed": False,
        "config_sha256": sha256_file(CONFIG), "phase5_manifest_sha256": phase5["manifest_sha256"],
        "phase8_gate_sha256": sha256_file(ROOT / config["phase8_gate"]), "git": state,
        "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "pilot.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

