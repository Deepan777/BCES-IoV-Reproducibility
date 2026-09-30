#!/usr/bin/env python3
"""Independent raw-branch replay of the 0.2-s development pilot summary."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "outputs/study_b/short_empty_freshness_development_v1"
BASE = ROOT / "outputs/study_b/negative_evidence_crossing_development_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def severe(row):
    counts = row["counts"]
    return bool(not row["actuation_contract_passed"]
                or row["outcome"] in ("planner_infeasible", "receiver_disappeared")
                or counts.get("geometric_overlap_ticks", 0)
                or counts.get("sumo_ego_collision_ticks", 0))


def close(a, b):
    return abs(a - b) <= 1e-8


def main():
    report = json.loads((NEW / "report.json").read_text(encoding="utf-8"))
    old = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((NEW / "protocol.json").read_text(encoding="utf-8"))
    assert digest(NEW / "protocol.json") == report["protocol_sha256"]
    assert digest(BASE / "report.json") == report["base_report_sha256"]
    assert digest(ROOT / "scripts/123_screen_short_empty_freshness_dev.py") == report["script_sha256"]
    assert digest(ROOT / "scripts/_negative_evidence_freshness_support.py") == report["support_sha256"]
    assert len(report["rows"]) == len(report["artifact_sha256"]) == 64
    assert len({(r["seed"], r["timing"], r["condition"], r["presence"])
                for r in report["rows"]}) == 64
    assert set(r["seed"] for r in report["rows"]) == set(protocol["seeds"])
    gains, present_gains, by_seed = [], [], {seed: [] for seed in protocol["seeds"]}
    adverse = engineering_failures = local_adverse = nonzero_local_bytes = 0
    for row in report["rows"]:
        seed, timing, condition, presence = (row[key] for key in
                                             ("seed", "timing", "condition", "presence"))
        name = f"{seed}_{timing}_{condition}_{presence}_periodic_0p2.json.gz"
        local_name = f"{seed}_{timing}_{condition}_{presence}_negative_evidence_local_only.json.gz"
        assert digest(NEW / name) == report["artifact_sha256"][name]
        assert digest(BASE / local_name) == old["artifact_sha256"][local_name]
        current, local = branch(NEW / name), branch(BASE / local_name)
        assert current["method"] == "periodic_payload" and local["method"] == "local_only"
        assert current["scenario_id"] == local["scenario_id"] == str(seed)
        assert current["initial_contract"]["snapshot"] == local["initial_contract"]["snapshot"]
        assert current["traffic"]["byte_conservation_ok"]
        assert current["traffic"]["component_conservation_ok"]
        assert all(a["used_timestamp_ms"] <= a["available_ms"]
                   for a in current["input_availability_audit"])
        valid = bool(current["actuation_contract_passed"] and row["prefix_equal"])
        engineering_failures += not valid
        adverse += severe(current)
        local_adverse += severe(local)
        nonzero_local_bytes += local["traffic"]["generated_bytes"] != 0
        gain = current["route_distance_lower_bound_m"] - local["route_distance_lower_bound_m"]
        assert close(gain, row["progress_gain_m"])
        assert current["traffic"]["generated_bytes"] == row["periodic_generated_bytes"]
        assert local["traffic"]["generated_bytes"] == row["local_generated_bytes"]
        assert severe(current) == row["periodic_severe"]
        assert severe(local) == row["local_severe"]
        if timing == "near" and presence == "absent":
            gains.append(gain)
            by_seed[seed].append(gain)
        if timing == "near" and presence == "present":
            present_gains.append(gain)
    summary = report["summary"]
    assert len(gains) == len(present_gains) == 16
    assert all(len(values) == 2 for values in by_seed.values())
    assert close(sum(gains) / len(gains), summary["mean_absent_near_progress_gain_m"])
    assert close(sum(present_gains) / len(present_gains), summary["mean_present_near_progress_gain_m"])
    assert all(close(sum(values) / 2, summary["seed_absent_near_progress_gain_m"][str(seed)])
               for seed, values in by_seed.items())
    assert sum(sum(values) > 0 for values in by_seed.values()) == summary["positive_absent_near_seed_clusters"]
    assert (adverse, local_adverse, engineering_failures, nonzero_local_bytes) == (0, 0, 0, 0)
    assert summary["prewritten_joint_dev_gate_passed"]
    print(json.dumps({"verified_artifacts": len(report["artifact_sha256"]),
                      "new_branches": len(report["rows"]),
                      "seed_clusters": len(by_seed),
                      "absent_near_mean_gain_m": sum(gains) / len(gains),
                      "present_near_mean_gain_m": sum(present_gains) / len(present_gains),
                      "sampled_severe_total": adverse + local_adverse,
                      "independently_replayed_gate": True}, indent=2))


if __name__ == "__main__":
    main()
