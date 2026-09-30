#!/usr/bin/env python3
"""Independent hash and raw-branch check for the stateless development pilot."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/study_b/stateless_empty_planner_development_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def severe(branch):
    c = branch["counts"]
    return bool(not branch["actuation_contract_passed"]
                or branch["outcome"] in ("planner_infeasible", "receiver_disappeared")
                or c.get("geometric_overlap_ticks", 0)
                or c.get("sumo_ego_collision_ticks", 0))


def main():
    report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    assert sha(BASE / "protocol.json") == report["protocol_sha256"]
    assert sha(ROOT / "scripts/125_screen_stateless_empty_planner_dev.py") == report["script_sha256"]
    assert sha(ROOT / "bces/simulation/development_stateless_empty_planner.py") == report["planner_source_sha256"]
    assert len(report["artifact_sha256"]) == len(report["rows"]) == 128
    cells = {}
    adverse = 0
    policy_hashes = set()
    for row in report["rows"]:
        key = tuple(row[k] for k in ("seed", "timing", "condition", "presence"))
        name = f"{row['seed']}_{row['timing']}_{row['condition']}_{row['presence']}_{row['method']}.json.gz"
        assert sha(BASE / name) == report["artifact_sha256"][name]
        branch = load(BASE / name)
        assert branch["method"] == row["method"] and branch["scenario_id"] == str(row["seed"])
        assert branch["initial_contract"]["parameters"]["actor_presence"] == row["presence"]
        assert abs(branch["route_distance_lower_bound_m"] - row["progress_m"]) < 1e-8
        assert branch["traffic"]["generated_bytes"] == row["generated_bytes"]
        assert branch["traffic"]["byte_conservation_ok"] and branch["traffic"]["component_conservation_ok"]
        assert all(x["used_timestamp_ms"] <= x["available_ms"] for x in branch["input_availability_audit"])
        assert branch["actuation_contract_passed"] and severe(branch) == row["severe"]
        adverse += severe(branch)
        policy_hashes.add(branch["initial_contract"]["policy_hash"])
        cells.setdefault(key, {})[row["method"]] = branch
    assert len(cells) == 64 and all(len(value) == 2 for value in cells.values())
    assert len(policy_hashes) == 1 and list(policy_hashes) == report["summary"]["unique_policy_hashes"]
    absent, present = [], []
    by_seed = {seed: [] for seed in protocol["seeds"]}
    for (seed, timing, _condition, presence), value in cells.items():
        periodic, local = value["periodic_payload"], value["local_only"]
        assert periodic["initial_contract"]["snapshot"] == local["initial_contract"]["snapshot"]
        assert local["traffic"]["generated_bytes"] == 0
        gain = periodic["route_distance_lower_bound_m"] - local["route_distance_lower_bound_m"]
        if timing == "near" and presence == "absent":
            absent.append(gain)
            by_seed[seed].append(gain)
        if timing == "near" and presence == "present":
            present.append(gain)
    summary = report["summary"]
    assert len(absent) == len(present) == 16
    assert abs(sum(absent) / 16 - summary["mean_absent_near_periodic_minus_local_progress_m"]) < 1e-8
    assert abs(sum(present) / 16 - summary["mean_present_near_periodic_minus_local_progress_m"]) < 1e-8
    assert all(len(gains) == 2 and abs(sum(gains) / 2 - summary["absent_near_seed_mean_gains_m"][str(seed)]) < 1e-8
               for seed, gains in by_seed.items())
    assert sum(sum(gains) > 0 for gains in by_seed.values()) == summary["positive_absent_near_seed_clusters"]
    assert adverse == 0 and summary["joint_development_gate_passed"]
    print(json.dumps({"verified_artifacts": 128, "paired_cells": 64,
                      "seed_clusters": len(by_seed), "sampled_severe": adverse,
                      "absent_near_mean_gain_m": sum(absent) / 16,
                      "present_near_mean_gain_m": sum(present) / 16,
                      "independent_gate_replay": True}, indent=2))


if __name__ == "__main__":
    main()
