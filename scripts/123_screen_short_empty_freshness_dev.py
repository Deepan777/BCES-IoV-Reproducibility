#!/usr/bin/env python3
"""Frozen 0.2-s negative-evidence freshness sensitivity on opened dev seeds."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import _severe
from bces.network.events import NetworkCondition
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation import development_map_loop
from bces.simulation.development_negative_evidence_planner import DevelopmentNegativeEvidencePlanner
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file
from scripts import _negative_evidence_freshness_support as support

BASE = ROOT / "outputs/study_b/negative_evidence_crossing_development_v1"
BASE_REPORT_SHA256 = "c16589984c7ece771cce70884c6a4265dc65357c0067b002f80a2fb40d3530cd"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
RUNNER_SHA256 = "2d7dbc56e357e8a4b9f91e6f69701d853638c4eb54bce7d3103a6a942d89a3c2"
PLANNER_SHA256 = "a05b56c2f7d1c29064037bc64b99836e84d9b45c33024bfdfbfe7d0feda7517a"
OUTPUT = ROOT / "outputs/study_b/short_empty_freshness_development_v1"
AGE_S = 0.2


@contextmanager
def short_age():
    previous = DevelopmentNegativeEvidencePlanner.maximum_empty_age_s
    if previous != 0.4:
        raise RuntimeError("previous development planner age changed")
    DevelopmentNegativeEvidencePlanner.maximum_empty_age_s = AGE_S
    try:
        yield
    finally:
        DevelopmentNegativeEvidencePlanner.maximum_empty_age_s = previous


def read_branch(name):
    with gzip.open(BASE / name, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    expected = (
        (BASE / "report.json", BASE_REPORT_SHA256),
        (REGISTRATION, REGISTRATION_SHA256),
        (ROOT / "bces/simulation/development_map_loop.py", RUNNER_SHA256),
        (ROOT / "bces/simulation/development_negative_evidence_planner.py", PLANNER_SHA256),
    )
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise RuntimeError("frozen source changed: " + str(path))
    previous = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    for name, digest in previous["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("prior branch changed: " + name)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    selected = support.selected_development_seeds(ROOT)
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    config = dict(registration["config"])
    config["reference_time_s"] = 3.0
    if config["cooperative_range_m"] < 120.0:
        raise RuntimeError("cooperative range below pilot contract")
    protocol = {
        "status": "SHORT_EMPTY_FRESHNESS_DEVELOPMENT_ONLY_V1",
        "script_sha256": sha256_file(Path(__file__)),
        "support_sha256": sha256_file(ROOT / "scripts/_negative_evidence_freshness_support.py"),
        "base_report_sha256": BASE_REPORT_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "seeds": [item["seed"] for item in selected],
        "timings": ["near", "timing_control"],
        "conditions": list(conditions),
        "presence": ["present", "absent"],
        "method": "negative-evidence periodic payload with 0.2-s age; saved same-planner local-only comparator",
        "expected_new_branches": 64,
        "age_s": AGE_S,
        "cluster_unit": "seed; two conditions and timings are nested",
        "prewritten_joint_dev_gate": {
            "all_engineering_checks": True,
            "sampled_severe_present_and_absent": 0,
            "minimum_mean_absent_near_progress_gain_m": 10.0,
            "minimum_seed_clusters_with_positive_absent_near_gain": 6,
            "minimum_mean_present_near_progress_gain_m": -5.0,
            "minimum_total_fresh_empty_uses_absent": 1,
        },
        "no_BCES_or_TTL_result": True,
        "not_independent_confirmation": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows, artifacts = [], {}
    for index, item in enumerate(selected):
        seed = item["seed"]
        for timing in protocol["timings"]:
            position = item["near_position_m" if timing == "near" else "control_position_m"]
            spec = replace(scenario_spec("unprotected_crossing"),
                           actor_depart_position_m=position,
                           actor_depart_speed_mps=9.0, hidden_from_local=False)
            for condition_index, (condition_name, condition) in enumerate(conditions.items()):
                for presence in protocol["presence"]:
                    local_name = (f"{seed}_{timing}_{condition_name}_{presence}_"
                                  "negative_evidence_local_only.json.gz")
                    local = read_branch(local_name)
                    parameters = {
                        "benchmark_stratum": "short_empty_freshness_development_v1",
                        "ego_initial_speed_mps": item["speed_mps"],
                        "ego_post_reference_acceleration_mps2": 0.0,
                        "braking_event_delay_s": 0.2,
                        "actor_presence": presence,
                    }
                    factory = lambda _seed, _config: (spec, parameters)
                    with short_age(), support.branch_binding(presence):
                        branch = development_map_loop.run_branch(
                            seed=seed, method="periodic_payload", config=config,
                            condition=condition,
                            network_seed=seed + 970000 + 100000 * condition_index,
                            horizon_s=7.0, scenario_factory=factory,
                            map_blockers=support.BLOCKERS)
                    prefix = compare_snapshots(local["initial_contract"]["snapshot"],
                                               branch["initial_contract"]["snapshot"],
                                               tolerance=1e-6)["equal_within_tolerance"]
                    engineering = bool(prefix and branch["actuation_contract_passed"]
                                       and branch["traffic"]["byte_conservation_ok"]
                                       and branch["traffic"]["component_conservation_ok"]
                                       and all(row["used_timestamp_ms"] <= row["available_ms"]
                                               for row in branch["input_availability_audit"]))
                    name = f"{seed}_{timing}_{condition_name}_{presence}_periodic_0p2.json.gz"
                    with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                        json.dump(branch, handle, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False)
                    artifacts[name] = sha256_file(OUTPUT / name)
                    rows.append({"seed": seed, "timing": timing,
                                 "condition": condition_name, "presence": presence,
                                 "prefix_equal": prefix, "engineering_passed": engineering,
                                 "periodic_severe": _severe(branch),
                                 "local_severe": _severe(local),
                                 "progress_gain_m": branch["route_distance_lower_bound_m"]
                                                    - local["route_distance_lower_bound_m"],
                                 "periodic_generated_bytes": branch["traffic"]["generated_bytes"],
                                 "local_generated_bytes": local["traffic"]["generated_bytes"],
                                 "fresh_empty_uses": DevelopmentNegativeEvidencePlanner.last_instance.fresh_empty_uses})
        print(json.dumps({"completed_seed_clusters": index + 1,
                          "total_seed_clusters": len(selected)}), flush=True)
    absent_near = [row for row in rows if row["presence"] == "absent" and row["timing"] == "near"]
    present_near = [row for row in rows if row["presence"] == "present" and row["timing"] == "near"]
    seed_gains = {seed: sum(row["progress_gain_m"] for row in absent_near if row["seed"] == seed) / 2
                  for seed in protocol["seeds"]}
    summary = {
        "branches": len(rows),
        "engineering_all": all(row["engineering_passed"] for row in rows),
        "periodic_severe_total": sum(row["periodic_severe"] for row in rows),
        "local_severe_total": sum(row["local_severe"] for row in rows),
        "mean_absent_near_progress_gain_m": sum(row["progress_gain_m"] for row in absent_near) / len(absent_near),
        "mean_present_near_progress_gain_m": sum(row["progress_gain_m"] for row in present_near) / len(present_near),
        "seed_absent_near_progress_gain_m": seed_gains,
        "positive_absent_near_seed_clusters": sum(gain > 0 for gain in seed_gains.values()),
        "fresh_empty_uses_absent": sum(row["fresh_empty_uses"] for row in rows if row["presence"] == "absent"),
        "local_bytes_zero": all(row["local_generated_bytes"] == 0 for row in rows),
    }
    summary["prewritten_joint_dev_gate_passed"] = bool(
        summary["engineering_all"] and summary["periodic_severe_total"] == 0
        and summary["local_severe_total"] == 0
        and summary["mean_absent_near_progress_gain_m"] >= 10.0
        and summary["positive_absent_near_seed_clusters"] >= 6
        and summary["mean_present_near_progress_gain_m"] >= -5.0
        and summary["fresh_empty_uses_absent"] > 0 and summary["local_bytes_zero"])
    report = {"status": "SHORT_EMPTY_FRESHNESS_DEVELOPMENT_ONLY_V1",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "script_sha256": sha256_file(Path(__file__)),
              "support_sha256": protocol["support_sha256"],
              "base_report_sha256": BASE_REPORT_SHA256,
              "artifact_sha256": artifacts, "summary": summary, "rows": rows,
              "not_independent_confirmation": True,
              "no_BCES_or_TTL_result": True}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
