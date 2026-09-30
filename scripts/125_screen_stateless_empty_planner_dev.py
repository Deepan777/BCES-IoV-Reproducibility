#!/usr/bin/env python3
"""Balanced development-only closed-loop screen for stateless empty views."""

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
from bces.simulation.development_negative_evidence_planner import VerifiedEmptyObservation
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file
from scripts._negative_evidence_freshness_support import (
    BLOCKERS, SELECTED_PROTOCOL, SELECTED_PROTOCOL_SHA256,
)

REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
PREVIOUS = ROOT / "outputs/study_b/short_empty_freshness_development_v1/report.json"
PREVIOUS_SHA256 = "60e4306fadbeeeede2e6bb09c772be234019b98c489eac24994901dcb76c32af"
SUPPORT_SHA256 = "c170357b0c7efb71df882ec1a8e5904b56f756c500d7d583dc5a18d8be50bf4e"
PLANNER = ROOT / "bces/simulation/development_stateless_empty_planner.py"
PLANNER_SHA256 = "31544ba4af7584d64946ae093f4801ffa4b2f9631d976003afa7b619fdc2f538"
RUNNER = ROOT / "bces/simulation/development_map_loop.py"
RUNNER_SHA256 = "2d7dbc56e357e8a4b9f91e6f69701d853638c4eb54bce7d3103a6a942d89a3c2"
OUTPUT = ROOT / "outputs/study_b/stateless_empty_planner_development_v1"


@contextmanager
def branch_binding(presence):
    prior_planner = development_map_loop.ControlledDecisionPlanner
    prior_populate = development_map_loop._populate
    prior_payload_objects = development_map_loop.payload_objects

    def populate(connection, spec):
        prior_populate(connection, spec)
        if presence == "absent":
            connection.vehicle.remove(spec.actor_id)

    def objects(message):
        physical = prior_payload_objects(message)
        return (VerifiedEmptyObservation(),) if not physical else physical

    if presence not in ("present", "absent"):
        raise ValueError("unknown actor-presence stratum")
    try:
        development_map_loop.ControlledDecisionPlanner = DevelopmentStatelessEmptyPlanner
        development_map_loop._populate = populate
        development_map_loop.payload_objects = objects
        yield
    finally:
        development_map_loop.ControlledDecisionPlanner = prior_planner
        development_map_loop._populate = prior_populate
        development_map_loop.payload_objects = prior_payload_objects


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, expected in ((REGISTRATION, REGISTRATION_SHA256),
                           (PREVIOUS, PREVIOUS_SHA256),
                           (PLANNER, PLANNER_SHA256),
                           (RUNNER, RUNNER_SHA256),
                           (ROOT / "scripts/_negative_evidence_freshness_support.py", SUPPORT_SHA256),
                           (ROOT / SELECTED_PROTOCOL, SELECTED_PROTOCOL_SHA256)):
        if sha256_file(path) != expected:
            raise RuntimeError("frozen input changed: " + str(path))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    selected = json.loads((ROOT / SELECTED_PROTOCOL).read_text(encoding="utf-8"))["selected"]
    if len(selected) != 8 or len({item["seed"] for item in selected}) != 8:
        raise RuntimeError("expected all eight selected development seeds")
    config = dict(registration["config"])
    config["reference_time_s"] = 3.0
    if config["cooperative_range_m"] < 120.0:
        raise RuntimeError("source-view range below pilot contract")
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    protocol = {
        "status": "STATELESS_EMPTY_PLANNER_DEVELOPMENT_ONLY_V1",
        "script_sha256": sha256_file(Path(__file__)),
        "planner_source_sha256": PLANNER_SHA256,
        "support_source_sha256": SUPPORT_SHA256,
        "runner_source_sha256": RUNNER_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "previous_short_age_report_sha256": PREVIOUS_SHA256,
        "seeds": [item["seed"] for item in selected],
        "timings": ["near", "timing_control"],
        "conditions": list(conditions),
        "presence": ["present", "absent"],
        "methods": ["periodic_payload", "local_only"],
        "expected_branches": 128,
        "maximum_empty_age_s": DevelopmentStatelessEmptyPlanner.maximum_empty_age_s,
        "primary_unit": "seed cluster, not network/timing/presence branch",
        "joint_development_gate": {
            "all_engineering_checks": True,
            "sampled_severe_total": 0,
            "minimum_absent_near_mean_progress_gain_m": 10.0,
            "minimum_positive_absent_near_seed_clusters": 6,
            "minimum_present_near_mean_progress_gain_m": -5.0,
            "minimum_absent_periodic_fresh_empty_uses": 1,
            "local_only_zero_bytes": True,
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
                    parameters = {
                        "benchmark_stratum": "stateless_empty_planner_development_v1",
                        "ego_initial_speed_mps": item["speed_mps"],
                        "ego_post_reference_acceleration_mps2": 0.0,
                        "braking_event_delay_s": 0.2,
                        "actor_presence": presence,
                    }
                    factory = lambda _seed, _config: (spec, parameters)
                    branches = {}
                    for method in protocol["methods"]:
                        with branch_binding(presence):
                            branch = development_map_loop.run_branch(
                                seed=seed, method=method, config=config,
                                condition=condition,
                                network_seed=seed + 970000 + 100000 * condition_index,
                                horizon_s=7.0, scenario_factory=factory,
                                map_blockers=BLOCKERS)
                        uses = DevelopmentStatelessEmptyPlanner.last_instance.fresh_empty_uses
                        branches[method] = branch
                        name = f"{seed}_{timing}_{condition_name}_{presence}_{method}.json.gz"
                        with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                            json.dump(branch, handle, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False)
                        artifacts[name] = sha256_file(OUTPUT / name)
                        rows.append({"seed": seed, "timing": timing,
                                     "condition": condition_name, "presence": presence,
                                     "method": method, "severe": _severe(branch),
                                     "progress_m": branch["route_distance_lower_bound_m"],
                                     "generated_bytes": branch["traffic"]["generated_bytes"],
                                     "fresh_empty_uses": uses,
                                     "policy_hash": branch["initial_contract"]["policy_hash"],
                                     "actuation_passed": branch["actuation_contract_passed"],
                                     "causal_inputs": all(a["used_timestamp_ms"] <= a["available_ms"]
                                                          for a in branch["input_availability_audit"]),
                                     "bytes_conserved": branch["traffic"]["byte_conservation_ok"]
                                                        and branch["traffic"]["component_conservation_ok"]})
                    same = compare_snapshots(
                        branches["periodic_payload"]["initial_contract"]["snapshot"],
                        branches["local_only"]["initial_contract"]["snapshot"],
                        tolerance=1e-6)["equal_within_tolerance"]
                    for row in rows[-2:]:
                        row["prefix_equal"] = same
        print(json.dumps({"completed_seed_clusters": index + 1,
                          "total_seed_clusters": len(selected)}), flush=True)
    pairs = []
    for item in selected:
        for timing in protocol["timings"]:
            for condition in conditions:
                for presence in protocol["presence"]:
                    subset = [r for r in rows if r["seed"] == item["seed"]
                              and r["timing"] == timing and r["condition"] == condition
                              and r["presence"] == presence]
                    periodic = next(r for r in subset if r["method"] == "periodic_payload")
                    local = next(r for r in subset if r["method"] == "local_only")
                    pairs.append({"seed": item["seed"], "timing": timing,
                                  "condition": condition, "presence": presence,
                                  "periodic_minus_local_progress_m": periodic["progress_m"] - local["progress_m"],
                                  "periodic_severe": periodic["severe"],
                                  "local_severe": local["severe"]})
    absent_near = [p for p in pairs if p["presence"] == "absent" and p["timing"] == "near"]
    present_near = [p for p in pairs if p["presence"] == "present" and p["timing"] == "near"]
    by_seed = {item["seed"]: sum(p["periodic_minus_local_progress_m"]
                                for p in absent_near if p["seed"] == item["seed"]) / 2
               for item in selected}
    engineering = all(r["actuation_passed"] and r["causal_inputs"]
                      and r["bytes_conserved"] and r["prefix_equal"] for r in rows)
    summary = {
        "branches": len(rows), "paired_cells": len(pairs),
        "engineering_all": engineering,
        "sampled_severe_total": sum(r["severe"] for r in rows),
        "mean_absent_near_periodic_minus_local_progress_m":
            sum(p["periodic_minus_local_progress_m"] for p in absent_near) / len(absent_near),
        "mean_present_near_periodic_minus_local_progress_m":
            sum(p["periodic_minus_local_progress_m"] for p in present_near) / len(present_near),
        "absent_near_seed_mean_gains_m": by_seed,
        "positive_absent_near_seed_clusters": sum(value > 0 for value in by_seed.values()),
        "fresh_empty_uses_absent_periodic": sum(r["fresh_empty_uses"] for r in rows
                                                if r["presence"] == "absent"
                                                and r["method"] == "periodic_payload"),
        "all_local_bytes_zero": all(r["generated_bytes"] == 0 for r in rows
                                    if r["method"] == "local_only"),
        "unique_policy_hashes": sorted({r["policy_hash"] for r in rows}),
    }
    summary["joint_development_gate_passed"] = bool(
        engineering and summary["sampled_severe_total"] == 0
        and summary["mean_absent_near_periodic_minus_local_progress_m"] >= 10
        and summary["positive_absent_near_seed_clusters"] >= 6
        and summary["mean_present_near_periodic_minus_local_progress_m"] >= -5
        and summary["fresh_empty_uses_absent_periodic"] > 0
        and summary["all_local_bytes_zero"]
        and len(summary["unique_policy_hashes"]) == 1)
    report = {"status": protocol["status"],
              "script_sha256": protocol["script_sha256"],
              "planner_source_sha256": PLANNER_SHA256,
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "pairs": pairs, "summary": summary,
              "not_independent_confirmation": True,
              "no_BCES_or_TTL_result": True}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
