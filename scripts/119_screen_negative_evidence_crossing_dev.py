#!/usr/bin/env python3
"""Development-only balanced present/absent crossing information pilot."""

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
from bces.simulation.development_early_yield_planner import DevelopmentEarlyYieldPlanner
from bces.simulation.development_negative_evidence_planner import (
    DevelopmentNegativeEvidencePlanner, VerifiedEmptyObservation,
)
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file

BASE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
BASE_REPORT_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
BASE_PROTOCOL_SHA256 = "b85ec159b2ac8323eee2dfd5deeea924302231c6a85ed74346484ba6fc6d58f3"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
RUNNER_SOURCE = ROOT / "bces/simulation/development_map_loop.py"
RUNNER_SOURCE_SHA256 = "2d7dbc56e357e8a4b9f91e6f69701d853638c4eb54bce7d3103a6a942d89a3c2"
EARLY_SOURCE = ROOT / "bces/simulation/development_early_yield_planner.py"
EARLY_SOURCE_SHA256 = "734e5d37d2e66aa46a3484110223c3cdfa28e3e50a4231c6fad4186984690346"
NEGATIVE_SOURCE = ROOT / "bces/simulation/development_negative_evidence_planner.py"
NEGATIVE_SOURCE_SHA256 = "a05b56c2f7d1c29064037bc64b99836e84d9b45c33024bfdfbfe7d0feda7517a"
EARLIER_REPORT = ROOT / "outputs/study_b/early_yield_crossing_development_v1/report.json"
EARLIER_REPORT_SHA256 = "509f3e70133565c34047e77c8bd01aaf2e824ea0112093eac8598d8c25d00b1f"
OUTPUT = ROOT / "outputs/study_b/negative_evidence_crossing_development_v1"
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)
PRESENCE = ("present", "absent")
PLANNERS = ("early_yield", "negative_evidence")
METHODS = ("periodic_payload", "local_only")
TIMINGS = ("near", "timing_control")


@contextmanager
def branch_binding(planner_name, actor_presence, method):
    original_planner = development_map_loop.ControlledDecisionPlanner
    original_populate = development_map_loop._populate
    original_payload_objects = development_map_loop.payload_objects

    def populate(connection, spec):
        original_populate(connection, spec)
        if actor_presence == "absent":
            connection.vehicle.remove(spec.actor_id)

    def payload_objects(message):
        physical = original_payload_objects(message)
        if (not physical and planner_name == "negative_evidence"
                and method == "periodic_payload"):
            # In-memory receiver interpretation of a *decoded* empty packet;
            # no marker is transmitted or inserted into sender data.
            return (VerifiedEmptyObservation(),)
        return physical

    try:
        development_map_loop.ControlledDecisionPlanner = (
            DevelopmentNegativeEvidencePlanner if planner_name == "negative_evidence"
            else DevelopmentEarlyYieldPlanner)
        development_map_loop._populate = populate
        development_map_loop.payload_objects = payload_objects
        yield
    finally:
        development_map_loop.ControlledDecisionPlanner = original_planner
        development_map_loop._populate = original_populate
        development_map_loop.payload_objects = original_payload_objects


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    expected = ((BASE / "report.json", BASE_REPORT_SHA256),
                (BASE / "protocol.json", BASE_PROTOCOL_SHA256),
                (REGISTRATION, REGISTRATION_SHA256),
                (RUNNER_SOURCE, RUNNER_SOURCE_SHA256),
                (EARLY_SOURCE, EARLY_SOURCE_SHA256),
                (NEGATIVE_SOURCE, NEGATIVE_SOURCE_SHA256),
                (EARLIER_REPORT, EARLIER_REPORT_SHA256))
    for filename, digest in expected:
        if sha256_file(filename) != digest:
            raise RuntimeError("frozen input changed: " + str(filename))
    previous = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    baseline = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if len(previous["selected"]) != 8 or baseline["branches"] != 128:
        raise RuntimeError("incomplete preselected development cohort")
    for name, digest in baseline["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("saved branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    config = dict(registration["config"])
    config["reference_time_s"] = 3.0
    if config["cooperative_range_m"] < 120.0:
        raise RuntimeError("empty view does not cover the mapped crossing region")
    protocol = {"status": "NEGATIVE_EVIDENCE_CROSSING_DEVELOPMENT_ONLY_V1",
                "script_sha256": sha256_file(Path(__file__)),
                "registration_sha256": REGISTRATION_SHA256,
                "runner_source_sha256": RUNNER_SOURCE_SHA256,
                "negative_planner_source_sha256": NEGATIVE_SOURCE_SHA256,
                "seeds": [row["seed"] for row in previous["selected"]],
                "timings": list(TIMINGS), "conditions": list(conditions),
                "actor_presence": list(PRESENCE), "planners": list(PLANNERS),
                "methods": list(METHODS), "branches_expected": 256,
                "reference_time_s": 3.0, "branch_horizon_s": 7.0,
                "empty_evidence_maximum_age_s": DevelopmentNegativeEvidencePlanner.maximum_empty_age_s,
                "in_memory_empty_marker_not_wire_field": True,
                "scenario_cluster_unit": "seed, not repeated branch or frame",
                "no_BCES_or_TTL_result": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows, artifacts = [], {}
    for index, selected in enumerate(previous["selected"]):
        seed = selected["seed"]
        for timing in TIMINGS:
            position = selected[f"{timing if timing == 'near' else 'control'}_position_m"]
            spec = replace(scenario_spec("unprotected_crossing"),
                           actor_depart_position_m=position,
                           actor_depart_speed_mps=9.0, hidden_from_local=False)
            for condition_index, (condition_name, condition) in enumerate(conditions.items()):
                for presence in PRESENCE:
                    parameters = {"benchmark_stratum": "balanced_negative_evidence_development_v1",
                                  "ego_initial_speed_mps": selected["speed_mps"],
                                  "ego_post_reference_acceleration_mps2": 0.0,
                                  "braking_event_delay_s": 0.2,
                                  "actor_presence": presence}
                    factory = lambda _seed, _config: (spec, parameters)
                    branches = {}
                    fresh_uses = {}
                    for planner_name in PLANNERS:
                        for method in METHODS:
                            with branch_binding(planner_name, presence, method):
                                branch = development_map_loop.run_branch(
                                    seed=seed, method=method, config=config,
                                    condition=condition,
                                    network_seed=seed + 970000 + 100000 * condition_index,
                                    horizon_s=7.0, scenario_factory=factory,
                                    map_blockers=BLOCKERS)
                            branches[(planner_name, method)] = branch
                            fresh_uses[(planner_name, method)] = (
                                DevelopmentNegativeEvidencePlanner.last_instance.fresh_empty_uses
                                if planner_name == "negative_evidence" else 0)
                    reference = branches[("early_yield", "periodic_payload")]["initial_contract"]["snapshot"]
                    for (planner_name, method), branch in branches.items():
                        prefix = compare_snapshots(reference,
                            branch["initial_contract"]["snapshot"],
                            tolerance=1e-6)["equal_within_tolerance"]
                        engineering = bool(
                            prefix and branch["actuation_contract_passed"]
                            and branch["traffic"]["byte_conservation_ok"]
                            and branch["traffic"]["component_conservation_ok"]
                            and all(item["used_timestamp_ms"] <= item["available_ms"]
                                    for item in branch["input_availability_audit"]))
                        name = (f"{seed}_{timing}_{condition_name}_{presence}_"
                                f"{planner_name}_{method}.json.gz")
                        with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                            json.dump(branch, handle, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False)
                        artifacts[name] = sha256_file(OUTPUT / name)
                        rows.append({"seed": seed, "timing": timing,
                                     "condition": condition_name, "presence": presence,
                                     "planner": planner_name, "method": method,
                                     "prefix_equal": prefix, "engineering_passed": engineering,
                                     "severe": _severe(branch),
                                     "progress_m": branch["route_distance_lower_bound_m"],
                                     "generated_bytes": branch["traffic"]["generated_bytes"],
                                     "fresh_empty_uses": fresh_uses[(planner_name, method)]})
        print(json.dumps({"completed_seed_clusters": index + 1,
                          "total_seed_clusters": len(previous["selected"])}), flush=True)
    def row_for(seed, timing, condition, presence, planner, method):
        return next(r for r in rows if r["seed"] == seed and r["timing"] == timing
                    and r["condition"] == condition and r["presence"] == presence
                    and r["planner"] == planner and r["method"] == method)
    pairs = []
    for selected in previous["selected"]:
        seed = selected["seed"]
        for timing in TIMINGS:
            for condition in conditions:
                for presence in PRESENCE:
                    np = row_for(seed, timing, condition, presence,
                                 "negative_evidence", "periodic_payload")
                    nl = row_for(seed, timing, condition, presence,
                                 "negative_evidence", "local_only")
                    ep = row_for(seed, timing, condition, presence,
                                 "early_yield", "periodic_payload")
                    el = row_for(seed, timing, condition, presence,
                                 "early_yield", "local_only")
                    pairs.append({"seed": seed, "timing": timing,
                                  "condition": condition, "presence": presence,
                                  "negative_periodic_severe": np["severe"],
                                  "negative_local_severe": nl["severe"],
                                  "early_periodic_severe": ep["severe"],
                                  "early_local_severe": el["severe"],
                                  "negative_periodic_minus_local_progress_m":
                                      np["progress_m"] - nl["progress_m"],
                                  "negative_periodic_minus_early_periodic_progress_m":
                                      np["progress_m"] - ep["progress_m"],
                                  "negative_periodic_bytes": np["generated_bytes"],
                                  "negative_local_bytes": nl["generated_bytes"],
                                  "fresh_empty_uses": np["fresh_empty_uses"]})
    absent_near = [p for p in pairs if p["presence"] == "absent" and p["timing"] == "near"]
    present_near = [p for p in pairs if p["presence"] == "present" and p["timing"] == "near"]
    summary = {"branches": len(rows), "pairs": len(pairs),
               "engineering_passed_all": all(r["engineering_passed"] for r in rows),
               "negative_periodic_severe_present":
                   sum(p["negative_periodic_severe"] for p in pairs if p["presence"] == "present"),
               "negative_periodic_severe_absent":
                   sum(p["negative_periodic_severe"] for p in pairs if p["presence"] == "absent"),
               "negative_local_severe_present":
                   sum(p["negative_local_severe"] for p in pairs if p["presence"] == "present"),
               "negative_local_severe_absent":
                   sum(p["negative_local_severe"] for p in pairs if p["presence"] == "absent"),
               "mean_absent_near_periodic_minus_local_progress_m":
                   sum(p["negative_periodic_minus_local_progress_m"] for p in absent_near)
                   / len(absent_near),
               "mean_present_near_periodic_minus_local_progress_m":
                   sum(p["negative_periodic_minus_local_progress_m"] for p in present_near)
                   / len(present_near),
               "early_periodic_minus_local_absent_near_mean_m":
                   sum((row_for(p["seed"], p["timing"], p["condition"], "absent",
                                "early_yield", "periodic_payload")["progress_m"]
                        - row_for(p["seed"], p["timing"], p["condition"], "absent",
                                  "early_yield", "local_only")["progress_m"])
                       for p in absent_near) / len(absent_near),
               "fresh_empty_uses_absent_periodic":
                   sum(p["fresh_empty_uses"] for p in pairs if p["presence"] == "absent"),
               "negative_local_bytes_zero": all(p["negative_local_bytes"] == 0 for p in pairs)}
    summary["development_information_utility_screen"] = bool(
        summary["engineering_passed_all"]
        and summary["negative_periodic_severe_present"] == 0
        and summary["negative_periodic_severe_absent"] == 0
        and summary["negative_local_severe_present"] == 0
        and summary["negative_local_severe_absent"] == 0
        and summary["mean_absent_near_periodic_minus_local_progress_m"] >= 10.0
        and summary["mean_present_near_periodic_minus_local_progress_m"] >= -5.0
        and abs(summary["early_periodic_minus_local_absent_near_mean_m"]) < 1e-9
        and summary["fresh_empty_uses_absent_periodic"] > 0
        and summary["negative_local_bytes_zero"])
    report = {"status": "NEGATIVE_EVIDENCE_CROSSING_DEVELOPMENT_ONLY_V1",
              "script_sha256": sha256_file(Path(__file__)),
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts,
              "summary": summary, "rows": rows, "pairs": pairs,
              "limitations": ["actor absence is a synthetic intervention, not a sampled traffic prior",
                              "empty marker exists only after decoded packet; no signed region/coverage wire field",
                              "ideal centered-world RSU observation and 120-m radius are unvalidated",
                              "freshness <=0.4 s is development-only, not certified safe for actor emergence",
                              "near/control absent branches share identical physical absence and are nested duplicates",
                              "no BCES/learned-TTL comparison, packet-bound policy identity, or independent confirmation",
                              "0.2-s sampled overlaps and modeled network bytes do not prove real-world safety"]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
