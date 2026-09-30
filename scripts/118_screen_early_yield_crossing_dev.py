#!/usr/bin/env python3
"""Development-only all-seed early-yield information-utility pilot."""

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
STOP_SOURCE = ROOT / "bces/simulation/development_stop_hold_candidate.py"
STOP_SOURCE_SHA256 = "b7b64a0d42c5b6b625df5a8a224085a5ea5fb844bdc7048c00c9a66df3ddaf0b"
OUTPUT = ROOT / "outputs/study_b/early_yield_crossing_development_v1"
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)
PLANNERS = ("original", "early_yield")
METHODS = ("periodic_payload", "local_only")
TIMINGS = ("near", "timing_control")


@contextmanager
def planner_binding(name):
    original = development_map_loop.ControlledDecisionPlanner
    try:
        if name == "early_yield":
            development_map_loop.ControlledDecisionPlanner = DevelopmentEarlyYieldPlanner
        yield
    finally:
        development_map_loop.ControlledDecisionPlanner = original


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for path, digest in ((BASE / "report.json", BASE_REPORT_SHA256),
                         (BASE / "protocol.json", BASE_PROTOCOL_SHA256),
                         (REGISTRATION, REGISTRATION_SHA256),
                         (RUNNER_SOURCE, RUNNER_SOURCE_SHA256),
                         (EARLY_SOURCE, EARLY_SOURCE_SHA256),
                         (STOP_SOURCE, STOP_SOURCE_SHA256)):
        if sha256_file(path) != digest:
            raise RuntimeError("frozen input changed: " + str(path))
    baseline = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    previous_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if baseline["branches"] != 128 or len(previous_protocol["selected"]) != 8:
        raise RuntimeError("all-eight-seed prior development cohort unavailable")
    for name, digest in baseline["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("saved development branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    config = dict(registration["config"])
    config["reference_time_s"] = 3.0
    protocol = {"status": "EARLY_YIELD_CROSSING_DEVELOPMENT_ONLY_V1",
                "script_sha256": sha256_file(Path(__file__)),
                "registration_sha256": REGISTRATION_SHA256,
                "runner_sha256": RUNNER_SOURCE_SHA256,
                "early_policy_sha256": EARLY_SOURCE_SHA256,
                "seeds": [item["seed"] for item in previous_protocol["selected"]],
                "timings": list(TIMINGS), "conditions": list(conditions),
                "planners": list(PLANNERS), "methods": list(METHODS),
                "reference_time_s": 3.0, "branch_horizon_s": 7.0,
                "branches_expected": 128,
                "no_BCES_or_TTL_result": True,
                "new_policy_requires_distinct_binding_and_training": True,
                "all_seed_scenario_cluster_unit": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifacts = {}
    rows = []
    for index, selected in enumerate(previous_protocol["selected"]):
        seed = selected["seed"]
        for timing in TIMINGS:
            position = selected[f"{timing if timing == 'near' else 'control'}_position_m"]
            spec = replace(scenario_spec("unprotected_crossing"),
                           actor_depart_position_m=position,
                           actor_depart_speed_mps=9.0, hidden_from_local=False)
            parameters = {"benchmark_stratum": "early_yield_crossing_development_v1",
                          "ego_initial_speed_mps": selected["speed_mps"],
                          "ego_post_reference_acceleration_mps2": 0.0,
                          "braking_event_delay_s": 0.2}
            factory = lambda _seed, _config: (spec, parameters)
            for condition_index, (condition_name, condition) in enumerate(conditions.items()):
                branches = {}
                for planner_name in PLANNERS:
                    for method in METHODS:
                        with planner_binding(planner_name):
                            branch = development_map_loop.run_branch(
                                seed=seed, method=method, config=config,
                                condition=condition,
                                network_seed=seed + 970000 + 100000 * condition_index,
                                horizon_s=7.0, scenario_factory=factory,
                                map_blockers=BLOCKERS)
                        branches[(planner_name, method)] = branch
                reference = branches[("original", "periodic_payload")]["initial_contract"]["snapshot"]
                for (planner_name, method), branch in branches.items():
                    prefix = compare_snapshots(
                        reference, branch["initial_contract"]["snapshot"],
                        tolerance=1e-6)["equal_within_tolerance"]
                    engineering = bool(
                        prefix and branch["actuation_contract_passed"]
                        and branch["traffic"]["byte_conservation_ok"]
                        and branch["traffic"]["component_conservation_ok"]
                        and all(item["used_timestamp_ms"] <= item["available_ms"]
                                for item in branch["input_availability_audit"]))
                    name = f"{seed}_{timing}_{condition_name}_{planner_name}_{method}.json.gz"
                    with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                        json.dump(branch, handle, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
                    artifacts[name] = sha256_file(OUTPUT / name)
                    rows.append({"seed": seed, "timing": timing, "condition": condition_name,
                                 "planner": planner_name, "method": method,
                                 "prefix_equal": prefix, "engineering_passed": engineering,
                                 "severe": _severe(branch),
                                 "route_progress_m": branch["route_distance_lower_bound_m"],
                                 "generated_bytes": branch["traffic"]["generated_bytes"],
                                 "geometric_overlap_ticks":
                                     branch["counts"].get("geometric_overlap_ticks", 0),
                                 "sumo_ego_collision_ticks":
                                     branch["counts"].get("sumo_ego_collision_ticks", 0)})
        print(json.dumps({"completed_seed_clusters": index + 1,
                          "total_seed_clusters": len(previous_protocol["selected"])}), flush=True)
    def row_for(seed, timing, condition, planner, method):
        return next(row for row in rows if row["seed"] == seed and row["timing"] == timing
                    and row["condition"] == condition and row["planner"] == planner
                    and row["method"] == method)
    pairs = []
    for selected in previous_protocol["selected"]:
        seed = selected["seed"]
        for timing in TIMINGS:
            for condition in conditions:
                periodic = row_for(seed, timing, condition, "early_yield", "periodic_payload")
                local = row_for(seed, timing, condition, "early_yield", "local_only")
                original_periodic = row_for(seed, timing, condition, "original", "periodic_payload")
                original_local = row_for(seed, timing, condition, "original", "local_only")
                pairs.append({"seed": seed, "timing": timing, "condition": condition,
                              "early_periodic_severe": periodic["severe"],
                              "early_local_severe": local["severe"],
                              "original_periodic_severe": original_periodic["severe"],
                              "original_local_severe": original_local["severe"],
                              "early_periodic_minus_local_progress_m":
                                  periodic["route_progress_m"] - local["route_progress_m"],
                              "early_periodic_minus_original_periodic_progress_m":
                                  periodic["route_progress_m"] - original_periodic["route_progress_m"],
                              "early_periodic_bytes": periodic["generated_bytes"],
                              "early_local_bytes": local["generated_bytes"]})
    near = [pair for pair in pairs if pair["timing"] == "near"]
    periodic_safe = [pair for pair in pairs if not pair["original_periodic_severe"]]
    summary = {"branches": len(rows), "pairs": len(pairs),
               "engineering_passed_all": all(row["engineering_passed"] for row in rows),
               "early_periodic_severe_near": sum(pair["early_periodic_severe"] for pair in near),
               "early_local_severe_near": sum(pair["early_local_severe"] for pair in near),
               "early_periodic_severe_where_original_periodic_safe":
                   sum(pair["early_periodic_severe"] for pair in periodic_safe),
               "mean_near_periodic_minus_local_progress_m":
                   sum(pair["early_periodic_minus_local_progress_m"] for pair in near) / len(near),
               "mean_near_periodic_minus_original_periodic_progress_m":
                   sum(pair["early_periodic_minus_original_periodic_progress_m"] for pair in near) / len(near),
               "early_local_generated_bytes_zero": all(pair["early_local_bytes"] == 0 for pair in pairs)}
    summary["development_information_utility_screen"] = bool(
        summary["engineering_passed_all"]
        and summary["early_periodic_severe_where_original_periodic_safe"] == 0
        and summary["early_periodic_severe_near"] == 0
        and summary["mean_near_periodic_minus_local_progress_m"] >= 5.0
        and summary["mean_near_periodic_minus_original_periodic_progress_m"] >= -5.0
        and summary["early_local_generated_bytes_zero"])
    report = {"status": "EARLY_YIELD_CROSSING_DEVELOPMENT_ONLY_V1",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "script_sha256": sha256_file(Path(__file__)),
              "artifact_sha256": artifacts, "summary": summary,
              "rows": rows, "pairs": pairs,
              "limitations": ["new planner has no BCES or learned-TTL model binding",
                              "same eight opened, synthetic development seeds",
                              "earlier branch point changes trajectory and estimand versus prior t=4.0 experiments",
                              "periodic and local-only traffic differ by design; no matched-byte communication comparison",
                              "geometric overlaps sampled every 0.2 s, not continuous-time proof",
                              "scenario-specific map hazard and ideal cooperative sender are not field validation"]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
