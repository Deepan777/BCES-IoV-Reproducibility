#!/usr/bin/env python3
"""Frozen development-only original/recovery planner crossing comparison."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import gzip
import importlib.util
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
from bces.simulation.development_recovery_planner import DevelopmentCruiseRecoveryPlanner
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
PREFIX = ROOT / "outputs/study_b/recovery_crossing_prefix_development_v1"
PREFIX_REPORT_SHA256 = "566ca346075c3249002869ba1d908d1c0b87debe93a72c7ad7725d502dafb6dc"
RECOVERY_PLANNER_SHA256 = "599b6f9903f7ce987e8e9b8fb9f9d6860e3b1d8982275a2f80cd5e499492c76e"
METRICS_HELPER = ROOT / "scripts/85_screen_crossing_arrival_development_v2.py"
METRICS_HELPER_SHA256 = "fec358a9c43828170bb3c158c3d1908714cb35040f1252da0f7ab6626847ce29"
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)
PLANNERS = ("original", "recovery")
METHODS = ("periodic_payload", "local_only")
TIMINGS = ("near", "timing_control")
NEAR_LOW_S, NEAR_HIGH_S = -0.10, 0.50
CONTROL_MIN_ABS_ETA_S = 1.0


def load_metrics():
    if sha256_file(METRICS_HELPER) != METRICS_HELPER_SHA256:
        raise RuntimeError("frozen metrics helper changed")
    spec = importlib.util.spec_from_file_location("recovery_crossing_metrics", METRICS_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import metrics helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def planner_binding(name):
    original = development_map_loop.ControlledDecisionPlanner
    try:
        if name == "recovery":
            development_map_loop.ControlledDecisionPlanner = DevelopmentCruiseRecoveryPlanner
        yield
    finally:
        development_map_loop.ControlledDecisionPlanner = original


def eligible_near(row):
    # The earlier prefix-only run failed its stricter gate. This revised
    # selection is frozen before any driving outcomes on these seeds.
    if row["near_status"] == "near_hit_on_approach":
        probe = row["probes"][2]
        return 2, probe
    eligible = [(index, probe) for index, probe in enumerate(row["probes"][:2])
                if probe.get("signed_eta_difference_s") is not None
                and NEAR_LOW_S <= probe["signed_eta_difference_s"] <= NEAR_HIGH_S
                and str(probe.get("actor_lane", "")).startswith("A1B1")
                and str(probe.get("ego_lane", "")).startswith("B0B1")]
    if not eligible:
        return None
    return min(eligible, key=lambda pair: (abs(pair[1]["signed_eta_difference_s"] - 0.20),
                                            pair[0]))


def eligible_control(row):
    eligible = [(index, probe) for index, probe in enumerate(row["probes"][:2])
                if probe.get("signed_eta_difference_s") is not None
                and abs(probe["signed_eta_difference_s"]) >= CONTROL_MIN_ABS_ETA_S
                and str(probe.get("ego_lane", "")).startswith("B0B1")
                and probe.get("actor_present")]
    if not eligible:
        return None
    return max(eligible, key=lambda pair: (abs(pair[1]["signed_eta_difference_s"]),
                                            -pair[0]))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    metrics = load_metrics()
    if sha256_file(PREFIX / "report.json") != PREFIX_REPORT_SHA256:
        raise RuntimeError("prefix report changed")
    if sha256_file(ROOT / "bces/simulation/development_recovery_planner.py") != RECOVERY_PLANNER_SHA256:
        raise RuntimeError("recovery planner changed")
    prefix = json.loads((PREFIX / "report.json").read_text(encoding="utf-8"))
    if prefix["solver_feasible"] or not prefix["no_policy_outcomes"] or len(prefix["rows"]) != 8:
        raise RuntimeError("wrong prefix cohort")
    for name, digest in prefix["artifact_sha256"].items():
        if sha256_file(PREFIX / name) != digest:
            raise RuntimeError("prefix artifact changed: " + name)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    selected = []
    for row in sorted(prefix["rows"], key=lambda item: item["seed"]):
        near, control = eligible_near(row), eligible_control(row)
        if near is None or control is None:
            raise RuntimeError(f"revised prefix eligibility failed for {row['seed']}")
        selected.append((row, {"near": near, "timing_control": control}))
    config = dict(registration["config"])
    config["reference_time_s"] = 4.0
    conditions = tuple(registration["condition_order"])
    protocol = {
        "status": "FROZEN_RECOVERY_CROSSING_DRIVING_DEVELOPMENT_V1",
        "registration_sha256": sha256_file(REGISTRATION),
        "failed_prefix_report_sha256": PREFIX_REPORT_SHA256,
        "recovery_planner_sha256": RECOVERY_PLANNER_SHA256,
        "original_runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "metrics_helper_sha256": METRICS_HELPER_SHA256,
        "script_sha256": sha256_file(Path(__file__)),
        "selected": [{"seed": row["seed"], "speed_mps": row["ego_speed_setting_mps"],
                      "near_index": choices["near"][0],
                      "near_position_m": choices["near"][1]["actor_depart_position_m"],
                      "control_index": choices["timing_control"][0],
                      "control_position_m": choices["timing_control"][1]["actor_depart_position_m"]}
                     for row, choices in selected],
        "timings": list(TIMINGS), "conditions": list(conditions),
        "planners": list(PLANNERS), "methods": list(METHODS),
        "branches_expected": 8 * len(TIMINGS) * len(conditions) * len(PLANNERS) * len(METHODS),
        "near_rule": "interpolated +0.20-s hit if present; otherwise fixed endpoint on both approach lanes with signed ETA in [-0.10,+0.50] s closest to +0.20 s",
        "control_rule": "fixed endpoint with greatest absolute signed ETA gap >=1.0 s, ego on approach and actor present; actor may have entered junction",
        "feasibility_rule": "all engineering/prefix/byte checks; no severe event in any periodic near branch or any timing-control branch; >=4 near scenario seeds with recovery periodic safe/local severe in both conditions over >=3 ego speeds; mean recovery periodic route-progress deficit vs recovery local <=10 m; mean recovery periodic progress gain over original periodic >=4 m",
        "severe_endpoint": "sampled oriented-rectangle overlap, SUMO ego collision, failed actuation, planner infeasibility, or receiver disappearance",
        "cluster_unit": "scenario seed; timings, planners, conditions, methods and frames are paired/repeated",
        "no_bces_or_ttl_outcomes": True, "not_confirmatory": True,
        "limitations": ["prefix design chosen after earlier prefix-only failure but before this cohort's driving outcomes",
                        "ideal sender view", "2-D obstruction rather than calibrated sensing",
                        "coarse 0.2-s geometric overlap is not confirmed physical collision"],
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    network = {name: NetworkCondition(**registration["conditions"][name])
               for name in conditions}
    rows, artifacts = [], {}
    for cluster_index, (chosen, choices) in enumerate(selected):
        seed, speed = chosen["seed"], chosen["ego_speed_setting_mps"]
        for timing in TIMINGS:
            prefix_index, prefix_row = choices[timing]
            position = prefix_row["actor_depart_position_m"]
            frozen_prefix = json.loads((PREFIX / f"{seed}_prefix_{prefix_index}.json").read_text(encoding="utf-8"))
            spec = replace(scenario_spec("unprotected_crossing"),
                           actor_depart_position_m=position,
                           actor_depart_speed_mps=9.0, hidden_from_local=False)
            parameters = {"benchmark_stratum": "prefix_eta_solver_development_v1",
                          "ego_initial_speed_mps": speed,
                          "ego_post_reference_acceleration_mps2": 0.0,
                          "braking_event_delay_s": 0.2}
            factory = lambda _seed, _config: (spec, parameters)
            for condition_index, condition_name in enumerate(conditions):
                branches = {}
                for planner_name in PLANNERS:
                    for method in METHODS:
                        with planner_binding(planner_name):
                            branch = development_map_loop.run_branch(
                                seed=seed, method=method, config=config,
                                condition=network[condition_name],
                                network_seed=seed + 970000 + 100000 * condition_index,
                                horizon_s=registration["horizon_s"],
                                scenario_factory=factory, map_blockers=BLOCKERS)
                        branches[(planner_name, method)] = branch
                        filename = f"{seed}_{timing}_{condition_name}_{planner_name}_{method}.json.gz"
                        with gzip.open(OUTPUT / filename, "xt", encoding="utf-8") as handle:
                            json.dump(branch, handle, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False)
                        artifacts[filename] = sha256_file(OUTPUT / filename)
                snapshots = [branch["initial_contract"]["snapshot"] for branch in branches.values()]
                paired_prefix = all(compare_snapshots(snapshots[0], snapshot, tolerance=1e-6)["equal_within_tolerance"]
                                    for snapshot in snapshots[1:])
                source_prefix = all(compare_snapshots(frozen_prefix["snapshot"], snapshot, tolerance=1e-6)["equal_within_tolerance"]
                                    for snapshot in snapshots)
                engineering = paired_prefix and source_prefix and all(
                    branch["actuation_contract_passed"]
                    and branch["traffic"]["byte_conservation_ok"]
                    and branch["traffic"]["component_conservation_ok"]
                    and all(item["used_timestamp_ms"] <= item["available_ms"]
                            for item in branch["input_availability_audit"])
                    for branch in branches.values())
                for planner_name in PLANNERS:
                    periodic, local = (branches[(planner_name, name)] for name in METHODS)
                    rows.append({"seed": seed, "timing": timing, "condition": condition_name,
                                 "planner": planner_name, "ego_initial_speed_mps": speed,
                                 "actor_depart_position_m": position,
                                 "prefix_signed_eta_difference_s": prefix_row["signed_eta_difference_s"],
                                 "prefix_index": prefix_index,
                                 "paired_prefix_equal": paired_prefix,
                                 "source_prefix_equal": source_prefix,
                                 "engineering_passed": engineering,
                                 "periodic_severe": _severe(periodic),
                                 "local_severe": _severe(local),
                                 "periodic_progress_m": periodic["route_distance_lower_bound_m"],
                                 "local_progress_m": local["route_distance_lower_bound_m"],
                                 "periodic_minimum_sampled_clearance_m": metrics.minimum_clearance(periodic),
                                 "local_minimum_sampled_clearance_m": metrics.minimum_clearance(local),
                                 "periodic_bytes": periodic["traffic"]["generated_bytes"],
                                 "local_bytes": local["traffic"]["generated_bytes"],
                                 "periodic_sumo_ego_collision_ticks": periodic["counts"].get("sumo_ego_collision_ticks", 0),
                                 "local_sumo_ego_collision_ticks": local["counts"].get("sumo_ego_collision_ticks", 0)})
        print(json.dumps({"completed_scenario_clusters": cluster_index + 1,
                          "total_scenario_clusters": len(selected)}), flush=True)
    near_recovery = [row for row in rows if row["timing"] == "near" and row["planner"] == "recovery"]
    controls = [row for row in rows if row["timing"] == "timing_control"]
    utility_seeds = {seed for seed in {row["seed"] for row in near_recovery}
                     if all(not row["periodic_severe"] and row["local_severe"]
                            for row in near_recovery if row["seed"] == seed)}
    utility_speeds = {row["ego_initial_speed_mps"] for row in near_recovery
                      if row["seed"] in utility_seeds}
    deficits = [(row["local_progress_m"] - row["periodic_progress_m"])
                for row in near_recovery if row["seed"] in utility_seeds]
    mean_deficit = sum(deficits) / len(deficits) if deficits else None
    periodic_by_key = {(row["seed"], row["timing"], row["condition"], row["planner"]): row
                       for row in rows}
    gains = [periodic_by_key[(row["seed"], "near", row["condition"], "recovery")]["periodic_progress_m"]
             - periodic_by_key[(row["seed"], "near", row["condition"], "original")]["periodic_progress_m"]
             for row in near_recovery]
    mean_gain = sum(gains) / len(gains) if gains else None
    all_engineering = all(row["engineering_passed"] for row in rows)
    near_periodic_safe = all(not row["periodic_severe"] for row in rows
                             if row["timing"] == "near")
    controls_safe = all(not row["periodic_severe"] and not row["local_severe"]
                        for row in controls)
    gate = bool(all_engineering and near_periodic_safe and controls_safe
                and len(utility_seeds) >= 4 and len(utility_speeds) >= 3
                and mean_deficit is not None and mean_deficit <= 10.0
                and mean_gain is not None and mean_gain >= 4.0)
    report = {"status": "RECOVERY_CROSSING_DRIVING_DEVELOPMENT_ONLY_V1",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "scenario_clusters": len(selected), "branches": len(artifacts),
              "engineering_passed": all_engineering,
              "near_periodic_safe": near_periodic_safe,
              "timing_controls_safe": controls_safe,
              "near_recovery_utility_seeds": len(utility_seeds),
              "near_recovery_utility_speeds": sorted(utility_speeds),
              "mean_recovery_utility_progress_deficit_m": mean_deficit,
              "mean_recovery_vs_original_periodic_progress_gain_m": mean_gain,
              "feasibility_gate": gate,
              "scientific_success": False, "manuscript_allowed": False,
              "limitations": protocol["limitations"] +
              ["no BCES or learned TTL policy outcome", "no independent confirmation"]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("scenario_clusters", "branches", "engineering_passed",
                       "near_periodic_safe", "timing_controls_safe",
                       "near_recovery_utility_seeds", "near_recovery_utility_speeds",
                       "mean_recovery_utility_progress_deficit_m",
                       "mean_recovery_vs_original_periodic_progress_gain_m",
                       "feasibility_gate")}, indent=2))


if __name__ == "__main__":
    main()
