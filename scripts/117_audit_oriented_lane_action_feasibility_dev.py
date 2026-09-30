#!/usr/bin/env python3
"""Development-only orientation-aware map-tube action-feasibility audit."""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.oriented_lane_tube import straight_lane_possible_overlap
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import route_for
from bces.simulation.development_stop_hold_candidate import stop_hold_points
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file

MAP_AUDIT = ROOT / "scripts/113_audit_map_reachable_action_feasibility_dev.py"
MAP_AUDIT_SHA256 = "13d5ac8db73739e8ce7b0d76e44f537cba1290ac151fbbb1c367f265d440c4e5"
MAP_REPORT = ROOT / "outputs/study_b/map_reachable_action_feasibility_development_v1/report.json"
MAP_REPORT_SHA256 = "b24bf20e6bc1b72f681aa3dc02e138b130f9bd7d19e19e42feac416ea458b987"
STOP_REPORT = ROOT / "outputs/study_b/stop_hold_map_feasibility_development_v1/report.json"
STOP_REPORT_SHA256 = "6bd265c277424b3f18d6580e9ec5db9ddb49b167c6469cc9d1f17d31d616681a"
ORIENTED_SOURCE = ROOT / "bces/geometry/oriented_lane_tube.py"
ORIENTED_SOURCE_SHA256 = "b6a516dec56c6e7cf8cefaf02818181f380b349d558e3bdabdb25f7e7000d171"
STOP_SOURCE = ROOT / "bces/simulation/development_stop_hold_candidate.py"
STOP_SOURCE_SHA256 = "b7b64a0d42c5b6b625df5a8a224085a5ea5fb844bdc7048c00c9a66df3ddaf0b"
OUTPUT = ROOT / "outputs/study_b/oriented_lane_action_feasibility_development_v1"
HEADING_ERROR_BOUND_RAD = 0.35


def possible_conflict(points, paths, ego, actor, age, planner, audit):
    straight = [item["path"] for item in paths if item["successor"]["to_edge"] == "B1C1"]
    right = [item["path"] for item in paths if item["successor"]["to_edge"] == "B1B0"]
    if len(straight) != 1 or len(right) != 1:
        raise RuntimeError("expected straight and right mapped continuations")
    margin = planner.config.collision_margin_m
    right_radius = (math.hypot(ego.length_m / 2 + margin, ego.width_m / 2 + margin)
                    + math.hypot(actor["length_m"] / 2 + margin,
                                 actor["width_m"] / 2 + margin)
                    + audit.LANE_CENTER_DEVIATION_M + audit.POSITION_ERROR_M)
    for point in points:
        elapsed = age + point.time_s
        if straight_lane_possible_overlap(
            straight[0], point,
            ego_length_m=ego.length_m, ego_width_m=ego.width_m,
            actor_length_m=actor["length_m"], actor_width_m=actor["width_m"],
            actor_speed_mps=actor["speed_mps"], elapsed_s=elapsed,
            position_error_m=audit.POSITION_ERROR_M,
            velocity_error_mps=audit.VELOCITY_ERROR_MPS,
            acceleration_bound_mps2=audit.ACCELERATION_BOUND_MPS2,
            lane_center_deviation_m=audit.LANE_CENTER_DEVIATION_M,
            collision_margin_m=margin,
            heading_error_bound_rad=HEADING_ERROR_BOUND_RAD):
            return True
        lo, hi = audit.reach_interval(actor["speed_mps"], elapsed)
        if right[0].min_distance_over_interval((point.x_m, point.y_m), lo, hi) <= right_radius:
            return True
    return False


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    expected = ((MAP_AUDIT, MAP_AUDIT_SHA256), (MAP_REPORT, MAP_REPORT_SHA256),
                (STOP_REPORT, STOP_REPORT_SHA256), (ORIENTED_SOURCE, ORIENTED_SOURCE_SHA256),
                (STOP_SOURCE, STOP_SOURCE_SHA256))
    for filename, digest in expected:
        if sha256_file(filename) != digest:
            raise RuntimeError("frozen input changed: " + str(filename))
    spec = importlib.util.spec_from_file_location("frozen_map_audit", MAP_AUDIT)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    for filename, digest in ((audit.MAP, audit.MAP_SHA256),
                             (audit.PREFLIGHT, audit.PREFLIGHT_SHA256),
                             (audit.REGISTRATION, audit.REGISTRATION_SHA256),
                             (audit.BASE / "report.json", audit.BASE_SHA256)):
        if sha256_file(filename) != digest:
            raise RuntimeError("frozen source changed: " + str(filename))
    baseline = json.loads(MAP_REPORT.read_text(encoding="utf-8"))
    stopped = json.loads(STOP_REPORT.read_text(encoding="utf-8"))
    preflight = json.loads(audit.PREFLIGHT.read_text(encoding="utf-8"))
    registration = json.loads(audit.REGISTRATION.read_text(encoding="utf-8"))
    branch_report = json.loads((audit.BASE / "report.json").read_text(encoding="utf-8"))
    for filename, digest in branch_report["artifact_sha256"].items():
        if sha256_file(audit.BASE / filename) != digest:
            raise RuntimeError("development branch changed: " + filename)
    for filename, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / filename) != digest:
            raise RuntimeError("registered source changed: " + filename)
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    scenario = scenario_spec("unprotected_crossing")
    map_root = ET.parse(audit.MAP).getroot()
    lane_by_seed = {row["seed"]: row["matching_lanes"][0] for row in preflight["rows"]}
    base_rows = {(r["seed"], r["condition"], r["timestamp_ms"]): r for r in baseline["rows"]}
    stop_rows = {(r["seed"], r["condition"], r["timestamp_ms"]): r for r in stopped["rows"]}
    rows = []
    for seed in sorted(lane_by_seed):
        for condition in sorted({row["condition"] for row in branch_report["rows"]}):
            filename = f"{seed}_near_{condition}_original_periodic_payload.json.gz"
            with gzip.open(audit.BASE / filename, "rt", encoding="utf-8") as handle:
                branch = json.load(handle)
            actor = audit.actor_at_reference(branch["initial_contract"]["snapshot"])
            paths = audit.paths_from_map(map_root, lane_by_seed[seed], actor["position"])
            visibility = {v["timestamp_ms"]: v for v in branch["visibility_audit"]}
            for sample in branch["rows"]:
                timestamp = sample["timestamp_ms"]
                if (timestamp < 4200 or timestamp > 6000 or "receiver" not in sample
                        or visibility[timestamp]["target_locally_visible"]):
                    continue
                key = seed, condition, timestamp
                if key not in base_rows or key not in stop_rows:
                    raise RuntimeError("cohort changed")
                ego = ego_from_sample(sample["receiver"])
                route = route_for(ego, scenario.intended_behavior, planner)
                age = (timestamp - 4000) / 1000.0
                ordinary_count = 0
                for maneuver in planner.config.lateral_maneuvers:
                    for acceleration in planner.config.longitudinal_accelerations_mps2:
                        points = planner._candidate(ego, route, acceleration, maneuver)
                        if points is not None and not possible_conflict(
                                points, paths, ego, actor, age, planner, audit):
                            ordinary_count += 1
                stop_points = stop_hold_points(
                    ego, route, braking_mps2=-4.0,
                    horizon_s=planner.config.horizon_s,
                    step_s=planner.config.step_s)
                stopped_clear = not possible_conflict(
                    stop_points, paths, ego, actor, age, planner, audit)
                previous_xy = visibility[timestamp]["target_xy_m"]
                tangent_labels, tangent_within, tangent_missing, stationary_steps = 0, 0, 0, 0
                for point in stop_points:
                    future_ms = int(round(timestamp + 1000 * point.time_s))
                    next_row = visibility.get(future_ms)
                    next_xy = None if next_row is None else next_row["target_xy_m"]
                    if previous_xy is None or next_xy is None:
                        tangent_missing += 1
                    else:
                        dx = next_xy[0] - previous_xy[0]
                        dy = next_xy[1] - previous_xy[1]
                        if math.hypot(dx, dy) <= 1e-9:
                            stationary_steps += 1
                        else:
                            tangent_labels += 1
                            tangent_within += dx > 0 and abs(math.atan2(dy, dx)) <= HEADING_ERROR_BOUND_RAD
                    previous_xy = next_xy
                rows.append({"seed": seed, "condition": condition, "timestamp_ms": timestamp,
                             "old_circle_with_stop_hold_available":
                                 stop_rows[key]["original_any_nonoverlap"]
                                 or stop_rows[key]["stop_hold_nonoverlap"],
                             "oriented_original_nonoverlap_count": ordinary_count,
                             "oriented_stop_hold_nonoverlap": stopped_clear,
                             "oriented_any_nonoverlap": ordinary_count > 0 or stopped_clear,
                             "future_tangent_labels": tangent_labels,
                             "future_tangent_within_assumed_angle": tangent_within,
                             "future_tangent_missing": tangent_missing,
                             "future_stationary_displacements": stationary_steps})
    if set(base_rows) != {(r["seed"], r["condition"], r["timestamp_ms"]) for r in rows}:
        raise RuntimeError("evaluated cohort differs from prior audit")
    by_seed = {}
    for seed in sorted(lane_by_seed):
        subset = [r for r in rows if r["seed"] == seed]
        available = sum(r["oriented_any_nonoverlap"] for r in subset)
        by_seed[str(seed)] = {"states": len(subset), "available_states": available,
                              "fraction": available / len(subset)}
    available = sum(r["oriented_any_nonoverlap"] for r in rows)
    qualifying_seeds = sum(item["fraction"] >= 0.8 for item in by_seed.values())
    tangents = sum(r["future_tangent_labels"] for r in rows)
    tangent_within = sum(r["future_tangent_within_assumed_angle"] for r in rows)
    tangent_missing = sum(r["future_tangent_missing"] for r in rows)
    summary = {"states": len(rows), "circle_with_stop_hold_available_states":
                   sum(r["old_circle_with_stop_hold_available"] for r in rows),
               "oriented_original_available_states":
                   sum(r["oriented_original_nonoverlap_count"] > 0 for r in rows),
               "oriented_stop_hold_available_states":
                   sum(r["oriented_stop_hold_nonoverlap"] for r in rows),
               "oriented_any_available_states": available,
               "seeds_at_least_80pct_action_available": qualifying_seeds,
               "future_displacement_tangents": tangents,
               "future_displacement_tangents_within_assumed_angle": tangent_within,
               "future_displacement_tangent_missing": tangent_missing,
               "future_stationary_displacements":
                   sum(r["future_stationary_displacements"] for r in rows),
               "development_usefulness_and_displacement_proxy_gate":
                   available / len(rows) >= 0.9 and qualifying_seeds >= 6
                   and tangent_missing == 0 and tangent_within == tangents,
               "by_seed": by_seed}
    report = {"status": "ORIENTED_LANE_ACTION_FEASIBILITY_DEVELOPMENT_ONLY_V1",
              "script_sha256": sha256_file(Path(__file__)),
              "map_report_sha256": MAP_REPORT_SHA256,
              "stop_report_sha256": STOP_REPORT_SHA256,
              "oriented_source_sha256": ORIENTED_SOURCE_SHA256,
              "assumed_actor_heading_error_bound_rad": HEADING_ERROR_BOUND_RAD,
              "summary": summary, "rows": rows,
              "limitations": ["actor body headings at future ticks are not saved; center-displacement tangent is only a proxy",
                              "heading, lane-deviation, and acceleration bounds lack independent field validation",
                              "right-turn path remains a conservative inflated-circle overbound",
                              "periodic-trajectory development states are off-policy for a new stop-hold controller",
                              "sampled action availability is not continuous-time collision avoidance",
                              "no changed-policy closed loop, wire binding, communication economy, or BCES/TTL comparison"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
