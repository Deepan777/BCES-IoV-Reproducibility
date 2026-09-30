#!/usr/bin/env python3
"""Development-only lane-union tube feasibility; never use future route labels."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.lane_reachable_tube import PolylinePath, longitudinal_reach_m
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import route_for
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


MAP = ROOT / "sumo/registered_grid_v1.net.xml"
MAP_SHA256 = "476cd09dd964e2f55258e9730f96bbdc0048ba4c2dd357016141e6a386127cfb"
PREFLIGHT = ROOT / "outputs/study_b/crossing_map_lane_contract_development_v1/report.json"
PREFLIGHT_SHA256 = "5716350c550b98f46cf3a07b35edd314232da42e1a88cbdd2f390197aa82883a"
ISOTROPIC = ROOT / "outputs/study_b/reachable_action_feasibility_development_v1/report.json"
ISOTROPIC_SHA256 = "cdbb8a82197f06e6146c7f5eff41e88446a9af4d8c6427f257eed6bbc81e3b2d"
BASE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
BASE_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
TUBE_SOURCE = ROOT / "bces/geometry/lane_reachable_tube.py"
TUBE_SOURCE_SHA256 = "dd208b39196d209dc3d44fc40faf2253b7b03e3e7e9d5aeacf2f19a6f94160f4"
OUTPUT = ROOT / "outputs/study_b/map_reachable_action_feasibility_development_v1"
POSITION_ERROR_M = 0.5
VELOCITY_ERROR_MPS = 1.0
ACCELERATION_BOUND_MPS2 = 4.0
LANE_CENTER_DEVIATION_M = 2.5
FIRST_MS, LAST_MS = 4200, 6000


def shape(text):
    return tuple(tuple(map(float, item.split(","))) for item in text.split())


def actor_at_reference(snapshot):
    raw = snapshot["vehicles"]["cross"]
    heading = math.radians(90.0 - raw["heading"])
    x = raw["position"][0] - 0.5 * raw["length"] * math.cos(heading)
    y = raw["position"][1] - 0.5 * raw["length"] * math.sin(heading)
    return {"position": (x, y), "speed_mps": raw["speed"],
            "length_m": raw["length"], "width_m": raw["width"]}


def append_shape(points, fragment):
    for point in fragment:
        if not points or math.dist(points[-1], point) > 1e-8:
            points.append(point)


def paths_from_map(map_root, lane_match, actor_xy):
    lanes = {lane.attrib["id"]: PolylinePath(shape(lane.attrib["shape"]))
             for lane in map_root.iter("lane") if "shape" in lane.attrib}
    source = lanes[lane_match["lane_id"]]
    lateral_distance, progress = source.nearest_progress(actor_xy)
    if lateral_distance > 2.5:
        raise RuntimeError("actor is not inside preflight lane tolerance")
    paths = []
    for successor in lane_match["successors"]:
        via_id = successor["via_internal_lane"]
        target_id = f'{successor["to_edge"]}_{successor["to_lane_index"]}'
        if via_id not in lanes or target_id not in lanes:
            raise RuntimeError("preflight successor map geometry missing")
        points = []
        append_shape(points, source.remaining_from(progress).points)
        append_shape(points, lanes[via_id].points)
        append_shape(points, lanes[target_id].points)
        paths.append({"successor": successor, "path": PolylinePath(tuple(points))})
    if not paths:
        raise RuntimeError("empty mapped continuation union")
    return paths


def reach_interval(speed, elapsed):
    return longitudinal_reach_m(speed_mps=speed, elapsed_s=elapsed,
                                position_error_m=POSITION_ERROR_M,
                                velocity_error_mps=VELOCITY_ERROR_MPS,
                                acceleration_bound_mps2=ACCELERATION_BOUND_MPS2)


def distance_to_union(paths, xy, speed, elapsed):
    low, high = reach_interval(speed, elapsed)
    return min(item["path"].min_distance_over_interval(xy, low, high)
               for item in paths)


def candidate_counts(planner, ego, route, paths, actor, age):
    ego_radius = 0.5 * math.hypot(ego.length_m, ego.width_m)
    actor_radius = 0.5 * math.hypot(actor["length_m"], actor["width_m"])
    separation = (ego_radius + actor_radius + LANE_CENTER_DEVIATION_M
                  + POSITION_ERROR_M + planner.config.collision_margin_m)
    candidate_count, sampled_nonoverlap = 0, 0
    safe_accelerations = []
    for maneuver in planner.config.lateral_maneuvers:
        for acceleration in planner.config.longitudinal_accelerations_mps2:
            points = planner._candidate(ego, route, acceleration, maneuver)
            if points is None:
                continue
            candidate_count += 1
            conflict = any(distance_to_union(paths, (point.x_m, point.y_m),
                                             actor["speed_mps"], age + point.time_s)
                           <= separation for point in points)
            if not conflict:
                sampled_nonoverlap += 1
                safe_accelerations.append(acceleration)
    return {"candidate_count": candidate_count,
            "map_tube_nonoverlap_count": sampled_nonoverlap,
            "map_tube_nonoverlap_accelerations_mps2": safe_accelerations,
            "separation_threshold_m": separation}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    expected = ((MAP, MAP_SHA256), (PREFLIGHT, PREFLIGHT_SHA256),
                (ISOTROPIC, ISOTROPIC_SHA256), (BASE / "report.json", BASE_SHA256),
                (REGISTRATION, REGISTRATION_SHA256), (TUBE_SOURCE, TUBE_SOURCE_SHA256))
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise RuntimeError("frozen input changed: " + str(path))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    isotropic = json.loads(ISOTROPIC.read_text(encoding="utf-8"))
    baseline = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if not preflight["structural_lane_and_successor_gate"]:
        raise RuntimeError("map lane preflight did not pass")
    for name, digest in baseline["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("baseline branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    behavior = scenario_spec("unprotected_crossing").intended_behavior
    net_root = ET.parse(MAP).getroot()
    lane_by_seed = {row["seed"]: row["matching_lanes"][0] for row in preflight["rows"]}
    rows = []
    path_metadata = {}
    for seed in sorted(lane_by_seed):
        for condition in sorted({row["condition"] for row in baseline["rows"]}):
            name = f"{seed}_near_{condition}_original_periodic_payload.json.gz"
            with gzip.open(BASE / name, "rt", encoding="utf-8") as handle:
                branch = json.load(handle)
            actor = actor_at_reference(branch["initial_contract"]["snapshot"])
            paths = paths_from_map(net_root, lane_by_seed[seed], actor["position"])
            path_metadata[f"{seed}_{condition}"] = [
                {"successor": item["successor"], "path_length_m": item["path"].length_m}
                for item in paths]
            visibility = {point["timestamp_ms"]: point for point in branch["visibility_audit"]}
            for point in branch["rows"]:
                timestamp = point["timestamp_ms"]
                if (timestamp < FIRST_MS or timestamp > LAST_MS
                        or "receiver" not in point
                        or visibility[timestamp]["target_locally_visible"]):
                    continue
                age = (timestamp - 4000) / 1000.0
                ego = ego_from_sample(point["receiver"])
                route = route_for(ego, behavior, planner)
                counts = candidate_counts(planner, ego, route, paths, actor, age)
                true_xy = visibility[timestamp]["target_xy_m"]
                if true_xy is None:
                    raise RuntimeError("actor truth unavailable for label-only containment check")
                centerline_distance = distance_to_union(paths, tuple(true_xy),
                                                        actor["speed_mps"], age)
                rows.append({"seed": seed, "condition": condition,
                             "timestamp_ms": timestamp,
                             "actor_locally_visible": False,
                             "successor_paths": len(paths),
                             "true_actor_centerline_distance_m_label_only": centerline_distance,
                             "true_center_within_assumed_lane_deviation":
                                 centerline_distance <= LANE_CENTER_DEVIATION_M + POSITION_ERROR_M,
                             **counts})
    iso_rows = {(row["seed"], row["condition"], row["timestamp_ms"]): row
                for row in isotropic["rows"] if row["acceleration_bound_mps2"] == 4.0}
    if set(iso_rows) != {(row["seed"], row["condition"], row["timestamp_ms"]) for row in rows}:
        raise RuntimeError("map/isotropic audit state cohorts differ")
    by_seed = {}
    for seed in sorted(lane_by_seed):
        subset = [row for row in rows if row["seed"] == seed]
        by_seed[str(seed)] = {"states": len(subset),
                              "map_tube_states_with_nonoverlap_action":
                                  sum(row["map_tube_nonoverlap_count"] > 0 for row in subset),
                              "true_center_within_assumed_tube_states":
                                  sum(row["true_center_within_assumed_lane_deviation"] for row in subset)}
    useful_states = sum(row["map_tube_nonoverlap_count"] > 0 for row in rows)
    seeds_80pct = sum(item["map_tube_states_with_nonoverlap_action"] / item["states"] >= 0.8
                      for item in by_seed.values())
    truth_covered = sum(row["true_center_within_assumed_lane_deviation"] for row in rows)
    summary = {"evaluated_states": len(rows),
               "map_tube_states_with_nonoverlap_action": useful_states,
               "isotropic_states_with_nonoverlap_action":
                   sum(row["no_sampled_overlap_count"] > 0 for row in iso_rows.values()),
               "seeds_at_least_80pct_action_available": seeds_80pct,
               "true_center_within_assumed_tube_states": truth_covered,
               "development_usefulness_gate":
                   useful_states / len(rows) >= 0.9 and seeds_80pct >= 6
                   and truth_covered == len(rows),
               "by_seed": by_seed}
    report = {"status": "MAP_REACHABLE_ACTION_FEASIBILITY_DEVELOPMENT_ONLY_V1",
              "script_sha256": sha256_file(Path(__file__)),
              "map_sha256": MAP_SHA256,
              "map_preflight_sha256": PREFLIGHT_SHA256,
              "isotropic_report_sha256": ISOTROPIC_SHA256,
              "tube_source_sha256": TUBE_SOURCE_SHA256,
              "assumptions": {"position_error_m": POSITION_ERROR_M,
                              "velocity_error_mps": VELOCITY_ERROR_MPS,
                              "acceleration_bound_mps2": ACCELERATION_BOUND_MPS2,
                              "lane_center_deviation_m": LANE_CENTER_DEVIATION_M,
                              "all_legal_successor_paths": True,
                              "actor_remains_on_matched_lane_or_successor": True},
              "summary": summary, "path_metadata": path_metadata, "rows": rows,
              "limitations": ["selected synthetic development prefixes only; no fresh confirmation",
                              "actor true position used only to audit assumed tube coverage",
                              "lane retention and longitudinal acceleration bounds are not field-validated",
                              "ego/actor circles and 0.2-s future samples are conservative but not continuous-time proof",
                              "no receiver protocol binding or communication outcome"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
