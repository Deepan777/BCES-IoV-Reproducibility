#!/usr/bin/env python3
"""Post-result scalar geometry replay of the frozen map-tube diagnostic."""

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

from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import route_for
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


AUDIT = ROOT / "scripts/113_audit_map_reachable_action_feasibility_dev.py"
AUDIT_SHA256 = "13d5ac8db73739e8ce7b0d76e44f537cba1290ac151fbbb1c367f265d440c4e5"
SOURCE_REPORT = ROOT / "outputs/study_b/map_reachable_action_feasibility_development_v1/report.json"
SOURCE_SHA256 = "b24bf20e6bc1b72f681aa3dc02e138b130f9bd7d19e19e42feac416ea458b987"
OUT = ROOT / "outputs/study_b/map_reachable_action_feasibility_scalar_replay_dev_v1"


def segment_distance(point, start, end):
    vx = end[0] - start[0]
    vy = end[1] - start[1]
    denom = vx * vx + vy * vy
    if denom == 0:
        return math.dist(point, start)
    parameter = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / denom
    parameter = min(1.0, max(0.0, parameter))
    return math.hypot(point[0] - start[0] - parameter * vx,
                      point[1] - start[1] - parameter * vy)


def clipped_distance(points, point, lo, hi):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    total = sum(lengths)
    nearest = math.inf
    distance_along = 0.0
    for start, end, length in zip(points, points[1:], lengths):
        if length:
            enter = max(lo, distance_along)
            leave = min(hi, distance_along + length)
            if enter <= leave:
                a = ((1 - (enter - distance_along) / length) * start[0]
                     + (enter - distance_along) / length * end[0],
                     (1 - (enter - distance_along) / length) * start[1]
                     + (enter - distance_along) / length * end[1])
                b = ((1 - (leave - distance_along) / length) * start[0]
                     + (leave - distance_along) / length * end[0],
                     (1 - (leave - distance_along) / length) * start[1]
                     + (leave - distance_along) / length * end[1])
                nearest = min(nearest, segment_distance(point, a, b))
        distance_along += length
    if hi > total:
        nearest = min(nearest, max(0.0, math.dist(point, points[-1]) - (hi - total)))
    return nearest


def independent_union_distance(paths, point, speed, elapsed):
    uncertainty = 0.5 + elapsed + 2.0 * elapsed * elapsed
    lo = max(0.0, speed * elapsed - uncertainty)
    hi = speed * elapsed + uncertainty
    return min(clipped_distance(item["path"].points, point, lo, hi) for item in paths)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha256_file(AUDIT) != AUDIT_SHA256 or sha256_file(SOURCE_REPORT) != SOURCE_SHA256:
        raise RuntimeError("frozen audit or result changed")
    spec = importlib.util.spec_from_file_location("frozen_map_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    preflight = json.loads(module.PREFLIGHT.read_text(encoding="utf-8"))
    baseline = json.loads((module.BASE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(module.REGISTRATION.read_text(encoding="utf-8"))
    for filename, digest in baseline["artifact_sha256"].items():
        if sha256_file(module.BASE / filename) != digest:
            raise RuntimeError("saved branch changed: " + filename)
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    behavior = scenario_spec("unprotected_crossing").intended_behavior
    map_root = ET.parse(module.MAP).getroot()
    lane_by_seed = {row["seed"]: row["matching_lanes"][0] for row in preflight["rows"]}
    stored = {(row["seed"], row["condition"], row["timestamp_ms"]): row
              for row in source["rows"]}
    mismatches = []
    checked = 0
    for seed in sorted(lane_by_seed):
        for condition in sorted({row["condition"] for row in baseline["rows"]}):
            filename = f"{seed}_near_{condition}_original_periodic_payload.json.gz"
            with gzip.open(module.BASE / filename, "rt", encoding="utf-8") as handle:
                branch = json.load(handle)
            actor = module.actor_at_reference(branch["initial_contract"]["snapshot"])
            paths = module.paths_from_map(map_root, lane_by_seed[seed], actor["position"])
            visibility = {row["timestamp_ms"]: row for row in branch["visibility_audit"]}
            for row in branch["rows"]:
                timestamp = row["timestamp_ms"]
                if (timestamp < 4200 or timestamp > 6000 or "receiver" not in row
                        or visibility[timestamp]["target_locally_visible"]):
                    continue
                key = (seed, condition, timestamp)
                saved = stored[key]
                ego = ego_from_sample(row["receiver"])
                route = route_for(ego, behavior, planner)
                age = (timestamp - 4000) / 1000.0
                separation = (math.hypot(ego.length_m, ego.width_m) / 2
                              + math.hypot(actor["length_m"], actor["width_m"]) / 2
                              + 2.5 + 0.5 + planner.config.collision_margin_m)
                candidates, clear = 0, 0
                for maneuver in planner.config.lateral_maneuvers:
                    for acceleration in planner.config.longitudinal_accelerations_mps2:
                        prediction = planner._candidate(ego, route, acceleration, maneuver)
                        if prediction is None:
                            continue
                        candidates += 1
                        conflict = False
                        for point in prediction:
                            d = independent_union_distance(paths, (point.x_m, point.y_m),
                                                           actor["speed_mps"], age + point.time_s)
                            if d <= separation:
                                conflict = True
                                break
                        if not conflict:
                            clear += 1
                true_xy = visibility[timestamp]["target_xy_m"]
                true_distance = independent_union_distance(paths, tuple(true_xy),
                                                           actor["speed_mps"], age)
                if (candidates != saved["candidate_count"]
                        or clear != saved["map_tube_nonoverlap_count"]
                        or abs(true_distance - saved["true_actor_centerline_distance_m_label_only"]) > 1e-8):
                    mismatches.append({"key": key, "candidate_count": candidates,
                                       "clear_count": clear, "true_center_distance": true_distance})
                checked += 1
    if checked != len(stored):
        raise RuntimeError("replay cohort mismatch")
    receipt = {"status": "POSTHOC_MAP_TUBE_SCALAR_REPLAY_DEV_V1",
               "source_report_sha256": SOURCE_SHA256, "frozen_audit_sha256": AUDIT_SHA256,
               "states_checked": checked, "mismatch_count": len(mismatches),
               "mismatches": mismatches,
               "limitations": ["same opened synthetic development trajectories",
                               "independent distance arithmetic but shared path assembly and planner candidates",
                               "no future-horizon actor-bound or continuous-time validation"]}
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"states_checked": checked, "mismatch_count": len(mismatches)}))


if __name__ == "__main__":
    main()
