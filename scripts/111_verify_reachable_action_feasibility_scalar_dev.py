#!/usr/bin/env python3
"""Scalar SAT replay of the development reachable-action count."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.oracle.planner import KinematicPlanner, PlannerConfig, _rect_overlap
from bces.oracle.world import WorldObject
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import route_for
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


SOURCE = ROOT / "outputs/study_b/reachable_action_feasibility_development_v1/report.json"
SOURCE_SHA256 = "cdbb8a82197f06e6146c7f5eff41e88446a9af4d8c6427f257eed6bbc81e3b2d"
BASE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
BASE_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
OUTPUT = ROOT / "outputs/study_b/reachable_action_feasibility_scalar_replay_dev_v1"


def reference_actor(snapshot):
    raw = snapshot["vehicles"]["cross"]
    heading = math.radians(90.0 - raw["heading"])
    ux, uy = math.cos(heading), math.sin(heading)
    x = raw["position"][0] - 0.5 * raw["length"] * ux
    y = raw["position"][1] - 0.5 * raw["length"] * uy
    return x, y, raw["speed"] * ux, raw["speed"] * uy, raw["length"], raw["width"]


def scalar_counts(planner, ego, route, shadow):
    total, safe = 0, 0
    for maneuver in planner.config.lateral_maneuvers:
        for acceleration in planner.config.longitudinal_accelerations_mps2:
            points = planner._candidate(ego, route, acceleration, maneuver)
            if points is None:
                continue
            total += 1
            overlap = False
            for point in points:
                predicted = shadow.propagated(point.time_s)
                overlap |= _rect_overlap(
                    point.x_m, point.y_m, point.heading_rad,
                    ego.length_m, ego.width_m,
                    predicted.x_m, predicted.y_m, predicted.heading_rad,
                    predicted.length_m, predicted.width_m,
                    planner.config.collision_margin_m)
            safe += not overlap
    return total, safe


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("feasibility report changed")
    if sha256_file(BASE / "report.json") != BASE_SHA256:
        raise RuntimeError("baseline report changed")
    if sha256_file(REGISTRATION) != REGISTRATION_SHA256:
        raise RuntimeError("registered planner changed")
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    base_report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    behavior = scenario_spec("unprotected_crossing").intended_behavior
    cache = {}
    mismatches = []
    for row in report["rows"]:
        key = (row["seed"], row["condition"])
        if key not in cache:
            name = f"{key[0]}_near_{key[1]}_original_periodic_payload.json.gz"
            if sha256_file(BASE / name) != base_report["artifact_sha256"][name]:
                raise RuntimeError("baseline branch changed")
            with gzip.open(BASE / name, "rt", encoding="utf-8") as handle:
                branch = json.load(handle)
            cache[key] = (reference_actor(branch["initial_contract"]["snapshot"]),
                          {point["timestamp_ms"]: point for point in branch["rows"]
                           if "receiver" in point})
        actor, states = cache[key]
        age = (row["timestamp_ms"] - 4000) / 1000.0
        horizon = planner.config.horizon_s
        elapsed = age + horizon
        expected_half = (0.5 + elapsed
                         + 0.5 * row["acceleration_bound_mps2"] * elapsed**2
                         + 0.5 * math.hypot(actor[4], actor[5]))
        if not math.isclose(row["shadow_width_m"], 2 * expected_half, abs_tol=1e-9):
            raise RuntimeError("shadow width formula mismatch")
        shadow = WorldObject("vehicle:cross", actor[0] + actor[2] * age,
                             actor[1] + actor[3] * age, actor[2], actor[3],
                             0.0, row["shadow_width_m"], row["shadow_width_m"])
        ego = ego_from_sample(states[row["timestamp_ms"]]["receiver"])
        route = route_for(ego, behavior, planner)
        total, safe = scalar_counts(planner, ego, route, shadow)
        if total != row["candidate_count"] or safe != row["no_sampled_overlap_count"]:
            mismatches.append({"seed": row["seed"], "condition": row["condition"],
                               "timestamp_ms": row["timestamp_ms"],
                               "acceleration_bound_mps2": row["acceleration_bound_mps2"],
                               "scalar_total": total, "scalar_safe": safe,
                               "vector_total": row["candidate_count"],
                               "vector_safe": row["no_sampled_overlap_count"]})
    result = {"status": "SCALAR_SAT_REPLAY_OF_REACHABLE_ACTIONS",
              "source_report_sha256": SOURCE_SHA256,
              "script_sha256": sha256_file(Path(__file__)),
              "evaluated_states_and_bound_combinations": len(report["rows"]),
              "mismatches": mismatches,
              "uses_same_saved_ego_states_but_independent_collision_evaluator": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evaluated": len(report["rows"]), "mismatches": len(mismatches)}, indent=2))


if __name__ == "__main__":
    main()
