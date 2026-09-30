#!/usr/bin/env python3
"""Frozen-development diagnostic of a brake-to-zero planner candidate."""

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
from bces.simulation.development_stop_hold_candidate import stop_hold_points
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file

MAP_AUDIT = ROOT / "scripts/113_audit_map_reachable_action_feasibility_dev.py"
MAP_AUDIT_SHA256 = "13d5ac8db73739e8ce7b0d76e44f537cba1290ac151fbbb1c367f265d440c4e5"
MAP_REPORT = ROOT / "outputs/study_b/map_reachable_action_feasibility_development_v1/report.json"
MAP_REPORT_SHA256 = "b24bf20e6bc1b72f681aa3dc02e138b130f9bd7d19e19e42feac416ea458b987"
STOP_SOURCE = ROOT / "bces/simulation/development_stop_hold_candidate.py"
STOP_SOURCE_SHA256 = "b7b64a0d42c5b6b625df5a8a224085a5ea5fb844bdc7048c00c9a66df3ddaf0b"
OUTPUT = ROOT / "outputs/study_b/stop_hold_map_feasibility_development_v1"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for filename, digest in ((MAP_AUDIT, MAP_AUDIT_SHA256),
                             (MAP_REPORT, MAP_REPORT_SHA256),
                             (STOP_SOURCE, STOP_SOURCE_SHA256)):
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
    preflight = json.loads(audit.PREFLIGHT.read_text(encoding="utf-8"))
    registration = json.loads(audit.REGISTRATION.read_text(encoding="utf-8"))
    branch_report = json.loads((audit.BASE / "report.json").read_text(encoding="utf-8"))
    for name, digest in branch_report["artifact_sha256"].items():
        if sha256_file(audit.BASE / name) != digest:
            raise RuntimeError("saved branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    scenario = scenario_spec("unprotected_crossing")
    lane_by_seed = {row["seed"]: row["matching_lanes"][0] for row in preflight["rows"]}
    map_root = ET.parse(audit.MAP).getroot()
    baseline_rows = {(row["seed"], row["condition"], row["timestamp_ms"]): row
                     for row in baseline["rows"]}
    rows = []
    for seed in sorted(lane_by_seed):
        for condition in sorted({row["condition"] for row in branch_report["rows"]}):
            filename = f"{seed}_near_{condition}_original_periodic_payload.json.gz"
            with gzip.open(audit.BASE / filename, "rt", encoding="utf-8") as handle:
                branch = json.load(handle)
            actor = audit.actor_at_reference(branch["initial_contract"]["snapshot"])
            paths = audit.paths_from_map(map_root, lane_by_seed[seed], actor["position"])
            visibility = {sample["timestamp_ms"]: sample for sample in branch["visibility_audit"]}
            for sample in branch["rows"]:
                timestamp = sample["timestamp_ms"]
                if (timestamp < 4200 or timestamp > 6000 or "receiver" not in sample
                        or visibility[timestamp]["target_locally_visible"]):
                    continue
                key = seed, condition, timestamp
                saved = baseline_rows[key]
                ego = ego_from_sample(sample["receiver"])
                route = route_for(ego, scenario.intended_behavior, planner)
                original_minus4_available = planner._candidate(ego, route, -4.0, "keep") is not None
                points = stop_hold_points(ego, route, braking_mps2=-4.0,
                                          horizon_s=planner.config.horizon_s,
                                          step_s=planner.config.step_s)
                age = (timestamp - 4000) / 1000.0
                separation = (0.5 * math.hypot(ego.length_m, ego.width_m)
                              + 0.5 * math.hypot(actor["length_m"], actor["width_m"])
                              + audit.LANE_CENTER_DEVIATION_M + audit.POSITION_ERROR_M
                              + planner.config.collision_margin_m)
                minimum_gap = min(
                    audit.distance_to_union(paths, (point.x_m, point.y_m),
                                            actor["speed_mps"], age + point.time_s)
                    - separation for point in points)
                stop_hold_nonoverlap = minimum_gap > 0
                future_labels = 0
                future_covered = 0
                future_missing = 0
                maximum_true_centerline_distance = 0.0
                for point in points:
                    future_ms = int(round(timestamp + 1000 * point.time_s))
                    observed = visibility.get(future_ms)
                    if observed is None or observed["target_xy_m"] is None:
                        future_missing += 1
                        continue
                    future_labels += 1
                    distance = audit.distance_to_union(
                        paths, tuple(observed["target_xy_m"]),
                        actor["speed_mps"], age + point.time_s)
                    maximum_true_centerline_distance = max(maximum_true_centerline_distance,
                                                           distance)
                    future_covered += distance <= audit.LANE_CENTER_DEVIATION_M + audit.POSITION_ERROR_M
                rows.append({"seed": seed, "condition": condition, "timestamp_ms": timestamp,
                             "ego_speed_mps": ego.speed_mps,
                             "original_minus4_candidate_available": original_minus4_available,
                             "original_any_nonoverlap": saved["map_tube_nonoverlap_count"] > 0,
                             "stop_hold_nonoverlap": stop_hold_nonoverlap,
                             "stop_hold_minimum_sampled_gap_m": minimum_gap,
                             "stop_hold_final_speed_mps": points[-1].speed_mps,
                             "future_actor_labels": future_labels,
                             "future_actor_covered": future_covered,
                             "future_actor_missing": future_missing,
                             "maximum_true_centerline_distance_m": maximum_true_centerline_distance})
    if set(baseline_rows) != {(r["seed"], r["condition"], r["timestamp_ms"]) for r in rows}:
        raise RuntimeError("state cohort changed")
    by_seed = {}
    for seed in sorted(lane_by_seed):
        subset = [r for r in rows if r["seed"] == seed]
        available = sum(r["stop_hold_nonoverlap"] or r["original_any_nonoverlap"] for r in subset)
        by_seed[str(seed)] = {"states": len(subset), "any_nonoverlap_with_stop_hold": available,
                              "fraction": available / len(subset)}
    total_available = sum(r["stop_hold_nonoverlap"] or r["original_any_nonoverlap"] for r in rows)
    qualifying_seeds = sum(item["fraction"] >= 0.8 for item in by_seed.values())
    future_labels = sum(r["future_actor_labels"] for r in rows)
    future_covered = sum(r["future_actor_covered"] for r in rows)
    future_missing = sum(r["future_actor_missing"] for r in rows)
    summary = {"states": len(rows),
               "original_minus4_candidate_absent_states":
                   sum(not r["original_minus4_candidate_available"] for r in rows),
               "original_any_nonoverlap_states": sum(r["original_any_nonoverlap"] for r in rows),
               "stop_hold_nonoverlap_states": sum(r["stop_hold_nonoverlap"] for r in rows),
               "any_nonoverlap_with_stop_hold_states": total_available,
               "seeds_at_least_80pct_action_available": qualifying_seeds,
               "future_actor_labels": future_labels,
               "future_actor_covered": future_covered,
               "future_actor_missing": future_missing,
               "development_usefulness_and_synthetic_coverage_gate":
                   total_available / len(rows) >= 0.9 and qualifying_seeds >= 6
                   and future_missing == 0 and future_covered == future_labels,
               "by_seed": by_seed}
    report = {"status": "STOP_HOLD_MAP_FEASIBILITY_DEVELOPMENT_ONLY_V1",
              "script_sha256": sha256_file(Path(__file__)),
              "map_report_sha256": MAP_REPORT_SHA256,
              "stop_hold_source_sha256": STOP_SOURCE_SHA256,
              "summary": summary, "rows": rows,
              "limitations": ["new candidate is not in registered planner or BCES/TTL policy hashes",
                              "saved trajectories are opened synthetic development branches, not fresh trials",
                              "future actor truth used only for retrospective bound audit",
                              "0.2-s sampled nonoverlap does not prove continuous-time safety",
                              "future synthetic coverage is not a real-data or source-view motion bound",
                              "no changed-policy closed loop, packet/byte comparison, or regret certificate"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
