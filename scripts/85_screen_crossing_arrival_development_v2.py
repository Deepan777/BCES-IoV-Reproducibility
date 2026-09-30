#!/usr/bin/env python3
"""Prospective development-only arrival-timing screen; no BCES/TTL scoring."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import _severe
from bces.network.events import NetworkCondition
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.development_map_loop import run_branch
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/map_crossing_arrival_development_v2"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
PRIOR_REPORT = ROOT / "outputs/study_b/map_crossing_utility_development_v1/report.json"
PRIOR_REPORT_SHA256 = "1c9774463e6b8125fa29b56b6ebb8bf23a63327ba4e563f4269373762f2eafc3"
SPEED_POSITION_BASE = ((11.0, 8.0), (12.0, 16.0), (13.0, 23.0), (14.0, 29.0))
POSITION_OFFSETS = (-4.0, 0.0, 4.0)
REPLICATES = (0, 1)
METHODS = ("periodic_payload", "local_only")
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)


def map_gap(branch: dict, reference_ms: int) -> dict:
    rows = branch["visibility_audit"]
    blocked = [row["timestamp_ms"] for row in rows
               if row["timestamp_ms"] >= reference_ms
               and row["target_within_local_range"]
               and row["target_visible_without_map"]
               and not row["target_locally_visible"]]
    reveal = next((row["timestamp_ms"] for row in rows
                   if blocked and row["timestamp_ms"] > blocked[-1]
                   and row["target_locally_visible"]), None)
    return {"post_reference_map_blocked_frames": len(blocked),
            "first_map_blocked_ms": blocked[0] if blocked else None,
            "first_reveal_after_gap_ms": reveal}


def minimum_clearance(branch: dict) -> float | None:
    """Sampled separation of perpendicular rectangles, not continuous CCD."""
    vehicles = branch["initial_contract"]["snapshot"]["vehicles"]
    ego, actor = vehicles["ego"], vehicles["cross"]
    x_extent = float(ego["width"]) / 2 + float(actor["length"]) / 2
    y_extent = float(ego["length"]) / 2 + float(actor["width"]) / 2
    origin_ms = round(float(branch["initial_contract"]["snapshot"]["time_s"]) * 1000)
    distances = []
    for row in branch["visibility_audit"]:
        if row["timestamp_ms"] < origin_ms or row["target_xy_m"] is None:
            continue
        ex, ey = row["ego_xy_m"]
        ax, ay = row["target_xy_m"]
        distances.append(math.hypot(max(abs(ex - ax) - x_extent, 0.0),
                                    max(abs(ey - ay) - y_extent, 0.0)))
    return min(distances) if distances else None


def prefix_etas(initial: dict) -> dict:
    ego, actor = (initial["snapshot"]["vehicles"][name] for name in ("ego", "cross"))
    ex, ey = ego["position"]
    ax, ay = actor["position"]
    if ego["speed"] <= 0 or actor["speed"] <= 0:
        return {"ego_eta_s": None, "actor_eta_s": None, "signed_eta_difference_s": None}
    ego_eta = (ay - ey) / ego["speed"]
    actor_eta = (ex - ax) / actor["speed"]
    return {"ego_eta_s": ego_eta, "actor_eta_s": actor_eta,
            "signed_eta_difference_s": ego_eta - actor_eta}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(PRIOR_REPORT) != PRIOR_REPORT_SHA256:
        raise RuntimeError("prior development report changed")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    grid = [(speed, base + offset, offset, replicate)
            for speed, base in SPEED_POSITION_BASE
            for offset in POSITION_OFFSETS
            for replicate in REPLICATES]
    conditions = tuple(registration["condition_order"])
    if len(grid) != 24 or len(conditions) != 2:
        raise RuntimeError("fixed grid mismatch")
    protocol = {
        "status": "FROZEN_DEVELOPMENT_FEASIBILITY_ONLY_NOT_CONFIRMATION",
        "registration_sha256": sha256_file(REGISTRATION),
        "prior_screen_sha256": PRIOR_REPORT_SHA256,
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility_map.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "seeds": [8500000 + i for i in range(len(grid))],
        "grid": [{"ego_initial_speed_mps": speed, "actor_depart_position_m": position,
                  "position_offset_from_physical_eta_center_m": offset, "replicate": replicate}
                 for speed, position, offset, replicate in grid],
        "timing_law": "position centers 8,16,23,29 m for ego speeds 11,12,13,14 m/s derived from 5-s constant-speed crossing ETA equality; offsets -4,0,+4 m; no outcome filtering",
        "family": "unprotected_crossing", "actor_depart_speed_mps": 9.0,
        "braking_event_delay_s": 0.2,
        "blockers": [asdict(item) for item in BLOCKERS],
        "methods": list(METHODS), "conditions": list(conditions),
        "primary_feasibility_rule": "at least 4 distinct position/speed cells with periodic-safe/local-severe across the 24 seeds, plus at least 8 seeds with >=3 post-reference map-blocked frames and later reveal",
        "secondary_metrics": ["paired route-progress difference", "sampled axis-aligned crossing clearance", "prefix-only signed ETA difference", "modeled generated bytes"],
        "sender_limit": "ideal centered-world cooperative range; RSU occlusion is not modeled",
        "no_policy_scoring": True, "not_confirmatory": True,
        "scientific_success": False, "manuscript_allowed": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    network = {name: NetworkCondition(**registration["conditions"][name])
               for name in conditions}
    rows, artifact_sha256 = [], {}
    for index, (speed, position, offset, replicate) in enumerate(grid):
        seed = 8500000 + index
        spec = replace(scenario_spec("unprotected_crossing"),
                       actor_depart_position_m=position,
                       actor_depart_speed_mps=9.0, hidden_from_local=False)
        parameters = {"benchmark_stratum": "map_crossing_arrival_development_v2",
                      "ego_initial_speed_mps": speed,
                      "ego_post_reference_acceleration_mps2": 0.0,
                      "braking_event_delay_s": 0.2}
        factory = lambda _seed, _config: (spec, parameters)
        for condition_index, condition_name in enumerate(conditions):
            branches = {}
            for method in METHODS:
                branch = run_branch(seed=seed, method=method, config=registration["config"],
                                    condition=network[condition_name],
                                    network_seed=seed + 910000 + 100000 * condition_index,
                                    horizon_s=registration["horizon_s"],
                                    scenario_factory=factory, map_blockers=BLOCKERS)
                branches[method] = branch
                filename = f"{seed}_{condition_name}_{method}.json.gz"
                with gzip.open(OUTPUT / filename, "xt", encoding="utf-8") as handle:
                    json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                artifact_sha256[filename] = sha256_file(OUTPUT / filename)
            periodic, local = (branches[name] for name in METHODS)
            prefix_equal = compare_snapshots(periodic["initial_contract"],
                                             local["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
            engineering = prefix_equal and all(
                branch["actuation_contract_passed"]
                and branch["traffic"]["byte_conservation_ok"]
                and branch["traffic"]["component_conservation_ok"]
                and all(item["used_timestamp_ms"] <= item["available_ms"]
                        for item in branch["input_availability_audit"])
                for branch in branches.values())
            gap = map_gap(periodic, round(registration["config"]["reference_time_s"] * 1000))
            rows.append({"seed": seed, "condition": condition_name,
                         "grid_index": index, "parameters": protocol["grid"][index],
                         "engineering_passed": engineering, "prefix_equal": prefix_equal,
                         **prefix_etas(periodic["initial_contract"]), **gap,
                         "periodic_severe": _severe(periodic), "local_severe": _severe(local),
                         "periodic_progress_m": periodic["route_distance_lower_bound_m"],
                         "local_progress_m": local["route_distance_lower_bound_m"],
                         "periodic_minimum_sampled_clearance_m": minimum_clearance(periodic),
                         "local_minimum_sampled_clearance_m": minimum_clearance(local),
                         "periodic_bytes": periodic["traffic"]["generated_bytes"],
                         "local_bytes": local["traffic"]["generated_bytes"]})
        print(json.dumps({"completed_scenario_clusters": index + 1,
                          "total_scenario_clusters": len(grid)}), flush=True)
    gap_seeds = {r["seed"] for r in rows if r["post_reference_map_blocked_frames"] >= 3
                 and r["first_reveal_after_gap_ms"] is not None}
    utility_cells = {(r["parameters"]["ego_initial_speed_mps"],
                      r["parameters"]["actor_depart_position_m"])
                     for r in rows if not r["periodic_severe"] and r["local_severe"]}
    report = {"status": "DEVELOPMENT_FEASIBILITY_ONLY_NOT_CONFIRMATION",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifact_sha256, "rows": rows,
              "scenario_clusters": len(grid), "branches": len(artifact_sha256),
              "engineering_failures": sum(not r["engineering_passed"] for r in rows),
              "gap_scenarios": len(gap_seeds),
              "communication_utility_scenarios": len({r["seed"] for r in rows
                                                       if not r["periodic_severe"] and r["local_severe"]}),
              "communication_utility_cells": len(utility_cells),
              "feasibility_gate": bool(all(r["engineering_passed"] for r in rows)
                                       and len(gap_seeds) >= 8 and len(utility_cells) >= 4),
              "scientific_success": False, "manuscript_allowed": False,
              "limitations": ["outcome-informed physical timing design on development only",
                              "sampled 0.2-s axis-aligned clearance, not continuous CCD",
                              "idealized RSU world view; no source-faithful sensing",
                              "periodic and local-only only; no BCES/TTL policy result"]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("scenario_clusters", "branches", "engineering_failures",
                       "gap_scenarios", "communication_utility_scenarios",
                       "communication_utility_cells", "feasibility_gate")}, indent=2))


if __name__ == "__main__":
    main()
