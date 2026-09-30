#!/usr/bin/env python3
"""Predeclared development feasibility screen, not a policy confirmation."""

from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import itertools
import json
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


OUTPUT = ROOT / "outputs/study_b/map_crossing_utility_development_v1"


def map_gap(branch: dict, reference_ms: int) -> dict:
    rows = branch["visibility_audit"]
    blocked = [row["timestamp_ms"] for row in rows
               if row["timestamp_ms"] >= reference_ms
               and row["target_within_local_range"]
               and row["target_visible_without_map"]
               and not row["target_locally_visible"]]
    first_reveal = next((row["timestamp_ms"] for row in rows
                         if blocked and row["timestamp_ms"] > blocked[-1]
                         and row["target_locally_visible"]), None)
    return {"post_reference_map_blocked_frames": len(blocked),
            "first_map_blocked_ms": blocked[0] if blocked else None,
            "first_reveal_after_gap_ms": first_reveal}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    grid = list(itertools.product((0.0, 5.0, 10.0, 15.0), (12.0, 14.0), (0, 1)))
    blockers = (StaticMapObstruction("southwest_corner", 93.0, 89.5,
                                      0.0, 8.0, 6.0),)
    methods = ("periodic_payload", "local_only")
    conditions = tuple(registration["condition_order"])
    protocol = {
        "status": "DEVELOPMENT_FEASIBILITY_ONLY_NOT_CONFIRMATION",
        "registration_sha256": sha256_file(registration_path),
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility_map.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "seeds": [8300000 + i for i in range(len(grid))],
        "grid": [{"actor_depart_position_m": position,
                  "ego_initial_speed_mps": ego_speed,
                  "replicate": replicate}
                 for position, ego_speed, replicate in grid],
        "scenario_family": "unprotected_crossing",
        "actor_depart_speed_mps": 9.0,
        "braking_event_delay_s": 0.2,
        "blockers": [asdict(item) for item in blockers],
        "methods": list(methods),
        "conditions": list(conditions),
        "selection": "all 16 scenario seeds, both conditions, both methods; no result-based cell filtering",
        "feasibility_rule": "at least four seeds with a >=3-frame post-reference map gap and later reveal, and at least three seeds with periodic-safe/local-severe or >=5 m periodic progress advantage while both safe",
        "sender_limit": "ideal centered-world cooperative range; RSU occlusion is not modeled",
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    network = {name: NetworkCondition(**registration["conditions"][name])
               for name in conditions}

    rows = []
    artifacts = {}
    for index, (position, ego_speed, replicate) in enumerate(grid):
        seed = 8300000 + index
        spec = replace(scenario_spec("unprotected_crossing"),
                       actor_depart_position_m=position,
                       actor_depart_speed_mps=9.0,
                       hidden_from_local=False)
        parameters = {
            "benchmark_stratum": "map_crossing_development_feasibility",
            "ego_initial_speed_mps": ego_speed,
            "ego_post_reference_acceleration_mps2": 0.0,
            "braking_event_delay_s": 0.2,
        }
        factory = lambda _seed, _config: (spec, parameters)
        for condition_index, condition_name in enumerate(conditions):
            branches = {}
            for method in methods:
                branch = run_branch(seed=seed, method=method, config=registration["config"],
                                    condition=network[condition_name],
                                    network_seed=seed + 910000 + 100000 * condition_index,
                                    horizon_s=registration["horizon_s"],
                                    scenario_factory=factory, map_blockers=blockers)
                branches[method] = branch
                name = f"{seed}_{condition_name}_{method}.json.gz"
                with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                    json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                artifacts[name] = sha256_file(OUTPUT / name)
            periodic, local = branches[methods[0]], branches[methods[1]]
            prefix = compare_snapshots(periodic["initial_contract"],
                                       local["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
            engineering = prefix and all(
                branch["actuation_contract_passed"]
                and branch["traffic"]["byte_conservation_ok"]
                and branch["traffic"]["component_conservation_ok"]
                and all(item["used_timestamp_ms"] <= item["available_ms"]
                        for item in branch["input_availability_audit"])
                for branch in branches.values())
            gap = map_gap(periodic, round(registration["config"]["reference_time_s"] * 1000))
            periodic_severe, local_severe = _severe(periodic), _severe(local)
            progress_advantage = (periodic["route_distance_lower_bound_m"]
                                  - local["route_distance_lower_bound_m"])
            rows.append({"seed": seed, "grid_index": index,
                         "condition": condition_name, "replicate": replicate,
                         "parameters": protocol["grid"][index],
                         "engineering_passed": engineering,
                         **gap,
                         "periodic_severe": periodic_severe,
                         "local_severe": local_severe,
                         "periodic_progress_m": periodic["route_distance_lower_bound_m"],
                         "local_progress_m": local["route_distance_lower_bound_m"],
                         "periodic_progress_advantage_m": progress_advantage,
                         "periodic_bytes": periodic["traffic"]["generated_bytes"],
                         "local_bytes": local["traffic"]["generated_bytes"]})
        print(json.dumps({"completed_scenario_clusters": index + 1,
                          "total_scenario_clusters": len(grid)}), flush=True)

    gap_seeds = {row["seed"] for row in rows
                 if row["post_reference_map_blocked_frames"] >= 3
                 and row["first_reveal_after_gap_ms"] is not None}
    utility_seeds = {row["seed"] for row in rows
                     if (not row["periodic_severe"] and row["local_severe"])
                     or (not row["periodic_severe"] and not row["local_severe"]
                         and row["periodic_progress_advantage_m"] >= 5.0)}
    report = {
        "status": "DEVELOPMENT_FEASIBILITY_ONLY_NOT_CONFIRMATION",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": artifacts,
        "rows": rows,
        "scenario_clusters": len(grid),
        "branches": len(rows) * len(methods),
        "engineering_failures": sum(not row["engineering_passed"] for row in rows),
        "gap_scenarios": len(gap_seeds),
        "communication_utility_scenarios": len(utility_seeds),
        "feasibility_gate": bool(all(row["engineering_passed"] for row in rows)
                                 and len(gap_seeds) >= 4 and len(utility_seeds) >= 3),
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "scenario_clusters", "branches", "engineering_failures",
        "gap_scenarios", "communication_utility_scenarios", "feasibility_gate"
    )}, indent=2))


if __name__ == "__main__":
    main()
