#!/usr/bin/env python3
"""Paired development-only branches from frozen prefix-selected crossings."""

from __future__ import annotations

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
from bces.simulation.development_map_loop import run_branch
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/prefix_matched_crossing_utility_development_v2"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
SOLVER_DIR = ROOT / "outputs/study_b/prefix_eta_solver_development_v2"
SOLVER_REPORT_SHA256 = "4b7523e7bcea917f5978d48a075bb615e646f878712aced82aabaa55f2213a3c"
METRICS_HELPER = ROOT / "scripts/85_screen_crossing_arrival_development_v2.py"
METRICS_HELPER_SHA256 = "fec358a9c43828170bb3c158c3d1908714cb35040f1252da0f7ab6626847ce29"
UNRUN_DRAFT = ROOT / "scripts/88_screen_prefix_matched_crossing_dev.py"
METHODS = ("periodic_payload", "local_only")
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)


def metrics_helper():
    if sha256_file(METRICS_HELPER) != METRICS_HELPER_SHA256:
        raise RuntimeError("frozen metrics helper changed")
    spec = importlib.util.spec_from_file_location("crossing_metrics_for_v2", METRICS_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen metrics helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    metrics = metrics_helper()
    if sha256_file(SOLVER_DIR / "report.json") != SOLVER_REPORT_SHA256:
        raise RuntimeError("prefix-only solver report changed")
    solver = json.loads((SOLVER_DIR / "report.json").read_text(encoding="utf-8"))
    if not solver["solver_feasible"] or len(solver["rows"]) != 8 or not solver["no_policy_outcomes"]:
        raise RuntimeError("wrong prefix-only solver eligibility")
    for name, digest in solver["artifact_sha256"].items():
        if sha256_file(SOLVER_DIR / name) != digest:
            raise RuntimeError("prefix snapshot changed: " + name)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    config = dict(registration["config"])
    config["reference_time_s"] = 4.0
    conditions = tuple(registration["condition_order"])
    selected = sorted(solver["rows"], key=lambda row: row["seed"])
    if len({row["seed"] for row in selected}) != 8 or any(
            row["status"] != "target_hit_on_approach" for row in selected):
        raise RuntimeError("prefix selection mismatch")
    protocol = {
        "status": "FROZEN_PREFIX_MATCHED_DEVELOPMENT_FEASIBILITY_ONLY_V2",
        "registration_sha256": sha256_file(REGISTRATION),
        "solver_report_sha256": SOLVER_REPORT_SHA256,
        "metrics_helper_sha256": METRICS_HELPER_SHA256,
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility_map.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "unrun_draft_sha256": sha256_file(UNRUN_DRAFT),
        "unrun_draft_issue": "static field-name error in blocker serialization, detected before any branch outcomes",
        "seeds": [row["seed"] for row in selected],
        "selected_positions_m": [row["selected_position_m"] for row in selected],
        "ego_speeds_mps": [row["ego_speed_setting_mps"] for row in selected],
        "reference_time_s": 4.0, "actor_depart_speed_mps": 9.0,
        "braking_event_delay_s": 0.2,
        "methods": list(METHODS), "conditions": list(conditions),
        "blockers": [asdict(item) for item in BLOCKERS],
        "selection": "all eight verified prefix-only solver seeds, both conditions, both methods; no branch-outcome filtering",
        "feasibility_rule": "all engineering checks; >=4 seeds with >=3 post-reference map-blocked frames and later reveal; >=3 periodic-safe/local-severe seeds spanning >=2 ego-speed settings; mean utility-seed periodic progress deficit <=10 m",
        "sender_limit": "ideal centered-world cooperative range; no separate RSU view model",
        "no_bces_or_ttl_outcomes": True, "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    network = {name: NetworkCondition(**registration["conditions"][name])
               for name in conditions}
    rows, artifact_sha256 = [], {}
    for index, chosen in enumerate(selected):
        seed = chosen["seed"]
        speed = chosen["ego_speed_setting_mps"]
        position = chosen["selected_position_m"]
        spec = replace(scenario_spec("unprotected_crossing"),
                       actor_depart_position_m=position,
                       actor_depart_speed_mps=9.0, hidden_from_local=False)
        parameters = {"benchmark_stratum": "prefix_matched_crossing_utility_development_v2",
                      "ego_initial_speed_mps": speed,
                      "ego_post_reference_acceleration_mps2": 0.0,
                      "braking_event_delay_s": 0.2}
        factory = lambda _seed, _config: (spec, parameters)
        frozen_prefix = json.loads((SOLVER_DIR / f"{seed}_prefix_2.json").read_text(encoding="utf-8"))
        for condition_index, condition_name in enumerate(conditions):
            branches = {}
            for method in METHODS:
                branch = run_branch(seed=seed, method=method, config=config,
                                    condition=network[condition_name],
                                    network_seed=seed + 910000 + 100000 * condition_index,
                                    horizon_s=registration["horizon_s"],
                                    scenario_factory=factory, map_blockers=BLOCKERS)
                branches[method] = branch
                filename = f"{seed}_{condition_name}_{method}.json.gz"
                with gzip.open(OUTPUT / filename, "xt", encoding="utf-8") as handle:
                    json.dump(branch, handle, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
                artifact_sha256[filename] = sha256_file(OUTPUT / filename)
            periodic, local = (branches[name] for name in METHODS)
            prefix_equal = compare_snapshots(periodic["initial_contract"],
                                             local["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
            solver_prefix_equal = all(compare_snapshots(
                frozen_prefix["snapshot"], branch["initial_contract"]["snapshot"],
                tolerance=1e-6)["equal_within_tolerance"] for branch in branches.values())
            engineering = prefix_equal and solver_prefix_equal and all(
                branch["actuation_contract_passed"]
                and branch["traffic"]["byte_conservation_ok"]
                and branch["traffic"]["component_conservation_ok"]
                and all(item["used_timestamp_ms"] <= item["available_ms"]
                        for item in branch["input_availability_audit"])
                for branch in branches.values())
            gap = metrics.map_gap(periodic, round(config["reference_time_s"] * 1000))
            rows.append({"seed": seed, "condition": condition_name,
                         "grid_index": index, "ego_initial_speed_mps": speed,
                         "actor_depart_position_m": position,
                         "engineering_passed": engineering,
                         "prefix_equal": prefix_equal,
                         "solver_prefix_equal": solver_prefix_equal,
                         **metrics.prefix_etas(periodic["initial_contract"]), **gap,
                         "periodic_severe": _severe(periodic), "local_severe": _severe(local),
                         "periodic_progress_m": periodic["route_distance_lower_bound_m"],
                         "local_progress_m": local["route_distance_lower_bound_m"],
                         "periodic_minimum_sampled_clearance_m": metrics.minimum_clearance(periodic),
                         "local_minimum_sampled_clearance_m": metrics.minimum_clearance(local),
                         "periodic_bytes": periodic["traffic"]["generated_bytes"],
                         "local_bytes": local["traffic"]["generated_bytes"]})
        print(json.dumps({"completed_scenario_clusters": index + 1,
                          "total_scenario_clusters": len(selected)}), flush=True)
    gap_seeds = {row["seed"] for row in rows
                 if row["post_reference_map_blocked_frames"] >= 3
                 and row["first_reveal_after_gap_ms"] is not None}
    utility = [row for row in rows
               if not row["periodic_severe"] and row["local_severe"]]
    utility_seeds = {row["seed"] for row in utility}
    utility_speeds = {row["ego_initial_speed_mps"] for row in utility}
    progress_cost_by_seed = {seed: sum(max(0.0, row["local_progress_m"] - row["periodic_progress_m"])
                                       for row in utility if row["seed"] == seed) /
                             sum(row["seed"] == seed for row in utility)
                             for seed in utility_seeds}
    mean_utility_progress_cost = (sum(progress_cost_by_seed.values()) / len(progress_cost_by_seed)
                                  if progress_cost_by_seed else None)
    report = {"status": "PREFIX_MATCHED_DEVELOPMENT_FEASIBILITY_ONLY_V2",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifact_sha256, "rows": rows,
              "scenario_clusters": len(selected), "branches": len(artifact_sha256),
              "engineering_failures": sum(not row["engineering_passed"] for row in rows),
              "gap_scenarios": len(gap_seeds),
              "communication_utility_scenarios": len(utility_seeds),
              "communication_utility_ego_speeds": sorted(utility_speeds),
              "mean_utility_progress_cost_m": mean_utility_progress_cost,
              "feasibility_gate": bool(all(row["engineering_passed"] for row in rows)
                                       and len(gap_seeds) >= 4
                                       and len(utility_seeds) >= 3
                                       and len(utility_speeds) >= 2
                                       and mean_utility_progress_cost is not None
                                       and mean_utility_progress_cost <= 10.0),
              "scientific_success": False, "manuscript_allowed": False,
              "limitations": ["selected on pre-decision timing only, but development seeds already opened",
                              "idealized sender view, 2-D local obstruction",
                              "no BCES/TTL policy outcome or independent confirmation"]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("scenario_clusters", "branches", "engineering_failures",
                       "gap_scenarios", "communication_utility_scenarios",
                       "communication_utility_ego_speeds",
                       "mean_utility_progress_cost_m", "feasibility_gate")}, indent=2))


if __name__ == "__main__":
    main()
