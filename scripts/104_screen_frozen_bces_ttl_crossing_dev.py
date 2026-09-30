#!/usr/bin/env python3
"""Development-only fair BCES/TTL screen on every existing crossing seed."""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import _severe
from bces.models.bound_policy import FrozenWirePolicy
from bces.network.events import NetworkCondition
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation import development_map_loop
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


BASE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
BASE_REPORT_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
BASE_PROTOCOL_SHA256 = "b85ec159b2ac8323eee2dfd5deeea924302231c6a85ed74346484ba6fc6d58f3"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
OUTPUT = ROOT / "outputs/study_b/frozen_bces_ttl_crossing_development_v1"
METHODS = ("surface", "scalar_ttl")
TIMINGS = ("near", "timing_control")
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)


def read_branch(name):
    with gzip.open(BASE / name, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(BASE / "report.json") != BASE_REPORT_SHA256:
        raise RuntimeError("development baseline report changed")
    if sha256_file(BASE / "protocol.json") != BASE_PROTOCOL_SHA256:
        raise RuntimeError("development baseline protocol changed")
    if sha256_file(REGISTRATION) != REGISTRATION_SHA256:
        raise RuntimeError("registered v2 policy specification changed")
    baseline = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    previous_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if baseline["branches"] != 128 or len(previous_protocol["selected"]) != 8:
        raise RuntimeError("baseline crossing screen incomplete")
    for name, digest in baseline["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("baseline branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    models = {method: FrozenWirePolicy(ROOT / rule["path"],
                                       expected_sha256=rule["sha256"],
                                       policy_hash=registration["policy_hash"])
              for method, rule in registration["model_rules"].items()
              if method in METHODS}
    if set(models) != set(METHODS):
        raise RuntimeError("frozen equal-feature BCES/TTL models missing")
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    config = dict(registration["config"])
    config["reference_time_s"] = 4.0
    import torch
    torch.set_num_threads(2)
    protocol = {"status": "FROZEN_BCES_TTL_CROSSING_DEVELOPMENT_V1",
                "baseline_report_sha256": BASE_REPORT_SHA256,
                "baseline_protocol_sha256": BASE_PROTOCOL_SHA256,
                "registration_sha256": REGISTRATION_SHA256,
                "script_sha256": sha256_file(Path(__file__)),
                "seeds": [row["seed"] for row in previous_protocol["selected"]],
                "timings": list(TIMINGS), "conditions": list(conditions),
                "methods": list(METHODS), "branches_expected": 64,
                "planner": "original registered-v2-compatible planner only; recovery planner has a different policy hash",
                "network_seed": "seed + 970000 + 100000*condition_index, exactly as baseline",
                "all_seeds_included": True,
                "success_screen": "method must pass all engineering checks, avoid severe endpoints on every periodic-safe near/control branch, avoid severe endpoints on all baseline communication-sensitive near branches, save >=25% mean modeled bytes vs periodic across all 32 paired scene-condition cells, and have maximum route-progress deficit vs periodic <=10 m",
                "severe_endpoint": "frozen sampled geometric overlap/SUMO collision/actuation/planner endpoint",
                "cluster_unit": "scenario seed, not branches or frames",
                "no_independent_confirmation": True,
                "known_baseline_failure": "earlier scenario feasibility gate failed on one periodic-severe seed and mean periodic-vs-local progress deficit"}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifacts, rows = {}, []
    for cluster_index, selected in enumerate(previous_protocol["selected"]):
        seed = selected["seed"]
        speed = selected["speed_mps"]
        for timing in TIMINGS:
            position = selected[f"{timing if timing == 'near' else 'control'}_position_m"]
            spec = replace(scenario_spec("unprotected_crossing"),
                           actor_depart_position_m=position,
                           actor_depart_speed_mps=9.0, hidden_from_local=False)
            parameters = {"benchmark_stratum": "prefix_eta_solver_development_v1",
                          "ego_initial_speed_mps": speed,
                          "ego_post_reference_acceleration_mps2": 0.0,
                          "braking_event_delay_s": 0.2}
            factory = lambda _seed, _config: (spec, parameters)
            for condition_index, (condition_name, condition) in enumerate(conditions.items()):
                periodic = read_branch(f"{seed}_{timing}_{condition_name}_original_periodic_payload.json.gz")
                local = read_branch(f"{seed}_{timing}_{condition_name}_original_local_only.json.gz")
                if not compare_snapshots(periodic["initial_contract"]["snapshot"],
                                         local["initial_contract"]["snapshot"],
                                         tolerance=1e-6)["equal_within_tolerance"]:
                    raise RuntimeError("baseline branch prefix mismatch")
                for method in METHODS:
                    branch = development_map_loop.run_branch(
                        seed=seed, method=method, config=config, condition=condition,
                        network_seed=seed + 970000 + 100000 * condition_index,
                        model=models[method], rule=registration["model_rules"][method],
                        horizon_s=registration["horizon_s"],
                        scenario_factory=factory, map_blockers=BLOCKERS)
                    prefix_equal = compare_snapshots(
                        periodic["initial_contract"]["snapshot"],
                        branch["initial_contract"]["snapshot"],
                        tolerance=1e-6)["equal_within_tolerance"]
                    engineering = bool(prefix_equal and branch["actuation_contract_passed"]
                                       and branch["traffic"]["byte_conservation_ok"]
                                       and branch["traffic"]["component_conservation_ok"]
                                       and all(item["used_timestamp_ms"] <= item["available_ms"]
                                               for item in branch["input_availability_audit"]))
                    name = f"{seed}_{timing}_{condition_name}_{method}.json.gz"
                    with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                        json.dump(branch, handle, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
                    artifacts[name] = sha256_file(OUTPUT / name)
                    periodic_bytes = periodic["traffic"]["generated_bytes"]
                    rows.append({"seed": seed, "timing": timing,
                                 "condition": condition_name, "method": method,
                                 "prefix_equal": prefix_equal,
                                 "engineering_passed": engineering,
                                 "periodic_severe": _severe(periodic),
                                 "local_severe": _severe(local),
                                 "policy_severe": _severe(branch),
                                 "periodic_progress_m": periodic["route_distance_lower_bound_m"],
                                 "policy_progress_m": branch["route_distance_lower_bound_m"],
                                 "progress_deficit_vs_periodic_m":
                                     periodic["route_distance_lower_bound_m"]
                                     - branch["route_distance_lower_bound_m"],
                                 "periodic_generated_bytes": periodic_bytes,
                                 "policy_generated_bytes": branch["traffic"]["generated_bytes"],
                                 "byte_reduction_vs_periodic":
                                     1 - branch["traffic"]["generated_bytes"] / periodic_bytes,
                                 "reuse_decisions": branch["counts"].get("reuse_decisions", 0)})
        print(json.dumps({"completed_seed_clusters": cluster_index + 1,
                          "total_seed_clusters": len(previous_protocol["selected"])}), flush=True)
    summary = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        periodic_safe = [row for row in group if not row["periodic_severe"]]
        sensitive = [row for row in group if row["timing"] == "near"
                     and not row["periodic_severe"] and row["local_severe"]]
        mean_saving = sum(row["byte_reduction_vs_periodic"] for row in group) / len(group)
        max_deficit = max(row["progress_deficit_vs_periodic_m"] for row in group)
        passed = (all(row["engineering_passed"] for row in group)
                  and all(not row["policy_severe"] for row in periodic_safe)
                  and all(not row["policy_severe"] for row in sensitive)
                  and mean_saving >= 0.25 and max_deficit <= 10.0)
        summary[method] = {"branches": len(group),
                           "engineering_passed": all(row["engineering_passed"] for row in group),
                           "periodic_safe_branches": len(periodic_safe),
                           "severe_on_periodic_safe": sum(row["policy_severe"] for row in periodic_safe),
                           "communication_sensitive_near_branches": len(sensitive),
                           "severe_on_communication_sensitive_near":
                               sum(row["policy_severe"] for row in sensitive),
                           "mean_byte_reduction_vs_periodic": mean_saving,
                           "max_progress_deficit_vs_periodic_m": max_deficit,
                           "development_success_screen_passed": passed}
    report = {"status": "FROZEN_BCES_TTL_CROSSING_DEVELOPMENT_ONLY_V1",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows, "summary": summary,
              "branches": len(artifacts), "not_independent_confirmation": True,
              "not_cross_site": True,
              "limitations": ["synthetic map obstruction and idealized sender",
                              "scenario construction tuned on development prefix and driving outcomes",
                              "periodic itself severe in one prior near seed",
                              "sampled geometry not verified physical collision",
                              "original planner only; recovery planner requires separately trained policies"]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
