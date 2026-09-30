#!/usr/bin/env python3
"""Development-only query-gap shadow intervention on all opened crossings."""

from __future__ import annotations

from contextlib import contextmanager
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
from bces.simulation import development_map_loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.development_gap_shadow import (GapShadowPlanner, GapShadowSession,
                                                    SHADOW_ASSUMPTIONS, clear_gap_shadow)
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


BASE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
BASE_REPORT_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
BASE_PROTOCOL_SHA256 = "b85ec159b2ac8323eee2dfd5deeea924302231c6a85ed74346484ba6fc6d58f3"
FROZEN_COMPARISON = ROOT / "outputs/study_b/frozen_bces_ttl_crossing_development_v1"
FROZEN_COMPARISON_SHA256 = "97efa491660881d7c712c1a3fb480cb64e34fd2f88d9d0ad43b5f78197ee1197"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
GAP_SOURCE = ROOT / "bces/simulation/development_gap_shadow.py"
GAP_SOURCE_SHA256 = "1abfa879e17a03a3c34669bb12216f8c238b395842e1ac8b211fcf745f9174d4"
SHADOW_SOURCE = ROOT / "bces/geometry/reachable_shadow.py"
SHADOW_SOURCE_SHA256 = "8cc749044e94392726ae1b9a9bfd1f8d4942a4c29ba8240211e0a44795f0694f"
OUTPUT = ROOT / "outputs/study_b/gap_shadow_crossing_development_v1"
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)
TIMINGS = ("near", "timing_control")


def read_branch(name):
    with gzip.open(BASE / name, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@contextmanager
def gap_binding():
    original_planner = development_map_loop.ControlledDecisionPlanner
    original_session = development_map_loop.BoundSession
    try:
        development_map_loop.ControlledDecisionPlanner = GapShadowPlanner
        development_map_loop.BoundSession = GapShadowSession
        yield
    finally:
        development_map_loop.ControlledDecisionPlanner = original_planner
        development_map_loop.BoundSession = original_session
        clear_gap_shadow()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    expected = ((BASE / "report.json", BASE_REPORT_SHA256),
                (BASE / "protocol.json", BASE_PROTOCOL_SHA256),
                (FROZEN_COMPARISON / "report.json", FROZEN_COMPARISON_SHA256),
                (REGISTRATION, REGISTRATION_SHA256),
                (GAP_SOURCE, GAP_SOURCE_SHA256),
                (SHADOW_SOURCE, SHADOW_SOURCE_SHA256))
    for path, digest in expected:
        if sha256_file(path) != digest:
            raise RuntimeError("frozen input/source changed: " + str(path))
    baseline = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    previous_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    comparison = json.loads((FROZEN_COMPARISON / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if baseline["branches"] != 128 or comparison["branches"] != 64:
        raise RuntimeError("baseline cohort incomplete")
    for directory, report in ((BASE, baseline), (FROZEN_COMPARISON, comparison)):
        for name, digest in report["artifact_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise RuntimeError("baseline branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    rule = registration["model_rules"]["surface"]
    model = FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                             policy_hash=registration["policy_hash"])
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    config = dict(registration["config"])
    config["reference_time_s"] = 4.0
    import torch
    torch.set_num_threads(2)
    protocol = {"status": "FROZEN_GAP_SHADOW_CROSSING_DEVELOPMENT_ONLY_V1",
                "baseline_report_sha256": BASE_REPORT_SHA256,
                "baseline_protocol_sha256": BASE_PROTOCOL_SHA256,
                "frozen_bces_ttl_comparison_sha256": FROZEN_COMPARISON_SHA256,
                "registration_sha256": REGISTRATION_SHA256,
                "gap_source_sha256": GAP_SOURCE_SHA256,
                "shadow_source_sha256": SHADOW_SOURCE_SHA256,
                "script_sha256": sha256_file(Path(__file__)),
                "seeds": [row["seed"] for row in previous_protocol["selected"]],
                "timings": list(TIMINGS), "conditions": list(conditions),
                "branches_expected": 32,
                "assumptions": {"position_error_m": SHADOW_ASSUMPTIONS.position_error_m,
                                "velocity_error_mps": SHADOW_ASSUMPTIONS.velocity_error_mps,
                                "acceleration_bound_mps2": SHADOW_ASSUMPTIONS.acceleration_bound_mps2},
                "shadow_planning_horizon_s": 3.0,
                "gap_rule": "old response-validated payload is used only as inflated obstacle while bound unavailable/violated; never counted as BCES valid reuse",
                "development_success_screen": "all engineering pass, no sampled severe on any periodic-safe branch, >=25% mean modeled byte saving vs periodic, max route-progress deficit vs periodic <=10 m, and no worse sampled severe/bytes than equal-feature TTL",
                "limitations": ["motion bounds are hypotheses and not field-validated",
                                "fallback changes the combined policy but is not yet bound to a new wire identity",
                                "simulated 0.2-s sampled geometry is not continuous-time or public-road safety",
                                "all scenarios are opened synthetic development cases"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifacts, rows = {}, []
    with gap_binding():
        for cluster_index, selected in enumerate(previous_protocol["selected"]):
            seed, speed = selected["seed"], selected["speed_mps"]
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
                    GapShadowPlanner.shadow_plans = 0
                    clear_gap_shadow()
                    branch = development_map_loop.run_branch(
                        seed=seed, method="surface", config=config, condition=condition,
                        network_seed=seed + 970000 + 100000 * condition_index,
                        model=model, rule=rule, horizon_s=registration["horizon_s"],
                        scenario_factory=factory, map_blockers=BLOCKERS)
                    branch["development_fallback"] = "conditional_reachable_shadow_not_wire_bound"
                    shadows = GapShadowPlanner.shadow_plans
                    prefix_equal = compare_snapshots(
                        periodic["initial_contract"]["snapshot"],
                        branch["initial_contract"]["snapshot"],
                        tolerance=1e-6)["equal_within_tolerance"]
                    engineering = bool(prefix_equal and branch["actuation_contract_passed"]
                                       and branch["traffic"]["byte_conservation_ok"]
                                       and branch["traffic"]["component_conservation_ok"]
                                       and all(item["used_timestamp_ms"] <= item["available_ms"]
                                               for item in branch["input_availability_audit"]))
                    filename = f"{seed}_{timing}_{condition_name}_gap_shadow.json.gz"
                    with gzip.open(OUTPUT / filename, "xt", encoding="utf-8") as handle:
                        json.dump(branch, handle, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
                    artifacts[filename] = sha256_file(OUTPUT / filename)
                    bytes_periodic = periodic["traffic"]["generated_bytes"]
                    rows.append({"seed": seed, "timing": timing,
                                 "condition": condition_name,
                                 "periodic_severe": _severe(periodic),
                                 "local_severe": _severe(local),
                                 "shadow_severe": _severe(branch),
                                 "shadow_plans": shadows,
                                 "engineering_passed": engineering,
                                 "prefix_equal": prefix_equal,
                                 "progress_deficit_vs_periodic_m":
                                     periodic["route_distance_lower_bound_m"]
                                     - branch["route_distance_lower_bound_m"],
                                 "periodic_generated_bytes": bytes_periodic,
                                 "shadow_generated_bytes": branch["traffic"]["generated_bytes"],
                                 "byte_reduction_vs_periodic":
                                     1 - branch["traffic"]["generated_bytes"] / bytes_periodic,
                                 "reuse_decisions": branch["counts"].get("reuse_decisions", 0)})
            print(json.dumps({"completed_seed_clusters": cluster_index + 1,
                              "total_seed_clusters": len(previous_protocol["selected"])}), flush=True)
    periodic_safe = [row for row in rows if not row["periodic_severe"]]
    sensitive = [row for row in rows if row["timing"] == "near"
                 and not row["periodic_severe"] and row["local_severe"]]
    mean_saving = sum(row["byte_reduction_vs_periodic"] for row in rows) / len(rows)
    ttl_summary = comparison["summary"]["scalar_ttl"]
    severe_count = sum(row["shadow_severe"] for row in periodic_safe)
    gate = (all(row["engineering_passed"] for row in rows)
            and severe_count == 0 and mean_saving >= 0.25
            and max(row["progress_deficit_vs_periodic_m"] for row in rows) <= 10.0
            and severe_count <= ttl_summary["severe_on_periodic_safe"]
            and mean_saving >= ttl_summary["mean_byte_reduction_vs_periodic"])
    summary = {"branches": len(rows),
               "engineering_passed": all(row["engineering_passed"] for row in rows),
               "periodic_safe_branches": len(periodic_safe),
               "severe_on_periodic_safe": severe_count,
               "communication_sensitive_near_branches": len(sensitive),
               "severe_on_communication_sensitive_near": sum(row["shadow_severe"] for row in sensitive),
               "mean_byte_reduction_vs_periodic": mean_saving,
               "max_progress_deficit_vs_periodic_m":
                   max(row["progress_deficit_vs_periodic_m"] for row in rows),
               "total_shadow_plans": sum(row["shadow_plans"] for row in rows),
               "ttl_comparator_summary": ttl_summary,
               "development_success_screen_passed": gate}
    report = {"status": "GAP_SHADOW_CROSSING_DEVELOPMENT_ONLY_V1",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows, "summary": summary,
              "branches": len(artifacts), "not_confirmatory": True,
              "not_wire_policy_bound": True,
              "no_deployment_or_regret_certificate": True}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
