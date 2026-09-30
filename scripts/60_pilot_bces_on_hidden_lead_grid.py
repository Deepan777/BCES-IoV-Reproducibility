#!/usr/bin/env python3
"""Evaluate both unchanged frozen BCES and TTL on the full development grid.

All 32 cells are included, not just the cell where periodic beat local-only.
This is a stress pilot, never fresh independent confirmation.
"""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import _severe
from bces.models.bound_policy import FrozenWirePolicy
from bces.network.events import NetworkCondition
from bces.simulation import controlled_closed_loop as loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


BASE = ROOT / "outputs/study_b/hidden_lead_communication_value_development_v1"
OUTPUT = ROOT / "outputs/study_b/hidden_lead_bces_ttl_development_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    base_report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    if len(base_protocol["grid"]) != 32 or len(base_report["rows"]) != 64:
        raise RuntimeError("incomplete full development grid")
    for name, digest in base_report["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("baseline grid artifact changed: " + name)
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    models = {
        method: FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                 policy_hash=registration["policy_hash"])
        for method, rule in registration["model_rules"].items()
    }
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    import torch
    torch.set_num_threads(2)
    protocol = {
        "status": "FULL_GRID_DEVELOPMENT_STRESS_ONLY",
        "base_protocol_sha256": sha256_file(BASE / "protocol.json"),
        "base_report_sha256": sha256_file(BASE / "report.json"),
        "registration_sha256": sha256_file(registration_path),
        "source_sha256": sha256_file(Path(__file__)),
        "methods": ["surface", "scalar_ttl"],
        "grid_indices": list(range(32)),
        "conditions": list(conditions),
        "not_independent_confirmation": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifacts = {}
    rows = []
    original_generator = loop.development_generator
    try:
        for index, cell in enumerate(base_protocol["grid"]):
            seed = 8000000 + index
            spec = replace(scenario_spec("straight_lead_braking"),
                           actor_depart_position_m=cell["actor_depart_position_m"],
                           actor_depart_speed_mps=cell["actor_depart_speed_mps"])
            parameters = {
                "benchmark_stratum": "prospective_hidden_lead_grid_development",
                "ego_initial_speed_mps": cell["ego_initial_speed_mps"],
                "ego_post_reference_acceleration_mps2": 0.0,
                "braking_event_delay_s": cell["braking_event_delay_s"],
            }
            loop.development_generator = lambda: SimpleNamespace(sample_scenario=lambda _seed, _config: (spec, parameters))
            for condition_name, condition in conditions.items():
                baseline_path = BASE / f"{seed}_{condition_name}_periodic_payload.json.gz"
                with gzip.open(baseline_path, "rt", encoding="utf-8") as handle:
                    periodic = json.load(handle)
                for method in protocol["methods"]:
                    branch = loop.run_branch(
                        seed=seed, method=method, config=registration["config"],
                        condition=condition, network_seed=seed + 910000,
                        model=models[method], rule=registration["model_rules"][method],
                        horizon_s=registration["horizon_s"],
                    )
                    if not compare_snapshots(periodic["initial_contract"], branch["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]:
                        raise AssertionError("prefix mismatch")
                    traffic = branch["traffic"]
                    if not traffic["byte_conservation_ok"] or not traffic["component_conservation_ok"]:
                        raise AssertionError("byte accounting mismatch")
                    if any(item["used_timestamp_ms"] > item["available_ms"] for item in branch["input_availability_audit"]):
                        raise AssertionError("future input reached branch")
                    name = f"{seed}_{condition_name}_{method}.json.gz"
                    with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                        json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    artifacts[name] = sha256_file(OUTPUT / name)
                    row = {
                        "seed": seed, "grid_index": index, "condition": condition_name,
                        "parameters": cell, "method": method,
                        "severe": _severe(branch),
                        "periodic_severe": _severe(periodic),
                        "progress_deficit_m": periodic["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"],
                        "adverse_vs_periodic": _severe(branch) or periodic["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"] > 5,
                        "generated_bytes": traffic["generated_bytes"],
                        "reduction_vs_periodic": 1 - traffic["generated_bytes"] / periodic["traffic"]["generated_bytes"],
                        "actuation_passed": branch["actuation_contract_passed"],
                        "reuse_decisions": branch["counts"].get("reuse_decisions", 0),
                    }
                    rows.append(row)
                    print(json.dumps({"grid_index": index, "condition": condition_name,
                                      "method": method, "severe": row["severe"],
                                      "reduction": row["reduction_vs_periodic"]}), flush=True)
    finally:
        loop.development_generator = original_generator
    summary = {}
    for method in protocol["methods"]:
        group = [row for row in rows if row["method"] == method]
        summary[method] = {
            "branches": len(group),
            "severe_branches": sum(row["severe"] for row in group),
            "adverse_vs_periodic_branches": sum(row["adverse_vs_periodic"] for row in group),
            "actuation_failures": sum(not row["actuation_passed"] for row in group),
            "mean_byte_reduction_vs_periodic": sum(row["reduction_vs_periodic"] for row in group) / len(group),
        }
    report = {
        "status": "FULL_GRID_DEVELOPMENT_STRESS_ONLY",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": artifacts,
        "rows": rows,
        "summary": summary,
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
