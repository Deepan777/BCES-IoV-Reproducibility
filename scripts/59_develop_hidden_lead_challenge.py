#!/usr/bin/env python3
"""Prospective parameter-grid development audit of communication value.

This is NOT confirmation. It does not edit the registered generator or use
registered confirmation seeds. Only periodic and local-only are compared to
test whether this scenario mechanism can expose V2X decision utility at all.
"""

from __future__ import annotations

from dataclasses import replace
import gzip
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import _severe
from bces.network.events import NetworkCondition
from bces.simulation import controlled_closed_loop as loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/hidden_lead_communication_value_development_v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    # Declared before seeing any outcome. Values place a hidden lead actor
    # near the ego's expected 5 s reference position; no per-case substitution.
    grid = list(itertools.product((50.0, 60.0, 70.0, 80.0),
                                  (10.0, 14.0), (4.0, 8.0), (0.2, 1.0)))
    protocol = {
        "status": "DEVELOPMENT_GRID_ONLY_NO_CONFIRMATORY_INFERENCE",
        "registration_sha256": sha256_file(registration_path),
        "source_sha256": sha256_file(Path(__file__)),
        "grid": [dict(actor_depart_position_m=position, ego_initial_speed_mps=ego,
                      actor_depart_speed_mps=actor, braking_event_delay_s=delay)
                 for position, ego, actor, delay in grid],
        "methods": ["periodic_payload", "local_only"],
        "conditions": list(conditions),
        "seed_rule": "8000000 + grid index; disjoint from all registered confirmation seeds",
        "selection": "all grid cells, both conditions and methods; no adaptive replacement",
        "purpose": "screen whether cooperative information changes driving outcome before designing a harder test",
        "scientific_success": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    original_generator = loop.development_generator
    rows = []
    artifacts = {}
    try:
        for index, (position, ego_speed, actor_speed, delay) in enumerate(grid):
            seed = 8000000 + index
            spec = replace(scenario_spec("straight_lead_braking"),
                           actor_depart_position_m=position,
                           actor_depart_speed_mps=actor_speed)
            parameters = {
                "benchmark_stratum": "prospective_hidden_lead_grid_development",
                "ego_initial_speed_mps": ego_speed,
                "ego_post_reference_acceleration_mps2": 0.0,
                "braking_event_delay_s": delay,
            }
            loop.development_generator = lambda: SimpleNamespace(sample_scenario=lambda _seed, _config: (spec, parameters))
            for condition_name, condition in conditions.items():
                branches = {}
                for method in protocol["methods"]:
                    branch = loop.run_branch(
                        seed=seed, method=method, config=registration["config"],
                        condition=condition, network_seed=seed + 910000,
                        horizon_s=registration["horizon_s"],
                    )
                    branches[method] = branch
                    name = f"{seed}_{condition_name}_{method}.json.gz"
                    with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                        json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    artifacts[name] = sha256_file(OUTPUT / name)
                periodic, local = branches["periodic_payload"], branches["local_only"]
                if not compare_snapshots(periodic["initial_contract"], local["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]:
                    raise AssertionError("prefix mismatch")
                if any(not branch["traffic"]["byte_conservation_ok"] or not branch["traffic"]["component_conservation_ok"]
                       for branch in branches.values()):
                    raise AssertionError("byte accounting mismatch")
                if any(any(item["used_timestamp_ms"] > item["available_ms"]
                           for item in branch["input_availability_audit"]) for branch in branches.values()):
                    raise AssertionError("future information reached a branch")
                row = {
                    "seed": seed, "grid_index": index, "condition": condition_name,
                    "parameters": protocol["grid"][index],
                    "periodic_severe": _severe(periodic),
                    "local_severe": _severe(local),
                    "periodic_outcome": periodic["outcome"],
                    "local_outcome": local["outcome"],
                    "periodic_actuation_passed": periodic["actuation_contract_passed"],
                    "local_actuation_passed": local["actuation_contract_passed"],
                    "periodic_distance_m": periodic["route_distance_lower_bound_m"],
                    "local_distance_m": local["route_distance_lower_bound_m"],
                    "local_progress_deficit_m": periodic["route_distance_lower_bound_m"] - local["route_distance_lower_bound_m"],
                    "periodic_geometric_overlap_ticks": periodic["counts"].get("geometric_overlap_ticks", 0),
                    "local_geometric_overlap_ticks": local["counts"].get("geometric_overlap_ticks", 0),
                    "periodic_bytes": periodic["traffic"]["generated_bytes"],
                    "local_bytes": local["traffic"]["generated_bytes"],
                }
                rows.append(row)
                print(json.dumps({"grid_index": index, "condition": condition_name,
                                  "periodic_severe": row["periodic_severe"],
                                  "local_severe": row["local_severe"],
                                  "progress_deficit_m": row["local_progress_deficit_m"]}), flush=True)
    finally:
        loop.development_generator = original_generator
    report = {
        "status": "DEVELOPMENT_GRID_ONLY_NO_CONFIRMATORY_INFERENCE",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": artifacts,
        "rows": rows,
        "cells": len(grid),
        "periodic_severe_branches": sum(row["periodic_severe"] for row in rows),
        "local_severe_branches": sum(row["local_severe"] for row in rows),
        "periodic_safe_local_severe_branches": sum(not row["periodic_severe"] and row["local_severe"] for row in rows),
        "local_progress_deficit_over_5m_branches": sum(row["local_progress_deficit_m"] > 5 for row in rows),
        "actuation_failures": sum(not row["periodic_actuation_passed"] or not row["local_actuation_passed"] for row in rows),
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cells", "periodic_severe_branches", "local_severe_branches", "periodic_safe_local_severe_branches", "local_progress_deficit_over_5m_branches", "actuation_failures")}, indent=2))


if __name__ == "__main__":
    main()
