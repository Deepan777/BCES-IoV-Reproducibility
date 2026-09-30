#!/usr/bin/env python3
"""Development-only lead-geometry ablation of the fixed adaptive margin gate.

This preserves the existing gate and frozen v2 sources. It removes only the
margin threshold for a distinct packet-bound policy on the already-open 32-cell
synthetic hidden-lead development grid. No result is confirmatory.
"""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import _severe
from bces.models.bound_policy import FrozenWirePolicy
import bces.network.adaptive_margin_policy as margin_module
from bces.network.adaptive_margin_policy import AdaptiveMarginWirePolicy
from bces.network.events import NetworkCondition
from bces.simulation import controlled_closed_loop as loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import canonical_json_hash, sha256_file

BASE = ROOT / "outputs/study_b/hidden_lead_communication_value_development_v1"
ADAPTIVE = ROOT / "outputs/study_b/adaptive_margin_hidden_grid_development_v1"
OUTPUT = ROOT / "outputs/study_b/lead_only_hidden_grid_ablation_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"


def lead_only(reference: dict) -> tuple[str, dict]:
    _, evidence = ORIGINAL_SELECTOR(reference)
    return ("scalar_ttl" if evidence["lead_like"] else "surface"), evidence


ORIGINAL_SELECTOR = margin_module.select_reference_method


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    base_report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    adaptive_report = json.loads((ADAPTIVE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if len(base_protocol["grid"]) != 32 or len(base_report["rows"]) != 64 or len(adaptive_report["rows"]) != 64:
        raise RuntimeError("incomplete fixed development grid")
    for folder, report in ((BASE, base_report), (ADAPTIVE, adaptive_report)):
        for name, digest in report["artifact_sha256"].items():
            if sha256_file(folder / name) != digest:
                raise RuntimeError("pre-existing artifact changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)

    protocol = {
        "status": "FROZEN_DEVELOPMENT_ABLATION_ONLY",
        "question": "Does removing only the margin threshold improve or degrade the fixed adaptive rule on the already-open synthetic hidden-lead grid?",
        "ablation": "choose learned TTL for every lead-like reference regardless of first-competitor total-cost gap; otherwise unchanged surface",
        "base_protocol_sha256": sha256_file(BASE / "protocol.json"),
        "base_report_sha256": sha256_file(BASE / "report.json"),
        "adaptive_report_sha256": sha256_file(ADAPTIVE / "report.json"),
        "v2_registration_sha256": sha256_file(REGISTRATION),
        "adaptive_source_sha256": sha256_file(ROOT / "bces/network/adaptive_margin_policy.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "grid_indices": list(range(32)),
        "conditions": list(registration["condition_order"]),
        "unit": "parameter cell with two nested channel conditions",
        "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    models = {method: FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                       policy_hash=registration["policy_hash"])
              for method, rule in registration["model_rules"].items()}
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    import torch
    torch.set_num_threads(2)
    rows, artifacts = [], {}
    prior = {(row["grid_index"], row["condition"]): row for row in adaptive_report["rows"]}
    for index, cell in enumerate(base_protocol["grid"]):
        seed = 8000000 + index
        spec = replace(scenario_spec("straight_lead_braking"),
                       actor_depart_position_m=cell["actor_depart_position_m"],
                       actor_depart_speed_mps=cell["actor_depart_speed_mps"])
        parameters = {"benchmark_stratum": "prospective_hidden_lead_grid_development",
                      "ego_initial_speed_mps": cell["ego_initial_speed_mps"],
                      "ego_post_reference_acceleration_mps2": 0.0,
                      "braking_event_delay_s": cell["braking_event_delay_s"]}
        generator = SimpleNamespace(sample_scenario=lambda _seed, _config: (spec, parameters))
        for condition_name, condition in conditions.items():
            policy = AdaptiveMarginWirePolicy(
                surface=models["surface"], scalar_ttl=models["scalar_ttl"],
                shrinkages={name: rule["shrinkage"] for name, rule in registration["model_rules"].items()})
            policy.sha256 = canonical_json_hash({
                "kind": "development_lead_only_ablation_v1",
                "parent_policy_sha256": policy.sha256,
                "script_sha256": protocol["script_sha256"],
            })
            with (patch.object(loop, "development_generator", lambda: generator),
                  patch.object(loop, "bind_reference", policy.bind),
                  patch.object(margin_module, "select_reference_method", lead_only)):
                branch = loop.run_branch(seed=seed, method="surface", config=registration["config"],
                    condition=condition, network_seed=seed + 910000,
                    model=policy, rule=registration["model_rules"]["surface"],
                    horizon_s=registration["horizon_s"])
            branch["method"] = "lead_only_ablation"
            branch["ablation_selections"] = policy.selections
            with gzip.open(BASE / f"{seed}_{condition_name}_periodic_payload.json.gz", "rt", encoding="utf-8") as handle:
                periodic = json.load(handle)
            with gzip.open(BASE / f"{seed}_{condition_name}_local_only.json.gz", "rt", encoding="utf-8") as handle:
                local_only = json.load(handle)
            if (not compare_snapshots(periodic["initial_contract"], branch["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
                    or not branch["actuation_contract_passed"]
                    or not branch["traffic"]["byte_conservation_ok"]
                    or not branch["traffic"]["component_conservation_ok"]
                    or any(item["used_timestamp_ms"] > item["available_ms"] for item in branch["input_availability_audit"])):
                raise RuntimeError("lead-only ablation engineering/causal gate failed")
            name = f"{seed}_{condition_name}_lead_only_ablation.json.gz"
            with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            artifacts[name] = sha256_file(OUTPUT / name)
            baseline = prior[index, condition_name]
            if baseline["seed"] != seed or baseline["periodic_generated_bytes"] != periodic["traffic"]["generated_bytes"]:
                raise RuntimeError("adaptive baseline not paired")
            rows.append({
                "seed": seed, "grid_index": index, "condition": condition_name,
                "lead_only_severe": _severe(branch),
                "adaptive_severe": baseline["adaptive_severe"],
                "local_only_severe": _severe(local_only),
                "lead_only_generated_bytes": branch["traffic"]["generated_bytes"],
                "adaptive_generated_bytes": baseline["adaptive_generated_bytes"],
                "periodic_generated_bytes": periodic["traffic"]["generated_bytes"],
                "lead_only_progress_deficit_m": periodic["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"],
                "ttl_bound_count": sum(item["family"] == "scalar_ttl" for item in policy.selections),
                "surface_bound_count": sum(item["family"] == "surface" for item in policy.selections),
                "not_confirmatory": True,
            })
            print(json.dumps({"completed": len(rows), "total": 64}), flush=True)
    mean = lambda values: sum(values) / len(values)
    result = {
        "status": "COMPLETE_DEVELOPMENT_ABLATION_ONLY",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": artifacts,
        "rows": rows,
        "summary": {
            "branches": len(rows),
            "lead_only_severe_branches": sum(row["lead_only_severe"] for row in rows),
            "adaptive_severe_branches": sum(row["adaptive_severe"] for row in rows),
            "local_only_severe_branches": sum(row["local_only_severe"] for row in rows),
            "mean_lead_only_byte_reduction_vs_periodic": mean([1 - row["lead_only_generated_bytes"] / row["periodic_generated_bytes"] for row in rows]),
            "mean_adaptive_byte_reduction_vs_periodic": mean([1 - row["adaptive_generated_bytes"] / row["periodic_generated_bytes"] for row in rows]),
            "lead_only_more_bytes_than_periodic_branches": sum(row["lead_only_generated_bytes"] > row["periodic_generated_bytes"] for row in rows),
            "lead_only_ttl_bound_count": sum(row["ttl_bound_count"] for row in rows),
            "lead_only_surface_bound_count": sum(row["surface_bound_count"] for row in rows),
        },
        "synthetic_visibility_only": True,
        "not_confirmatory": True,
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
