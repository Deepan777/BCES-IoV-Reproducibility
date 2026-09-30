#!/usr/bin/env python3
"""Full-grid development stress of the unchanged adaptive margin gate.

All 32 pre-existing hidden-lead cells and both conditions are included. The
synthetic hidden-actor flag is not a validated physical occlusion model.
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
from bces.network.adaptive_margin_policy import AdaptiveMarginWirePolicy
from bces.network.events import NetworkCondition
from bces.simulation import controlled_closed_loop as loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file

BASE = ROOT / "outputs/study_b/hidden_lead_communication_value_development_v1"
FROZEN = ROOT / "outputs/study_b/hidden_lead_bces_ttl_development_v1"
OUTPUT = ROOT / "outputs/study_b/adaptive_margin_hidden_grid_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    base_report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    frozen_report = json.loads((FROZEN / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if len(base_protocol["grid"]) != 32 or len(base_report["rows"]) != 64 or len(frozen_report["rows"]) != 128:
        raise RuntimeError("incomplete pre-existing development grid")
    for folder, report in ((BASE, base_report), (FROZEN, frozen_report)):
        for name, digest in report["artifact_sha256"].items():
            if sha256_file(folder / name) != digest:
                raise RuntimeError("pre-existing grid artifact changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    protocol = {"status": "FROZEN_DEVELOPMENT_STRESS_ONLY",
                "base_protocol_sha256": sha256_file(BASE / "protocol.json"),
                "base_report_sha256": sha256_file(BASE / "report.json"),
                "frozen_policy_grid_report_sha256": sha256_file(FROZEN / "report.json"),
                "v2_registration_sha256": sha256_file(REGISTRATION),
                "adaptive_source_sha256": sha256_file(ROOT / "bces/network/adaptive_margin_policy.py"),
                "script_sha256": sha256_file(Path(__file__)),
                "grid_indices": list(range(32)),
                "conditions": list(registration["condition_order"]),
                "new_method": "adaptive_margin_gate", "not_confirmatory": True}
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
            with patch.object(loop, "development_generator", lambda: generator), patch.object(loop, "bind_reference", policy.bind):
                branch = loop.run_branch(seed=seed, method="surface", config=registration["config"],
                    condition=condition, network_seed=seed + 910000,
                    model=policy, rule=registration["model_rules"]["surface"],
                    horizon_s=registration["horizon_s"])
            branch["method"] = "adaptive_margin_gate"
            branch["adaptive_selections"] = policy.selections
            with gzip.open(BASE / f"{seed}_{condition_name}_periodic_payload.json.gz", "rt", encoding="utf-8") as handle:
                periodic = json.load(handle)
            with gzip.open(BASE / f"{seed}_{condition_name}_local_only.json.gz", "rt", encoding="utf-8") as handle:
                local_only = json.load(handle)
            if (not compare_snapshots(periodic["initial_contract"], branch["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
                    or not branch["actuation_contract_passed"]
                    or not branch["traffic"]["byte_conservation_ok"]
                    or not branch["traffic"]["component_conservation_ok"]
                    or any(item["used_timestamp_ms"] > item["available_ms"] for item in branch["input_availability_audit"])):
                raise RuntimeError("adaptive hidden-grid engineering/causal gate failed")
            name = f"{seed}_{condition_name}_adaptive_margin_gate.json.gz"
            with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            artifacts[name] = sha256_file(OUTPUT / name)
            rows.append({"seed": seed, "grid_index": index, "condition": condition_name,
                         "parameters": cell, "adaptive_selections": [item["family"] for item in policy.selections],
                         "adaptive_severe": _severe(branch), "periodic_severe": _severe(periodic),
                         "local_only_severe": _severe(local_only),
                         "adaptive_generated_bytes": branch["traffic"]["generated_bytes"],
                         "periodic_generated_bytes": periodic["traffic"]["generated_bytes"],
                         "adaptive_byte_reduction_vs_periodic": 1 - branch["traffic"]["generated_bytes"] / periodic["traffic"]["generated_bytes"],
                         "adaptive_progress_deficit_m": periodic["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"],
                         "not_confirmatory": True})
            print(json.dumps({"completed": len(rows), "total": 64, "grid_index": index,
                              "condition": condition_name}), flush=True)
    result = {"status": "COMPLETE_DEVELOPMENT_STRESS_ONLY",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "summary": {"branches": len(rows),
                          "adaptive_severe_branches": sum(row["adaptive_severe"] for row in rows),
                          "local_only_severe_branches": sum(row["local_only_severe"] for row in rows),
                          "mean_adaptive_byte_reduction_vs_periodic": sum(row["adaptive_byte_reduction_vs_periodic"] for row in rows)/len(rows),
                          "adaptive_both_conditions_avoid_local_only_severe": all(not row["adaptive_severe"] for row in rows if row["local_only_severe"]),
                          "ttl_bound_count": sum(row["adaptive_selections"].count("scalar_ttl") for row in rows),
                          "surface_bound_count": sum(row["adaptive_selections"].count("surface") for row in rows)},
              "synthetic_visibility_only": True, "not_confirmatory": True}
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
