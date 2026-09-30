#!/usr/bin/env python3
"""Lead-only packet policy on the already-open 40-scene development mixture.

The protocol is saved before new branch outcomes. This is a post-selection
development ablation, not an independent validation or confirmation.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
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
from bces.utils.hashing import canonical_json_hash, sha256_file

BASE = ROOT / "outputs/study_b/adaptive_margin_gate_development_v1"
GRID_ABLATION = ROOT / "outputs/study_b/lead_only_hidden_grid_ablation_development_v1"
OUTPUT = ROOT / "outputs/study_b/lead_only_40scene_ablation_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
ORIGINAL_SELECTOR = margin_module.select_reference_method


def lead_only(reference: dict) -> tuple[str, dict]:
    _, evidence = ORIGINAL_SELECTOR(reference)
    return ("scalar_ttl" if evidence["lead_like"] else "surface"), evidence


def read_base(seed: int, condition: str, method: str, protocol_sha: str) -> dict:
    path = BASE / f"branch_{seed}_{condition}_{method}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        wrapped = json.load(handle)
    if (wrapped["protocol_sha256"] != protocol_sha or wrapped["seed"] != seed
            or wrapped["condition"] != condition or wrapped["method"] != method
            or canonical_json_hash(wrapped["branch"]) != wrapped["branch_sha256"]):
        raise RuntimeError("base branch identity/hash mismatch: " + path.name)
    return wrapped["branch"]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_protocol = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    base_report = json.loads((BASE / "report.json").read_text(encoding="utf-8"))
    grid_report = json.loads((GRID_ABLATION / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if len(base_protocol["selected"]) != 40 or len(base_report["rows"]) != 400 or len(grid_report["rows"]) != 64:
        raise RuntimeError("incomplete development reference")
    for name, digest in base_report["artifact_sha256"].items():
        if sha256_file(BASE / name) != digest:
            raise RuntimeError("base artifact changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    for case in base_protocol["selected"]:
        if sha256_file(ROOT / case["development_source"]) != case["development_source_sha256"]:
            raise RuntimeError("development source changed: " + case["development_source"])
    protocol = {
        "status": "FROZEN_POST_SELECTION_DEVELOPMENT_ABLATION_ONLY",
        "question": "Does lead geometry alone outperform the full margin gate on the already-open 40-scene mixture?",
        "base_protocol_sha256": sha256_file(BASE / "protocol.json"),
        "base_report_sha256": sha256_file(BASE / "report.json"),
        "grid_ablation_report_sha256": sha256_file(GRID_ABLATION / "report.json"),
        "v2_registration_sha256": sha256_file(REGISTRATION),
        "adaptive_source_sha256": sha256_file(ROOT / "bces/network/adaptive_margin_policy.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "selected": base_protocol["selected"],
        "conditions": base_protocol["conditions"],
        "comparators": ["adaptive_margin_gate", "scalar_ttl", "surface", "periodic_payload", "local_only"],
        "unit": "scenario with two nested channel conditions",
        "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    protocol_sha = sha256_file(OUTPUT / "protocol.json")
    models = {method: FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                       policy_hash=registration["policy_hash"])
              for method, rule in registration["model_rules"].items()}
    import torch
    torch.set_num_threads(2)
    rows, artifacts = [], {}
    for case in base_protocol["selected"]:
        seed, family = case["seed"], case["family"]
        for condition_name in base_protocol["conditions"]:
            condition = NetworkCondition(**registration["conditions"][condition_name])
            policy = AdaptiveMarginWirePolicy(
                surface=models["surface"], scalar_ttl=models["scalar_ttl"],
                shrinkages={name: rule["shrinkage"] for name, rule in registration["model_rules"].items()})
            policy.sha256 = canonical_json_hash({
                "kind": "development_lead_only_ablation_v1",
                "parent_policy_sha256": policy.sha256,
                "script_sha256": protocol["script_sha256"],
            })
            with (patch.object(loop, "bind_reference", policy.bind),
                  patch.object(margin_module, "select_reference_method", lead_only)):
                branch = loop.run_branch(seed=seed, method="surface", config=registration["config"],
                    condition=condition, network_seed=seed + 910000,
                    model=policy, rule=registration["model_rules"]["surface"],
                    horizon_s=registration["horizon_s"])
            branch["method"] = "lead_only_ablation"
            branch["ablation_selections"] = policy.selections
            periodic = read_base(seed, condition_name, "periodic_payload", protocol["base_protocol_sha256"])
            adaptive = read_base(seed, condition_name, "adaptive_margin_gate", protocol["base_protocol_sha256"])
            ttl = read_base(seed, condition_name, "scalar_ttl", protocol["base_protocol_sha256"])
            surface = read_base(seed, condition_name, "surface", protocol["base_protocol_sha256"])
            local = read_base(seed, condition_name, "local_only", protocol["base_protocol_sha256"])
            if (branch["family"] != family or not branch["actuation_contract_passed"]
                    or not branch["traffic"]["byte_conservation_ok"]
                    or not branch["traffic"]["component_conservation_ok"]
                    or any(item["used_timestamp_ms"] > item["available_ms"] for item in branch["input_availability_audit"])
                    or any(not compare_snapshots(periodic["initial_contract"], other["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
                           for other in (branch, adaptive, ttl, surface, local))):
                raise RuntimeError("lead-only branch engineering/comparability gate failed")
            name = f"branch_{seed}_{condition_name}_lead_only_ablation.json.gz"
            with gzip.open(OUTPUT / name, "xt", encoding="utf-8") as handle:
                json.dump({"protocol_sha256": protocol_sha, "seed": seed, "family": family,
                           "condition": condition_name, "method": "lead_only_ablation",
                           "branch_sha256": canonical_json_hash(branch), "branch": branch},
                          handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            artifacts[name] = sha256_file(OUTPUT / name)
            row = {"seed": seed, "family": family, "condition": condition_name,
                   "periodic_generated_bytes": periodic["traffic"]["generated_bytes"],
                   "lead_only_generated_bytes": branch["traffic"]["generated_bytes"],
                   "adaptive_generated_bytes": adaptive["traffic"]["generated_bytes"],
                   "ttl_generated_bytes": ttl["traffic"]["generated_bytes"],
                   "surface_generated_bytes": surface["traffic"]["generated_bytes"],
                   "lead_only_severe": _severe(branch), "adaptive_severe": _severe(adaptive),
                   "ttl_severe": _severe(ttl), "surface_severe": _severe(surface),
                   "local_only_severe": _severe(local),
                   "lead_only_progress_deficit_m": periodic["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"],
                   "ttl_bound_count": sum(item["family"] == "scalar_ttl" for item in policy.selections),
                   "not_confirmatory": True}
            rows.append(row)
            print(json.dumps({"completed": len(rows), "total": 80}), flush=True)
    result = {"status": "COMPLETE_POST_SELECTION_DEVELOPMENT_ABLATION_ONLY",
              "protocol_sha256": protocol_sha, "artifact_sha256": artifacts, "rows": rows,
              "not_confirmatory": True}
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"branches": len(rows), "artifacts": len(artifacts)}))


if __name__ == "__main__":
    main()
