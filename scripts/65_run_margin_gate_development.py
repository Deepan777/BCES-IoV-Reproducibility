#!/usr/bin/env python3
"""Frozen-before-outcome, resumable 40-scenario adaptive-gate development run.

The scenes are existing validation-partition SUMO scenarios, not fresh
confirmation. All 5 methods share complete prefixes and network seeds.
"""

from __future__ import annotations

import argparse
import gc
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
from bces.network.adaptive_margin_policy import AdaptiveMarginWirePolicy
from bces.network.events import NetworkCondition
from bces.simulation import controlled_closed_loop as loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.utils.hashing import canonical_json_hash, sha256_file

SOURCE = ROOT / "outputs/study_b/controlled_development_v2"
OUTPUT = ROOT / "outputs/study_b/adaptive_margin_gate_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
FAMILIES = ("straight_lead_braking", "on_ramp_merge", "unprotected_crossing", "pedestrian_crossing")
METHODS = ("periodic_payload", "local_only", "surface", "scalar_ttl", "adaptive_margin_gate")
PER_FAMILY = 10


def registration_plan() -> dict:
    frozen = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in {**frozen["source_sha256"], **frozen["asset_sha256"]}.items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source/asset changed: " + name)
    candidates = {family: [] for family in FAMILIES}
    for path in sorted(SOURCE.glob("validation_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            scene = json.load(handle)
        if scene["split"] != "validation":
            raise RuntimeError("nonvalidation development source")
        if scene["family"] in candidates:
            candidates[scene["family"]].append({"seed": int(scene["scenario_id"]),
                                                  "family": scene["family"],
                                                  "development_source": str(path.relative_to(ROOT)).replace("\\", "/"),
                                                  "development_source_sha256": sha256_file(path)})
    selected = []
    for family in FAMILIES:
        cases = sorted(candidates[family], key=lambda row: row["seed"])
        if len(cases) < PER_FAMILY:
            raise RuntimeError("insufficient development-validation scenes")
        selected.extend(cases[:PER_FAMILY])
    sources = ["scripts/65_run_margin_gate_development.py",
               "bces/network/adaptive_margin_policy.py",
               "bces/simulation/controlled_closed_loop.py",
               "bces/network/bound_session.py",
               "bces/protocol/bound_exchange.py"]
    return {"status": "FROZEN_DEVELOPMENT_ONLY", "scope": "development validation, not independent confirmation",
            "v2_registration_sha256": sha256_file(REGISTRATION),
            "development_manifest_sha256": sha256_file(SOURCE / "run_manifest.json"),
            "source_sha256": {name: sha256_file(ROOT / name) for name in sources},
            "selected": selected, "methods": list(METHODS),
            "conditions": list(frozen["condition_order"]),
            "network_seed_rule": "seed + 910000",
            "model_rules": frozen["model_rules"],
            "horizon_s": frozen["horizon_s"],
            "old_confirmation_outcomes_accessed": False,
            "no_interim_outcome_selection": True}


def save_branch(path: Path, wrapper: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with gzip.open(temporary, "xt", encoding="utf-8") as handle:
        json.dump(wrapper, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary.rename(path)


def read_branch(path: Path, protocol_sha: str, seed: int, condition: str, method: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        wrapper = json.load(handle)
    if (wrapper["protocol_sha256"] != protocol_sha or wrapper["seed"] != seed
            or wrapper["condition"] != condition or wrapper["method"] != method
            or canonical_json_hash(wrapper["branch"]) != wrapper["branch_sha256"]):
        raise RuntimeError("preserved development branch identity/hash mismatch: " + path.name)
    return wrapper["branch"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    plan = registration_plan()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTPUT / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("development protocol/source changed")
    else:
        protocol_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    protocol_sha = sha256_file(protocol_path)
    if args.register_only:
        print(json.dumps({"status": plan["status"], "protocol_sha256": protocol_sha,
                          "scenarios": len(plan["selected"]), "branches": len(plan["selected"])*len(plan["conditions"])*len(METHODS)}))
        return
    frozen = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    import torch
    torch.set_num_threads(2)
    models = {method: FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                       policy_hash=frozen["policy_hash"])
              for method, rule in frozen["model_rules"].items()}
    artifact_sha256 = {}
    rows = []
    total = len(plan["selected"]) * len(plan["conditions"]) * len(METHODS)
    completed = 0
    for case in plan["selected"]:
        seed, family = case["seed"], case["family"]
        for condition_name in plan["conditions"]:
            condition = NetworkCondition(**frozen["conditions"][condition_name])
            branches = {}
            for method in METHODS:
                path = OUTPUT / f"branch_{seed}_{condition_name}_{method}.json.gz"
                if path.exists():
                    branch = read_branch(path, protocol_sha, seed, condition_name, method)
                else:
                    if method == "adaptive_margin_gate":
                        adaptive = AdaptiveMarginWirePolicy(
                            surface=models["surface"], scalar_ttl=models["scalar_ttl"],
                            shrinkages={name: rule["shrinkage"] for name, rule in frozen["model_rules"].items()})
                        with patch.object(loop, "bind_reference", adaptive.bind):
                            branch = loop.run_branch(seed=seed, method="surface", config=frozen["config"],
                                condition=condition, network_seed=seed + 910000, model=adaptive,
                                rule=frozen["model_rules"]["surface"], horizon_s=frozen["horizon_s"])
                        branch["method"] = method
                        branch["adaptive_selections"] = adaptive.selections
                    else:
                        kwargs = {"model": models[method], "rule": frozen["model_rules"][method]} if method in models else {}
                        branch = loop.run_branch(seed=seed, method=method, config=frozen["config"],
                            condition=condition, network_seed=seed + 910000, horizon_s=frozen["horizon_s"], **kwargs)
                    if (branch["family"] != family or branch["method"] != method
                            or not branch["actuation_contract_passed"]
                            or not branch["traffic"]["byte_conservation_ok"]
                            or not branch["traffic"]["component_conservation_ok"]
                            or any(item["used_timestamp_ms"] > item["available_ms"] for item in branch["input_availability_audit"])):
                        raise RuntimeError("development branch engineering/causal gate failed")
                    save_branch(path, {"protocol_sha256": protocol_sha, "seed": seed,
                                       "family": family, "condition": condition_name, "method": method,
                                       "branch_sha256": canonical_json_hash(branch), "branch": branch})
                branches[method] = branch
                artifact_sha256[path.name] = sha256_file(path)
                completed += 1
                print(json.dumps({"completed_branches": completed, "total_branches": total,
                                  "seed": seed, "condition": condition_name, "method": method}), flush=True)
                gc.collect()
            prefix = branches["periodic_payload"]["initial_contract"]
            if any(not compare_snapshots(prefix, branch["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]
                   for branch in branches.values()):
                raise RuntimeError("unmatched development prefix")
            periodic = branches["periodic_payload"]
            for method, branch in branches.items():
                deficit = periodic["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"]
                rows.append({"seed": seed, "family": family, "condition": condition_name,
                             "method": method, "generated_bytes": branch["traffic"]["generated_bytes"],
                             "byte_reduction_vs_periodic": 1 - branch["traffic"]["generated_bytes"] / periodic["traffic"]["generated_bytes"],
                             "severe": _severe(branch), "progress_deficit_m": deficit,
                             "adverse_vs_periodic": _severe(branch) or deficit > 5.0,
                             "adaptive_selected_families": [item["family"] for item in branch.get("adaptive_selections", [])]})
    report = {"status": "COMPLETE_DEVELOPMENT_ONLY", "protocol_sha256": protocol_sha,
              "artifact_sha256": artifact_sha256, "rows": rows,
              "not_confirmatory": True, "manuscript_allowed_as_confirmatory": False}
    report_path = OUTPUT / "report.json"
    if report_path.exists():
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise RuntimeError("existing development report differs")
    else:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "branches": len(rows),
                      "artifact_count": len(artifact_sha256)}))


if __name__ == "__main__":
    main()
