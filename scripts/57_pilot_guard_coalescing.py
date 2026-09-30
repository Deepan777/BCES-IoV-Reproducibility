#!/usr/bin/env python3
"""Run a deterministic development-seed BCES request-coalescing pilot.

No confirmation/test seed is loaded. This is exploratory and cannot establish
benefit or safety. The registered implementation is not edited.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import partial
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
from bces.network.guard_coalescing import GuardCoalescingSession
from bces.simulation import controlled_closed_loop as loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.utils.hashing import sha256_file


def train_cases(family: str, count: int) -> list[int]:
    root = ROOT / "outputs/study_b/controlled_development_v2"
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    cases = []
    for filename, digest in sorted(manifest["artifact_sha256"].items()):
        if not filename.startswith("train_"):
            continue
        path = root / filename
        if sha256_file(path) != digest:
            raise RuntimeError("development input changed: " + filename)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            scene = json.load(handle)
        if scene["family"] == family:
            cases.append(int(scene["scenario_id"]))
    if len(cases) < count:
        raise RuntimeError("too few training cases")
    return cases[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("straight_lead_braking", "on_ramp_merge", "unprotected_crossing", "pedestrian_crossing"), default="straight_lead_braking")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--cooldown-ms", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cases <= 60 or args.cooldown_ms not in (200, 400, 600, 800, 1000):
        raise ValueError("unsupported development pilot design")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs/study_b") or output.exists():
        raise ValueError("output must be a new Study B directory")
    output.mkdir(parents=True)
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    for path, digest in registration["asset_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise RuntimeError("registered asset changed: " + path)
    for path, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise RuntimeError("registered source changed: " + path)
    models = {
        method: FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                 policy_hash=registration["policy_hash"])
        for method, rule in registration["model_rules"].items()
    }
    cases = train_cases(args.family, args.cases)
    conditions = {name: NetworkCondition(**registration["conditions"][name])
                  for name in registration["condition_order"]}
    import torch
    torch.set_num_threads(2)
    protocol = {
        "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATION",
        "family": args.family, "cases": cases,
        "case_selection": "first sorted hash-verified training scenarios in family",
        "cooldown_ms": args.cooldown_ms,
        "methods": ["periodic_payload", "scalar_ttl", "surface", "surface_coalesced"],
        "conditions": list(conditions),
        "registration_sha256": sha256_file(registration_path),
        "candidate_source_sha256": {
            "bces/network/guard_coalescing.py": sha256_file(ROOT / "bces/network/guard_coalescing.py"),
            "scripts/57_pilot_guard_coalescing.py": sha256_file(Path(__file__)),
        },
        "does_not_reuse_confirmation_seeds": True,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = []
    artifacts = {}
    original_session = loop.BoundSession
    try:
        for seed in cases:
            for condition_name, condition in conditions.items():
                branches = {}
                for method in protocol["methods"]:
                    actual_method = "surface" if method == "surface_coalesced" else method
                    loop.BoundSession = (partial(GuardCoalescingSession, guard_cooldown_ms=args.cooldown_ms)
                                         if method == "surface_coalesced" else original_session)
                    branch = loop.run_branch(
                        seed=seed, method=actual_method, config=registration["config"],
                        condition=condition, network_seed=seed + 910000,
                        model=models.get(actual_method), rule=registration["model_rules"].get(actual_method),
                        horizon_s=registration["horizon_s"],
                    )
                    branches[method] = branch
                    name = f"{seed}_{condition_name}_{method}.json.gz"
                    with gzip.open(output / name, "xt", encoding="utf-8") as handle:
                        json.dump(branch, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    artifacts[name] = sha256_file(output / name)
                    print(f"completed {seed} {condition_name} {method}", flush=True)
                baseline = branches["periodic_payload"]
                prefix = baseline["initial_contract"]
                for method, branch in branches.items():
                    traffic = branch["traffic"]
                    if not compare_snapshots(prefix, branch["initial_contract"], tolerance=1e-6)["equal_within_tolerance"]:
                        raise AssertionError("prefix mismatch")
                    if not traffic["byte_conservation_ok"] or not traffic["component_conservation_ok"]:
                        raise AssertionError("byte accounting failed")
                    if any(row["used_timestamp_ms"] > row["available_ms"] for row in branch["input_availability_audit"]):
                        raise AssertionError("future observation leakage")
                    rows.append({
                        "seed": seed, "family": args.family, "condition": condition_name, "method": method,
                        "bytes": traffic["generated_bytes"],
                        "byte_reduction_vs_periodic": 1 - traffic["generated_bytes"] / baseline["traffic"]["generated_bytes"],
                        "outcome": branch["outcome"], "duration_s": branch["duration_s"],
                        "comparable": branch["outcome"] == baseline["outcome"] and branch["duration_s"] == baseline["duration_s"],
                        "severe": _severe(branch),
                        "progress_deficit_m": baseline["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"],
                        "adverse_vs_periodic": _severe(branch) or baseline["route_distance_lower_bound_m"] - branch["route_distance_lower_bound_m"] > 5,
                        "reuse_decisions": branch["counts"].get("reuse_decisions", 0),
                        "guard_refresh_coalesced": traffic.get("session_counts", {}).get("guard_refresh_coalesced", 0),
                        "actuation_contract_passed": branch["actuation_contract_passed"],
                    })
    finally:
        loop.BoundSession = original_session
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    report = {
        "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATION",
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "artifact_sha256": artifacts,
        "rows": rows,
        "descriptive_summary": {
            method: {
                "branches": len(items),
                "mean_byte_reduction_vs_periodic": sum(x["byte_reduction_vs_periodic"] for x in items) / len(items),
                "adverse_branches": sum(x["adverse_vs_periodic"] for x in items),
                "noncomparable_branches": sum(not x["comparable"] for x in items),
                "actuation_failures": sum(not x["actuation_contract_passed"] for x in items),
                "guard_refresh_coalesced": sum(x["guard_refresh_coalesced"] for x in items),
            }
            for method, items in by_method.items()
        },
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["descriptive_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
