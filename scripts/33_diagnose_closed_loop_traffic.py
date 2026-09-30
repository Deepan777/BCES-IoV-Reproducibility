#!/usr/bin/env python3
"""Read-only post-completion traffic diagnosis; never a confirmatory analysis."""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import write_json_atomic

REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
CONFIRMATION = REGISTRATION.parent / "confirmation"
OUTPUT = ROOT / "outputs/study_b/closed_loop_v2_traffic_diagnosis/diagnosis.json"
METHODS = ("surface", "scalar_ttl", "periodic_payload")
CONDITIONS = ("nominal", "impaired")


def checked_branch(seed, family, condition, method, manifest, registration_hash):
    name = f"branch_{seed}_{condition}_{method}.json.gz"
    path = CONFIRMATION / name
    if sha256_file(path) != manifest["artifact_sha256"][name]:
        raise RuntimeError(f"artifact hash changed: {name}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        wrapper = json.load(handle)
    if (wrapper["registration_sha256"] != registration_hash
            or wrapper["seed"] != seed or wrapper["family"] != family
            or wrapper["condition"] != condition or wrapper["method"] != method
            or canonical_json_hash(wrapper["branch"]) != wrapper["branch_sha256"]):
        raise RuntimeError(f"branch identity changed: {name}")
    return wrapper["branch"]


def main():
    if OUTPUT.exists():
        raise FileExistsError("post-completion diagnosis is immutable")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    manifest_path = CONFIRMATION / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registration_hash = sha256_file(REGISTRATION)
    if manifest["status"] != "COMPLETE" or manifest["registration_sha256"] != registration_hash:
        raise RuntimeError("registered run is incomplete or mismatched")
    totals = {}
    for assignment in registration["scenario_sampling"]["selected"]:
        seed, family = assignment["seed"], assignment["family"]
        for condition in CONDITIONS:
            for method in METHODS:
                branch = checked_branch(seed, family, condition, method, manifest, registration_hash)
                key = (family, method)
                item = totals.setdefault(key, {"branches": 0, "generated_bytes": 0,
                    "reuse_decisions": 0, "local_only_decisions": 0,
                    "session_counts": Counter(), "component_bytes": Counter(),
                    "timeline_events": Counter()})
                item["branches"] += 1
                item["generated_bytes"] += branch["traffic"]["generated_bytes"]
                item["reuse_decisions"] += branch["counts"].get("reuse_decisions", 0)
                item["local_only_decisions"] += branch["counts"].get("local_only_decisions", 0)
                item["session_counts"].update(branch["traffic"].get("session_counts", {}))
                item["component_bytes"].update(branch["traffic"]["component_bytes"])
                item["timeline_events"].update(
                    row["event"] for row in branch["traffic"].get("timeline", []))
    families = {}
    for (family, method), item in sorted(totals.items()):
        decisions = item["reuse_decisions"] + item["local_only_decisions"]
        families.setdefault(family, {})[method] = {
            "branches": item["branches"],
            "generated_bytes": item["generated_bytes"],
            "mean_generated_bytes_per_branch": item["generated_bytes"] / item["branches"],
            "reuse_fraction": item["reuse_decisions"] / decisions if decisions else None,
            "session_counts": dict(sorted(item["session_counts"].items())),
            "component_bytes": dict(sorted(item["component_bytes"].items())),
            "timeline_events": dict(sorted(item["timeline_events"].items())),
        }
    report = {"status": "POSTHOC_DIAGNOSTIC_ONLY", "registration_sha256": registration_hash,
        "manifest_sha256": sha256_file(manifest_path), "source_sha256": sha256_file(Path(__file__)),
        "families": families, "design_rule": "Do not fit or select a revised method on v2 outcomes; use separate development seeds and new registration."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT, report)
    straight = families["straight_lead_braking"]
    print(json.dumps({method: straight[method] for method in METHODS}, indent=2))


if __name__ == "__main__":
    main()
