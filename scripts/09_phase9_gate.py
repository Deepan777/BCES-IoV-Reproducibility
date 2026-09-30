#!/usr/bin/env python3
"""Verify the locked 30-seed Phase-9 SUMO artifact."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

RUN = ROOT / "outputs/phase9/confirmatory_v1"
CONFIG = ROOT / "configs/simulation/phase9_confirmatory_v1.yaml"
OUTPUT = ROOT / "outputs/phase9/phase9_gate.json"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("Phase-9 gate output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("Phase-9 gate requires a clean committed revision")
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    raw_path = RUN / "routes.jsonl.gz"
    if sha256_file(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Phase-9 raw hash mismatch")
    with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    keys = Counter((row["seed"], row["condition"], row["method"]) for row in rows)
    states: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if row["status"] == "PASS":
            states[int(row["seed"])].add(row["pre_action_non_ego_state_hash"])
    budget = build_budget_report(Budget(workspace_root=ROOT, data_root=ROOT / "data"))
    checks = {
        "at_least_30_matched_seeds": manifest["matched_seed_count"] >= 30 and len(states) == manifest["matched_seed_count"],
        "exact_registered_branch_count": len(rows) == manifest["registered_branch_count"] == 300 and len(keys) == 300 and all(value == 1 for value in keys.values()),
        "no_seed_exclusions_or_failures": all(row["status"] == "PASS" for row in rows) and not manifest["failure_taxonomy"],
        "matched_pre_action_state": all(len(value) == 1 for value in states.values()) and all(row["matched_pre_action_state"] for row in rows),
        "byte_conservation": all(row["communication"]["byte_conservation_failures"] == 0 for row in rows),
        "no_rendered_media": manifest["rendered_frames"] == 0,
        "resource_caps": manifest["peak_cuda_allocated_bytes"] <= 5_500_000_000 and budget["measurements"]["output_reserve_preserved"] and budget["measurements"]["workspace_bytes"] < 10_000_000_000,
    }
    report = {
        "schema_version": 1, "phase": 9, "status": "PASS" if all(checks.values()) else "FAIL",
        "engineering_complete": all(checks.values()), "scientific_success": False,
        "checks": checks, "matched_seeds": manifest["matched_seed_count"],
        "failure_taxonomy": manifest["failure_taxonomy"],
        "scientific_limitations": ["zero collision and critical-event discordance across all methods", "Surface generated more complete-accounting bytes than scalar and always-fresh", "Surface reuse was sparse"],
        "manifest_sha256": sha256_file(RUN / "run_manifest.json"), "raw_sha256": manifest["raw_sha256"],
        "config_sha256": sha256_file(CONFIG), "budget": budget["measurements"],
        "git": state, "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(OUTPUT, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

