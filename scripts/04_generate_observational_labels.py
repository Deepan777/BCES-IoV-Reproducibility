#!/usr/bin/env python3
"""Generate frozen observational labels and independently recompute 100 pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.oracle.observational import (
    ObservationalConfig,
    generate_scene_labels,
    label_summary,
)
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

PLANNER_CONFIG = ROOT / "configs" / "planner" / "kinematic.yaml"
VALIDITY_CONFIG = ROOT / "configs" / "oracle" / "validity_v1.yaml"
REVIEW = ROOT / "data" / "manifests" / "v2xtraj_window_review_v1.json"
RAW_ROOT = ROOT / "data" / "raw" / "v2xtraj_primary"
OUTPUT = ROOT / "outputs" / "phase4" / "observational_v1"


def _config() -> tuple[PlannerConfig, ObservationalConfig, dict]:
    payload = yaml.safe_load(VALIDITY_CONFIG.read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(
        float(payload["max_cost_regret"]),
        float(payload["max_cached_risk"]),
        float(payload["max_trajectory_deviation_m"]),
    )
    scales = DriftScales(*map(float, payload["drift_scales"]))
    return (
        PlannerConfig.from_yaml(PLANNER_CONFIG),
        ObservationalConfig(
            thresholds=thresholds,
            cache_ages_s=tuple(map(float, payload["cache_ages_s"])),
            drift_scales=scales,
            maximum_objects_per_source=int(payload["maximum_objects_per_source"]),
        ),
        payload,
    )


def main() -> int:
    planner_config, observational_config, validity_payload = _config()
    planner = KinematicPlanner(planner_config)
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    expected = review.pop("review_sha256")
    if canonical_json_hash(review) != expected:
        raise ValueError("window review hash does not verify")
    valid_records = [row for row in review["records"] if row.get("valid_windows", 0) > 0]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels_path = OUTPUT / "labels.jsonl"
    manifest_path = OUTPUT / "run_manifest.json"
    if labels_path.exists() or manifest_path.exists():
        raise FileExistsError("observational_v1 is immutable; remove only after explicit review")
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data")
    budget_before = build_budget_report(budget)
    rows = []
    for index, record in enumerate(valid_records, start=1):
        rows.extend(
            generate_scene_labels(
                record,
                raw_root=RAW_ROOT,
                planner=planner,
                config=observational_config,
            )
        )
        if index % 25 == 0:
            print(f"labeled scenes {index}/{len(valid_records)}", flush=True)
    with labels_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    audit_ranked = sorted(rows, key=lambda row: canonical_json_hash({"audit": row["pair_id"]}))[:100]
    by_scenario = {str(row["scenario_id"]): row for row in valid_records}
    recomputed = {}
    for row in audit_ranked:
        scene_rows = generate_scene_labels(
            by_scenario[str(row["scenario_id"])],
            raw_root=RAW_ROOT,
            planner=planner,
            config=observational_config,
        )
        recomputed[row["pair_id"]] = next(item for item in scene_rows if item["pair_id"] == row["pair_id"])
    audit_mismatches = [
        row["pair_id"] for row in audit_ranked if canonical_json_hash(row) != canonical_json_hash(recomputed[row["pair_id"]])
    ]
    summary = label_summary(rows)
    minimum_cell = min(
        count for values in summary["decisions_by_split_behavior"].values() for count in values.values()
    )
    audit = {
        "schema_version": 1,
        "sample_count": len(audit_ranked),
        "selection": "lowest_sha256_of_pair_id_with_fixed_audit_namespace",
        "mismatches": audit_mismatches,
        "passed": not audit_mismatches and len(audit_ranked) == 100,
    }
    audit["audit_sha256"] = canonical_json_hash(audit)
    write_json_atomic(OUTPUT / "independent_audit.json", audit)
    budget_after = build_budget_report(budget)
    manifest = {
        "schema_version": 1,
        "phase": 4,
        "status": "PASS" if audit["passed"] and minimum_cell >= 50 else "FAIL",
        "started_from_review_sha256": expected,
        "planner_config_sha256": sha256_file(PLANNER_CONFIG),
        "validity_config_sha256": sha256_file(VALIDITY_CONFIG),
        "validity_config": validity_payload,
        "policy_hash": planner.policy_hash,
        "policy_config_sha256": planner_config.sha256,
        "summary": summary,
        "minimum_split_behavior_decisions": minimum_cell,
        "labels_sha256": sha256_file(labels_path),
        "independent_audit_sha256": audit["audit_sha256"],
        "thresholds_frozen_before_test": True,
        "label_wording": "empirically decision-valid for the frozen policy under stated thresholds",
        "git": git_state(ROOT),
        "budget_before": budget_before["measurements"],
        "budget_after": budget_after["measurements"],
        "completed_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
