#!/usr/bin/env python3
"""Generate corrected, leakage-controlled same-world SUMO pairs."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.simulation.branching import recompute_pair_label, run_same_world_pair
from bces.simulation.scenario_builder import scenario_spec
from bces.simulation.sumo_adapter import sumo_version
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "simulation" / "sumo_v3.yaml"
PLANNER_CONFIG = ROOT / "configs" / "planner" / "kinematic.yaml"
OUTPUT = ROOT / "outputs" / "phase5" / "sumo_v3"


def _partition(index: int, counts: dict[str, int]) -> str:
    cursor = 0
    for name, count in counts.items():
        cursor += int(count)
        if index < cursor:
            return name
    raise ValueError("partition counts do not cover all pairs")


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    actual_version = sumo_version()
    if actual_version != str(config["sumo_version"]):
        raise RuntimeError(f"SUMO version {actual_version} does not match pinned {config['sumo_version']}")
    network = ROOT / config["network"]
    if not network.is_file():
        raise FileNotFoundError(network)
    planner = KinematicPlanner(PlannerConfig.from_yaml(PLANNER_CONFIG))
    scale_path = ROOT / config["drift_scale_manifest"]
    scale_payload = json.loads(scale_path.read_text(encoding="utf-8"))
    scale_hash = scale_payload.pop("scales_sha256")
    if canonical_json_hash(scale_payload) != scale_hash:
        raise ValueError("frozen drift-scale manifest does not verify")
    if scale_payload["partition"] != "train" or scale_payload["test_partition_accessed"]:
        raise ValueError("drift scales were not fit exclusively on training data")
    drift_scales = DriftScales(*map(float, scale_payload["scales"]))
    threshold_values = config["label_thresholds"]
    thresholds = ValidityThresholds(
        threshold_values["max_cost_regret"],
        threshold_values["max_cached_risk"],
        threshold_values["max_trajectory_deviation_m"],
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels_path = OUTPUT / "pairs.jsonl"
    manifest_path = OUTPUT / "run_manifest.json"
    if labels_path.exists() or manifest_path.exists():
        raise FileExistsError("sumo_v3 output is immutable")
    part = labels_path.with_suffix(".jsonl.part")
    part.unlink(missing_ok=True)
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data")
    budget_before = build_budget_report(budget)
    rows = []
    families = tuple(config["families"])
    partitions = {str(key): int(value) for key, value in config["partitions"].items()}
    if sum(partitions.values()) != int(config["pair_count"]):
        raise ValueError("partition counts must equal pair_count")
    with part.open("x", encoding="utf-8", newline="\n") as handle:
        for index in range(int(config["pair_count"])):
            seed = int(config["seed_start"]) + index
            family = families[index % len(families)]
            row = run_same_world_pair(
                network=network,
                state_path=ROOT / "sumo" / "states" / f"phase5_{seed}.xml.gz",
                seed=seed,
                step_s=float(config["step_s"]),
                reference_time_s=float(config["reference_time_s"]),
                horizon_s=float(config["branch_horizon_s"]),
                spec=scenario_spec(family),
                planner=planner,
                thresholds=thresholds,
                drift_scales=drift_scales,
                split=_partition(index, partitions),
            )
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            if (index + 1) % 10 == 0:
                handle.flush()
                print(f"completed SUMO pairs {index + 1}/{config['pair_count']}", flush=True)
    os.replace(part, labels_path)
    recomputed = [recompute_pair_label(row, thresholds) for row in rows]
    development_rows = [row for row in rows if row["split"] != "test"]
    training_rows = [row for row in rows if row["split"] == "train"]
    adequacy = config["development_adequacy"]
    development_invalid = sum(not row["valid"] for row in development_rows)
    training_valid = sum(row["valid"] for row in training_rows)
    training_invalid = sum(not row["valid"] for row in training_rows)
    audit = {
        "schema_version": 3,
        "pair_count": len(rows),
        "state_equivalent_count": sum(row["state_equivalent"] for row in rows),
        "label_recomputation_mismatches": [
            row["pair_id"] for row, value in zip(rows, recomputed) if bool(row["valid"]) != value
        ],
        "family_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "input_contract_count": sum(
            row["reference"]["input_contract"] == "exact_reference_payload_and_frozen_policy_query_v2"
            for row in rows
        ),
        "development_class_counts": {
            "valid": len(development_rows) - development_invalid,
            "invalid": development_invalid,
        },
        "training_class_counts": {"valid": training_valid, "invalid": training_invalid},
        "test_labels_used_for_adequacy": False,
        "passed": (
            len(rows) == 100
            and all(row["state_equivalent"] for row in rows)
            and all(bool(row["valid"]) == value for row, value in zip(rows, recomputed))
            and development_invalid >= int(adequacy["minimum_non_test_invalid_pairs"])
            and training_valid >= int(adequacy["minimum_train_valid_pairs"])
            and training_invalid >= int(adequacy["minimum_train_invalid_pairs"])
        ),
    }
    audit["audit_sha256"] = canonical_json_hash(audit)
    write_json_atomic(OUTPUT / "audit.json", audit)
    budget_after = build_budget_report(budget)
    manifest = {
        "schema_version": 3,
        "phase": 5,
        "status": "PASS" if audit["passed"] else "FAIL",
        "sumo_version": actual_version,
        "sumo_config_sha256": sha256_file(CONFIG),
        "network_sha256": sha256_file(network),
        "planner_config_sha256": sha256_file(PLANNER_CONFIG),
        "branching_source_sha256": sha256_file(ROOT / "bces" / "simulation" / "branching.py"),
        "scenario_builder_source_sha256": sha256_file(ROOT / "bces" / "simulation" / "scenario_builder.py"),
        "drift_scale_manifest_sha256": sha256_file(scale_path),
        "drift_scales_sha256": scale_hash,
        "policy_hash": planner.policy_hash,
        "pairs_sha256": sha256_file(labels_path),
        "audit_sha256": audit["audit_sha256"],
        "pair_count": len(rows),
        "split_counts": audit["split_counts"],
        "development_class_counts": audit["development_class_counts"],
        "training_class_counts": audit["training_class_counts"],
        "valid_count": sum(row["valid"] for row in rows),
        "invalid_count": sum(not row["valid"] for row in rows),
        "rendered_frames": 0,
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
