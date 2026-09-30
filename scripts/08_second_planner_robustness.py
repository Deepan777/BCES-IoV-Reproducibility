#!/usr/bin/env python3
"""Run a frozen, matched, validation-only second-planner robustness study."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.models.calibration import clustered_upper_bound, quantize_numpy
from bces.models.dataset import SurfaceDataset, _records_for_split
from bces.models.surfacenet import ScalarTTLLite, SurfaceNetLite, normal_tensor
from bces.models.validity_mlp import ValidityMLPLite
from bces.oracle.observational_v2 import BEHAVIORS, build_reference, load_scene, observed_validity_point
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/second_planner_v1.yaml"


def _verified_json(path: Path, hash_field: str) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop(hash_field)
    if canonical_json_hash(payload) != expected:
        raise ValueError(f"hash does not verify: {path}")
    return payload, str(expected)


def _classification(
    accepted: np.ndarray,
    valid: np.ndarray,
    behaviors: np.ndarray,
    clusters: tuple[str, ...],
    config: dict,
    *,
    seed_offset: int,
) -> dict:
    result: dict[str, dict] = {}
    groups = [("overall", np.arange(len(valid)))] + [
        (str(behavior), np.flatnonzero(behaviors == behavior)) for behavior in range(5)
    ]
    for group_index, (name, indices) in enumerate(groups):
        group_accepted = accepted[indices]
        group_valid = valid[indices]
        unsafe = group_accepted & ~group_valid
        accepted_count = int(group_accepted.sum())
        valid_count = int(group_valid.sum())
        group_clusters = tuple(clusters[index] for index in indices)
        upper = clustered_upper_bound(
            group_accepted,
            unsafe,
            group_clusters,
            confidence=float(config["confidence"]),
            replicates=int(config["bootstrap_replicates"]),
            seed=int(config["bootstrap_seed"]) + seed_offset + group_index,
        )
        result[name] = {
            "total": len(indices),
            "accepted": accepted_count,
            "unsafe_accepted": int(unsafe.sum()),
            "unsafe_accept_rate": int(unsafe.sum()) / accepted_count if accepted_count else None,
            "unsafe_accept_upper": upper,
            "coverage": accepted_count / len(indices) if len(indices) else 0.0,
            "valid_coverage": int((group_accepted & group_valid).sum()) / valid_count if valid_count else 0.0,
            "target_met": accepted_count >= int(config["minimum_accepted"]) and upper <= float(config["unsafe_accept_upper_target"]),
        }
    return result


def _agreement(first: np.ndarray, second: np.ndarray) -> dict:
    first = first.astype(bool)
    second = second.astype(bool)
    both_valid = int((first & second).sum())
    a_only = int((first & ~second).sum())
    b_only = int((~first & second).sum())
    both_invalid = int((~first & ~second).sum())
    observed = float((first == second).mean())
    first_rate, second_rate = float(first.mean()), float(second.mean())
    expected = first_rate * second_rate + (1.0 - first_rate) * (1.0 - second_rate)
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return {
        "both_valid": both_valid,
        "policy_a_only_valid": a_only,
        "policy_b_only_valid": b_only,
        "both_invalid": both_invalid,
        "agreement": observed,
        "cohen_kappa": kappa,
        "policy_a_valid_fraction": first_rate,
        "policy_b_valid_fraction": second_rate,
    }


def _batch_predict(dataset: SurfaceDataset, config: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surface = SurfaceNetLite().to(device)
    scalar = ScalarTTLLite().to(device)
    mlp = ValidityMLPLite().to(device)
    model_specs = (
        (surface, ROOT / config["surface_root"] / f"seed_{config['surface_seed']}" / "best.pt"),
        (scalar, ROOT / config["scalar_root"] / f"seed_{config['scalar_seed']}" / "best.pt"),
        (mlp, ROOT / config["validity_mlp_root"] / f"seed_{config['validity_mlp_seed']}" / "best.pt"),
    )
    for model, path in model_specs:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state_dict"])
        model.eval()
    offsets, ttls, probabilities, drifts = [], [], [], []
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0):
            batch = {key: value.to(device) for key, value in raw.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16):
                predicted, _ = surface(batch)
                ttl, _ = scalar(batch)
                logits = mlp(batch)
            offsets.append(predicted.float().cpu().numpy().astype(np.float64))
            ttls.append(ttl.float().cpu().numpy().astype(np.float64))
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy().astype(np.float64))
            drifts.append(batch["normalized_drift"].float().cpu().numpy().astype(np.float64))
    return tuple(map(np.concatenate, (offsets, ttls, probabilities, drifts)))  # type: ignore[return-value]


def _selections(config: dict) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[str, Path]]:
    surface_path = ROOT / config["surface_calibration"]
    surface_payload = json.loads(surface_path.read_text(encoding="utf-8"))
    surface_seed = next(row for row in surface_payload["seed_results"] if int(row["seed"]) == int(config["surface_seed"]))
    surface = {int(key): float(value["shrinkage"]) for key, value in surface_seed["selections"].items()}
    scalar_path = ROOT / config["scalar_root"] / "run_manifest.json"
    scalar_payload = json.loads(scalar_path.read_text(encoding="utf-8"))
    scalar_seed = next(row for row in scalar_payload["seeds"] if int(row["seed"]) == int(config["scalar_seed"]))
    scalar = {int(key): float(value["selection"]["shrinkage"]) for key, value in scalar_seed["calibration"].items()}
    mlp_path = ROOT / config["validity_mlp_root"] / "run_manifest.json"
    mlp_payload = json.loads(mlp_path.read_text(encoding="utf-8"))
    mlp_seed = next(row for row in mlp_payload["seeds"] if int(row["seed"]) == int(config["validity_mlp_seed"]))
    mlp = {int(key): float(value["selection"]["threshold"]) for key, value in mlp_seed["calibration"].items()}
    return surface, scalar, mlp, {"surface_calibration": surface_path, "scalar_manifest": scalar_path, "mlp_manifest": mlp_path}


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["partition"] != "validation" or config["test_partition_access"] != "prohibited":
        raise RuntimeError("second-planner study is validation-only")
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("second_planner_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("second-planner study requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("second-planner transfer evaluation requires CUDA")

    phase4_root = ROOT / config["phase4_root"]
    phase4_manifest = json.loads((phase4_root / "run_manifest.json").read_text(encoding="utf-8"))
    for name, expected in phase4_manifest["artifact_sha256"].items():
        if sha256_file(phase4_root / f"{name}.jsonl.gz") != expected:
            raise ValueError(f"Phase-4 artifact hash mismatch: {name}")
    review, review_sha = _verified_json(ROOT / config["review_manifest"], "review_sha256")
    scales_payload, scales_sha = _verified_json(ROOT / config["drift_scales"], "scales_sha256")
    scales = DriftScales(*map(float, scales_payload["scales"]))
    validity_payload = yaml.safe_load((ROOT / config["validity_config"]).read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(float(validity_payload["max_cost_regret"]), float(validity_payload["max_cached_risk"]), float(validity_payload["max_trajectory_deviation_m"]))
    policy_a = KinematicPlanner(PlannerConfig.from_yaml(ROOT / config["policy_a_config"]))
    policy_b = KinematicPlanner(PlannerConfig.from_yaml(ROOT / config["policy_b_config"]))
    if policy_a.policy_hash == policy_b.policy_hash:
        raise RuntimeError("second planner must have a distinct policy hash")
    if int(phase4_manifest["policy_hash"]) != policy_a.policy_hash:
        raise RuntimeError("Phase-4 artifact is not bound to Policy A")

    candidates = [row for row in review["records"] if row.get("valid_windows", 0) > 0 and row["split"] == "validation"]
    candidates.sort(key=lambda row: canonical_json_hash({"salt": config["selection_salt"], "scenario_id": row["scenario_id"]}))
    records = candidates[: int(config["maximum_scenarios"])]
    selected_ids = {str(row["scenario_id"]) for row in records}
    a_points = {
        row["point_id"]: row
        for row in _records_for_split(phase4_root / "validity_points.jsonl.gz", "validation")
        if str(row["scenario_id"]) in selected_ids
    }
    references, points = [], []
    counts: Counter[str] = Counter()
    raw_root = ROOT / config["raw_root"]
    for scene_index, record in enumerate(records, 1):
        ego, vehicle, infrastructure, focal_id, states = load_scene(record, raw_root)
        reference_time = min(states)
        for behavior in BEHAVIORS:
            try:
                reference, _, query, cached, template = build_reference(
                    scenario_id=str(record["scenario_id"]), intersection_id=str(record["intersection_id"]),
                    split="validation", behavior=behavior, focal_id=focal_id, reference_timestamp=reference_time,
                    states=states, ego_snapshots=ego, vehicle_snapshots=vehicle,
                    infrastructure_snapshots=infrastructure, scales=scales, planner=policy_b,
                    maximum_objects=int(validity_payload["maximum_objects_per_source"]),
                )
            except RuntimeError:
                counts[f"policy_b_reference_infeasible:{behavior}"] += 1
                continue
            reference_points = []
            for age in map(float, config["validity_ages_s"]):
                point = observed_validity_point(
                    reference=reference, query=query, cached_reference=cached, template=template,
                    current_timestamp=reference_time + round(age * 1000), states=states, focal_id=focal_id,
                    ego_snapshots=ego, vehicle_snapshots=vehicle, infrastructure_snapshots=infrastructure,
                    planner=policy_b, thresholds=thresholds,
                    maximum_objects=int(validity_payload["maximum_objects_per_source"]),
                )
                if point is None or point["point_id"] not in a_points:
                    counts[f"unmatched:{behavior}"] += 1
                    continue
                point["policy_a_valid"] = bool(a_points[point["point_id"]]["valid"])
                point["policy_b_valid"] = bool(point["valid"])
                point["label_changed"] = point["policy_a_valid"] != point["policy_b_valid"]
                reference_points.append(point)
                counts[f"matched:{behavior}"] += 1
            if reference_points:
                references.append(reference)
                points.extend(reference_points)
        print(f"second planner scenes {scene_index}/{len(records)}", flush=True)

    per_behavior = {behavior: counts[f"matched:{behavior}"] for behavior in BEHAVIORS}
    evidence_ok = len(points) >= int(config["minimum_matched_points"]) and min(per_behavior.values()) >= int(config["minimum_matched_points_per_behavior"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".second_planner_v1_", dir=output.parent))
    try:
        for name, rows in (("references", references), ("validity_points", points), ("boundaries", [])):
            with gzip.open(temp_root / f"{name}.jsonl.gz", "wt", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        dataset = SurfaceDataset(temp_root, "validation")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        offsets, ttls, probabilities, drift = _batch_predict(dataset, config, device)
        surface_shrinkage, scalar_shrinkage, mlp_threshold, selection_paths = _selections(config)
        behaviors = dataset.reference_tensors["identifiers"][dataset.reference_indices, 0].numpy().astype(int)
        valid_b = dataset.valid.numpy().astype(bool)
        clusters = dataset.scenario_ids
        surface_adjustment = np.asarray([surface_shrinkage[value] for value in behaviors])
        surface_offsets = quantize_numpy(np.maximum(0.0, offsets - surface_adjustment[:, None]))
        surface_accepted = (surface_offsets - drift @ normal_tensor().numpy().astype(np.float64).T).min(axis=1) >= float(config["refresh_guard_band"])
        scalar_adjustment = np.asarray([scalar_shrinkage[value] for value in behaviors])
        scalar_accepted = np.maximum(0.0, ttls - scalar_adjustment) - drift[:, 6] >= float(config["refresh_guard_band"])
        thresholds_by_row = np.asarray([mlp_threshold[value] for value in behaviors])
        mlp_accepted = probabilities >= thresholds_by_row
        valid_a = np.asarray([bool(row["policy_a_valid"]) for row in points])
        agreement = {"overall": _agreement(valid_a, valid_b)}
        agreement["by_behavior"] = {str(value): _agreement(valid_a[behaviors == value], valid_b[behaviors == value]) for value in range(5)}
        methods = {
            "surface": _classification(surface_accepted, valid_b, behaviors, clusters, config, seed_offset=1000),
            "learned_scalar_ttl": _classification(scalar_accepted, valid_b, behaviors, clusters, config, seed_offset=2000),
            "validity_mlp": _classification(mlp_accepted, valid_b, behaviors, clusters, config, seed_offset=3000),
        }
        report = {
            "schema_version": 1, "phase": 8, "component": "second_planner_robustness",
            "status": "PASS" if evidence_ok else "FAIL",
            "scientific_target_met": any(all(row[str(key)]["target_met"] for key in range(5)) for row in methods.values()),
            "partition": "validation", "selection": config["scenario_selection"],
            "selected_scenarios": [str(row["scenario_id"]) for row in records],
            "counts": {"selected_scenarios": len(records), "references": len(references), "matched_points": len(points), "matched_points_by_behavior": per_behavior, **dict(counts)},
            "evidence_thresholds_met": evidence_ok,
            "policy_a": {"name": policy_a.config.policy_name, "policy_hash": policy_a.policy_hash},
            "policy_b": {"name": policy_b.config.policy_name, "policy_hash": policy_b.policy_hash},
            "policy_hash_mismatch": policy_a.policy_hash != policy_b.policy_hash,
            "transfer_interpretation": config["transfer_interpretation"],
            "oracle_label_agreement": agreement,
            "forced_transfer_metrics_on_policy_b_labels": methods,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "test_partition_accessed": False,
            "source_sha256": {
                "config": sha256_file(CONFIG), "review": review_sha, "drift_scales": scales_sha,
                "phase4_manifest": phase4_manifest["manifest_sha256"],
                "policy_a_config": sha256_file(ROOT / config["policy_a_config"]),
                "policy_b_config": sha256_file(ROOT / config["policy_b_config"]),
                **{key: sha256_file(path) for key, path in selection_paths.items()},
            },
            "artifact_sha256": {name: sha256_file(temp_root / f"{name}.jsonl.gz") for name in ("references", "validity_points", "boundaries")},
            "git": state, "completed_utc": utc_now(),
        }
        report["report_sha256"] = canonical_json_hash(report)
        write_json_atomic(temp_root / "second_planner.json", report)
        os.replace(temp_root, output)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "scientific_target_met": report["scientific_target_met"], "matched_points": len(points), "agreement": report["oracle_label_agreement"]["overall"], "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

