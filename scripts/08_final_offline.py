#!/usr/bin/env python3
"""One-time locked-test evaluation using only frozen development operating points."""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml  # type: ignore[import-untyped]
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.baselines.tuning import risk_coverage_curve
from bces.models.ablations import AblationSurfaceNetLite, ablation_normal_tensor
from bces.models.calibration import clustered_upper_bound, quantize_numpy
from bces.models.dataset import SurfaceDataset, _records_for_split
from bces.models.surfacenet import ScalarTTLLite, SurfaceNetLite, normal_tensor
from bces.models.validity_mlp import ValidityMLPLite
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs/evaluation/final_offline_v1.yaml"


def _move(raw: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in raw.items()}


def _predict(model: torch.nn.Module, dataset: SurfaceDataset, device: torch.device, kind: str) -> np.ndarray:
    rows = []
    model.eval()
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0):
            batch = _move(raw, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(batch)
            value = output if kind == "mlp" else output[0]
            if kind == "mlp":
                value = torch.sigmoid(value)
            rows.append(value.float().cpu().numpy().astype(np.float64))
    return np.concatenate(rows)


def _metrics(accepted: np.ndarray, valid: np.ndarray, clusters: tuple[str, ...], config: dict, seed: int) -> dict:
    accepted, valid = accepted.astype(bool), valid.astype(bool)
    unsafe = accepted & ~valid
    count, valid_count = int(accepted.sum()), int(valid.sum())
    upper = clustered_upper_bound(accepted, unsafe, clusters, confidence=float(config["confidence"]), replicates=int(config["bootstrap_replicates"]), seed=seed)
    return {
        "total": len(valid), "accepted": count, "unsafe_accepted": int(unsafe.sum()),
        "unsafe_accept_rate": int(unsafe.sum()) / count if count else None,
        "unsafe_accept_upper": upper, "coverage": count / len(valid),
        "valid_coverage": int((accepted & valid).sum()) / max(valid_count, 1),
        "safe_reject_rate": int((~accepted & valid).sum()) / max(valid_count, 1),
        "target_met": count >= int(config["minimum_accepted"]) and upper <= float(config["unsafe_accept_upper_target"]),
    }


def _group_metrics(accepted: np.ndarray, valid: np.ndarray, behaviors: np.ndarray, clusters: tuple[str, ...], config: dict, seed: int) -> dict:
    result = {"overall": _metrics(accepted, valid, clusters, config, seed)}
    result["by_behavior"] = {}
    for behavior in range(5):
        indices = np.flatnonzero(behaviors == behavior)
        result["by_behavior"][str(behavior)] = _metrics(accepted[indices], valid[indices], tuple(clusters[index] for index in indices), config, seed + behavior + 1)
    result["all_behavior_targets_met"] = all(row["target_met"] for row in result["by_behavior"].values())
    return result


def _aurc(score: np.ndarray, valid: np.ndarray) -> float:
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="stable")
    risk = np.cumsum(~valid[order].astype(bool)) / np.arange(1, len(valid) + 1)
    coverage = np.arange(1, len(valid) + 1) / len(valid)
    return float(np.trapezoid(risk, coverage))


def _top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(np.floor(len(score) * fraction)))
    accepted = np.zeros(len(score), dtype=bool)
    accepted[np.argsort(score, kind="stable")[:count]] = True
    return accepted


def _rule_scores(dataset: SurfaceDataset) -> dict[str, np.ndarray]:
    reference = dataset.reference_indices.numpy()
    drift = dataset.normalized_drift.numpy().astype(np.float64)
    tensors = dataset.reference_tensors
    scales = tensors["drift_scales"][reference].numpy().astype(np.float64)
    delay = tensors["estimated_delay_s"][reference].numpy().astype(np.float64)
    elapsed = drift[:, 6] * scales[:, 6]
    state_error = np.max(np.abs(drift[:, :6]), axis=1)
    objects = tensors["object_features"][reference].numpy().astype(np.float64)
    mask = tensors["object_mask"][reference].numpy().astype(bool)
    confidence = (objects[:, :, 7] * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    distance = np.hypot(objects[:, :, 0], objects[:, :, 1])
    nearest = np.min(np.where(mask, distance, np.inf), axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, 100.0)
    occlusion = tensors["occlusion_proxy"][reference].numpy().astype(np.float64)
    return {
        "fixed_ttl": elapsed, "aoi_gate": elapsed + delay,
        "aoii_state_error_gate": state_error, "confidence_gate": 1.0 - confidence,
        "voi_relevance_gate": occlusion + np.exp(-nearest / 20.0) * (0.5 + 0.5 * confidence),
    }


def _require_frozen(config: dict) -> dict[str, str]:
    paths = {
        "surface_calibration": ROOT / config["surface_calibration"], "scalar": ROOT / config["scalar_root"] / "run_manifest.json",
        "mlp": ROOT / config["validity_mlp_root"] / "run_manifest.json", "rules": ROOT / config["rule_tuning"],
        "ablations": ROOT / config["ablation_root"] / "run_manifest.json", "geometry": ROOT / config["geometry_audit"],
        "uncertainty": ROOT / config["object_uncertainty"], "second_planner": ROOT / config["second_planner"],
        "sumo_source": ROOT / config["sumo_source_ablation"], "network": ROOT / config["network_grid"],
    }
    hashes = {}
    for name, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status", "PASS") != "PASS" or payload.get("test_partition_accessed", False):
            raise RuntimeError(f"development artifact not frozen and clean: {name}")
        hashes[name] = sha256_file(path)
    return hashes


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["partition"] != "test" or config["selection_on_test"] != "prohibited":
        raise RuntimeError("invalid final-test contract")
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("final_offline_v1 output is immutable; test cannot be reopened")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("final offline evaluation requires a clean committed revision")
    if not torch.cuda.is_available():
        raise RuntimeError("final offline evaluation requires CUDA")
    frozen_hashes = _require_frozen(config)
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data")
    before = build_budget_report(budget)["measurements"]
    dataset = SurfaceDataset(ROOT / config["phase4_root"], "test", allow_test=True)
    valid = dataset.valid.numpy().astype(bool)
    behaviors = dataset.reference_tensors["identifiers"][dataset.reference_indices, 0].numpy().astype(int)
    clusters = dataset.scenario_ids
    point_rows = list(_records_for_split(ROOT / config["phase4_root"] / "validity_points.jsonl.gz", "test"))
    if len(point_rows) != len(dataset):
        raise RuntimeError("test point alignment failed")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    surface = SurfaceNetLite().to(device)
    surface_checkpoint = ROOT / config["surface_root"] / f"seed_{config['surface_seed']}" / "best.pt"
    surface.load_state_dict(torch.load(surface_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    surface_offsets = _predict(surface, dataset, device, "surface")
    surface_calibration = json.loads((ROOT / config["surface_calibration"]).read_text(encoding="utf-8"))
    surface_seed = next(row for row in surface_calibration["seed_results"] if int(row["seed"]) == int(config["surface_seed"]))
    surface_shrinkage = {int(key): float(value["shrinkage"]) for key, value in surface_seed["selections"].items()}
    shrink = np.asarray([surface_shrinkage[value] for value in behaviors])
    calibrated_offsets = quantize_numpy(np.maximum(0.0, surface_offsets - shrink[:, None]))
    surface_score = -(calibrated_offsets - dataset.normalized_drift.numpy().astype(np.float64) @ normal_tensor().numpy().astype(np.float64).T).min(axis=1)
    decisions: dict[str, np.ndarray] = {"surface": surface_score <= -float(config["refresh_guard_band"])}
    scores: dict[str, np.ndarray] = {"surface": surface_score}
    raw_surface_score = -(quantize_numpy(surface_offsets) - dataset.normalized_drift.numpy().astype(np.float64) @ normal_tensor().numpy().astype(np.float64).T).min(axis=1)
    decisions["no_calibration"] = raw_surface_score <= -float(config["refresh_guard_band"])
    scores["no_calibration"] = raw_surface_score

    scalar = ScalarTTLLite().to(device)
    scalar_checkpoint = ROOT / config["scalar_root"] / f"seed_{config['scalar_seed']}" / "best.pt"
    scalar.load_state_dict(torch.load(scalar_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    ttl = _predict(scalar, dataset, device, "scalar")
    scalar_payload = json.loads((ROOT / config["scalar_root"] / "run_manifest.json").read_text(encoding="utf-8"))
    scalar_seed = next(row for row in scalar_payload["seeds"] if int(row["seed"]) == int(config["scalar_seed"]))
    scalar_shrinkage = {int(key): float(value["selection"]["shrinkage"]) for key, value in scalar_seed["calibration"].items()}
    scalar_score = dataset.normalized_drift[:, 6].numpy().astype(np.float64) - np.maximum(0.0, ttl - np.asarray([scalar_shrinkage[value] for value in behaviors]))
    decisions["learned_scalar_ttl"] = scalar_score <= -float(config["refresh_guard_band"])
    decisions["time_only_drift"] = decisions["learned_scalar_ttl"].copy()
    decisions["constant_velocity_plus_strongest_scalar"] = decisions["learned_scalar_ttl"].copy()
    scores["learned_scalar_ttl"] = scalar_score

    mlp = ValidityMLPLite().to(device)
    mlp_checkpoint = ROOT / config["validity_mlp_root"] / f"seed_{config['validity_mlp_seed']}" / "best.pt"
    mlp.load_state_dict(torch.load(mlp_checkpoint, map_location=device, weights_only=True)["model_state_dict"])
    probability = _predict(mlp, dataset, device, "mlp")
    mlp_payload = json.loads((ROOT / config["validity_mlp_root"] / "run_manifest.json").read_text(encoding="utf-8"))
    mlp_seed = next(row for row in mlp_payload["seeds"] if int(row["seed"]) == int(config["validity_mlp_seed"]))
    mlp_threshold = {int(key): float(value["selection"]["threshold"]) for key, value in mlp_seed["calibration"].items()}
    decisions["validity_mlp"] = probability >= np.asarray([mlp_threshold[value] for value in behaviors])
    scores["validity_mlp"] = -probability

    rules = json.loads((ROOT / config["rule_tuning"]).read_text(encoding="utf-8"))["operating_points"]
    rule_scores = _rule_scores(dataset)
    for name, score in rule_scores.items():
        decisions[name] = score <= float(rules[name]["selection"]["threshold"])
        scores[name] = score
    behavior_thresholds = {int(key): float(value["selection"]["threshold"]) for key, value in rules["behavior_specific_ttl"]["behaviors"].items()}
    decisions["behavior_specific_ttl"] = rule_scores["fixed_ttl"] <= np.asarray([behavior_thresholds[value] for value in behaviors])
    scores["behavior_specific_ttl"] = rule_scores["fixed_ttl"] - np.asarray([behavior_thresholds[value] for value in behaviors])
    decisions["ungated_cached_reuse"] = np.ones(len(valid), dtype=bool)
    scores["ungated_cached_reuse"] = np.zeros(len(valid))

    ablation_manifest = json.loads((ROOT / config["ablation_root"] / "run_manifest.json").read_text(encoding="utf-8"))
    ablation_hashes = {}
    for entry in ablation_manifest["variants"]:
        variant = entry["variant"]
        name, count = variant["id"], int(variant["normal_count"])
        model = AblationSurfaceNetLite(count, remove_behavior=bool(variant.get("remove_behavior", False)), remove_objects=bool(variant.get("remove_objects", False))).to(device)
        checkpoint = ROOT / config["ablation_root"] / name / "best.pt"
        if sha256_file(checkpoint) != entry["checkpoint_sha256"]:
            raise ValueError(f"ablation checkpoint hash mismatch: {name}")
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state_dict"])
        offsets = _predict(model, dataset, device, "surface")
        shrinkages = {int(key): float(value["selection"]["shrinkage"]) for key, value in entry["calibration"].items()}
        calibrated = quantize_numpy(np.maximum(0.0, offsets - np.asarray([shrinkages[value] for value in behaviors])[:, None]))
        score = -(calibrated - dataset.normalized_drift.numpy().astype(np.float64) @ ablation_normal_tensor(count).numpy().astype(np.float64).T).min(axis=1)
        decisions[name] = score <= -float(config["refresh_guard_band"])
        scores[name] = score
        ablation_hashes[name] = sha256_file(checkpoint)

    results = {}
    for index, name in enumerate(sorted(decisions)):
        results[name] = _group_metrics(decisions[name], valid, behaviors, clusters, config, int(config["bootstrap_seed"]) + index * 100)
        if name in scores:
            results[name]["risk_coverage_area"] = _aurc(scores[name], valid)
    fraction = float(config["matched_coverage_fraction"])
    matched = {}
    for index, name in enumerate(("surface", "learned_scalar_ttl", "validity_mlp")):
        accepted = _top_fraction(scores[name], fraction)
        matched[name] = _group_metrics(accepted, valid, behaviors, clusters, config, int(config["bootstrap_seed"]) + 5000 + index * 100)
    results["always_fresh_oracle_reference"] = {"accepted_reuse": 0, "fresh_fraction": 1.0, "unsafe_accept_rate": None, "offline_status": "not_a_reuse_gate"}
    results["local_only_reference"] = {"offline_status": "requires_closed_loop_planner_evaluation_in_phase9"}

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / ".final_offline_decisions.jsonl.gz.part"
    temp.unlink(missing_ok=True)
    try:
        with gzip.open(temp, "xt", encoding="utf-8", newline="\n") as handle:
            for index, point in enumerate(point_rows):
                row = {"point_id": point["point_id"], "scenario_id": point["scenario_id"], "behavior": point["behavior"], "valid": bool(valid[index]), "decisions": {name: bool(value[index]) for name, value in decisions.items()}, "scores": {name: float(value[index]) for name, value in scores.items()}}
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        output.mkdir(parents=True, exist_ok=False)
        raw_path = output / "decisions.jsonl.gz"
        os.replace(temp, raw_path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    claims = {
        "five_percent_safety_target": "pass" if results["surface"]["all_behavior_targets_met"] else "fail",
        "nontrivial_coverage": "pass" if results["surface"]["overall"]["coverage"] >= fraction else "fail",
        "surface_lower_matched_coverage_uar_than_scalar": "pass" if matched["surface"]["overall"]["unsafe_accept_rate"] < matched["learned_scalar_ttl"]["overall"]["unsafe_accept_rate"] else "fail",
        "q1_geometric_superiority": "undetermined_pending_phase10_ci",
    }
    report = {
        "schema_version": 1, "phase": 8, "component": "locked_final_offline_evaluation", "status": "PASS",
        "scientific_success": False, "partition": "test", "data": dataset.summary,
        "operating_point_results": results, "matched_coverage_fraction": fraction,
        "matched_coverage_results": matched, "claims": claims,
        "alias_disclosures": {
            "time_only_drift": "identical to learned scalar TTL because its receiver inequality uses only normalized elapsed time",
            "constant_velocity_plus_strongest_scalar": "identical gate; cached cooperative tracks are propagated by constant velocity in the registered oracle",
            "observational_only_labels": "the primary SurfaceNet training source",
        },
        "test_selection_performed": False, "test_partition_accessed": True,
        "first_and_only_test_access": True, "upstream_artifact_sha256": frozen_hashes,
        "checkpoint_sha256": {"surface": sha256_file(surface_checkpoint), "scalar": sha256_file(scalar_checkpoint), "mlp": sha256_file(mlp_checkpoint), **ablation_hashes},
        "config_sha256": sha256_file(CONFIG), "raw_sha256": sha256_file(raw_path),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "budget_before": before, "budget_after": build_budget_report(budget)["measurements"],
        "git": state, "completed_utc": utc_now(),
    }
    report["report_sha256"] = canonical_json_hash(report)
    write_json_atomic(output / "run_manifest.json", report)
    print(json.dumps({"status": report["status"], "claims": claims, "surface": results["surface"]["overall"], "scalar": results["learned_scalar_ttl"]["overall"], "matched": {key: value["overall"] for key, value in matched.items()}, "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

