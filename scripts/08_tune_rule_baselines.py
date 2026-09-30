#!/usr/bin/env python3
"""Freeze validation-only operating points for deterministic rule baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.baselines.tuning import risk_coverage_curve, select_risk_constrained_threshold
from bces.models.dataset import SurfaceDataset
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / "configs" / "evaluation" / "baseline_tuning_v1.yaml"


def _thresholds(scores: np.ndarray, count: int) -> tuple[float, ...]:
    values = np.quantile(scores, np.linspace(0.0, 1.0, count))
    return tuple(float(value) for value in np.unique(values))


def _score_arrays(dataset: SurfaceDataset) -> dict[str, np.ndarray]:
    reference = dataset.reference_indices.numpy()
    drift = dataset.normalized_drift.numpy().astype(np.float64)
    tensors = dataset.reference_tensors
    scales = tensors["drift_scales"][reference].numpy().astype(np.float64)
    delay = tensors["estimated_delay_s"][reference].numpy().astype(np.float64)
    elapsed = drift[:, 6] * scales[:, 6]
    state_error = np.max(np.abs(drift[:, :6]), axis=1)
    objects = tensors["object_features"][reference].numpy().astype(np.float64)
    mask = tensors["object_mask"][reference].numpy().astype(bool)
    confidence_sum = (objects[:, :, 7] * mask).sum(axis=1)
    object_count = np.maximum(mask.sum(axis=1), 1)
    mean_confidence = confidence_sum / object_count
    distance = np.hypot(objects[:, :, 0], objects[:, :, 1])
    nearest = np.min(np.where(mask, distance, np.inf), axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, 100.0)
    occlusion = tensors["occlusion_proxy"][reference].numpy().astype(np.float64)
    relevance = occlusion + np.exp(-nearest / 20.0) * (0.5 + 0.5 * mean_confidence)
    return {
        "fixed_ttl": elapsed,
        "aoi_gate": elapsed + delay,
        "aoii_state_error_gate": state_error,
        "confidence_gate": 1.0 - mean_confidence,
        "voi_relevance_gate": relevance,
    }


def _tune(
    scores: np.ndarray,
    valid: np.ndarray,
    clusters: tuple[str, ...],
    config: dict,
    *,
    seed: int,
) -> dict:
    curve = risk_coverage_curve(
        scores,
        valid,
        clusters,
        _thresholds(scores, int(config["threshold_grid_points"])),
        confidence=float(config["confidence"]),
        bootstrap_replicates=int(config["bootstrap_replicates"]),
        seed=seed,
    )
    return {
        "selection": select_risk_constrained_threshold(
            curve,
            unsafe_upper_target=float(config["unsafe_accept_upper_target"]),
            minimum_accepted=int(config["minimum_accepted"]),
        ),
        "curve": curve,
        "cluster_count": len(set(clusters)),
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    if output.exists():
        raise FileExistsError("baseline_tuning_v1 output is immutable")
    state = git_state(ROOT)
    if state["dirty"]:
        raise RuntimeError("rule-baseline tuning requires a clean committed revision")
    dataset = SurfaceDataset(ROOT / config["artifact_root"], config["partition"])
    valid = dataset.valid.numpy().astype(bool)
    clusters = dataset.scenario_ids
    scores = _score_arrays(dataset)
    tuned = {}
    for index, (name, values) in enumerate(scores.items()):
        tuned[name] = _tune(
            values,
            valid,
            clusters,
            config,
            seed=int(config["bootstrap_seed"]) + index * 1000,
        )
    reference = dataset.reference_indices.numpy()
    behaviors = (
        dataset.reference_tensors["identifiers"][reference, 0].numpy().astype(int)
    )
    elapsed = scores["fixed_ttl"]
    behavior_results = {}
    for behavior in range(5):
        indices = np.flatnonzero(behaviors == behavior)
        behavior_results[str(behavior)] = _tune(
            elapsed[indices],
            valid[indices],
            tuple(clusters[index] for index in indices),
            config,
            seed=int(config["bootstrap_seed"]) + 10000 + behavior * 1000,
        )
    tuned["behavior_specific_ttl"] = {
        "behaviors": behavior_results,
        "all_targets_met": all(
            row["selection"]["target_met"] for row in behavior_results.values()
        ),
    }
    accepted = np.ones(len(valid), dtype=bool)
    unsafe_count = int((accepted & ~valid).sum())
    tuned["ungated_cached_reuse"] = {
        "accepted": len(valid),
        "unsafe_accepted": unsafe_count,
        "unsafe_accept_rate": unsafe_count / len(valid),
        "coverage": 1.0,
    }
    linked = {}
    for name, key in (
        ("surface", "surface_calibration"),
        ("learned_scalar_ttl", "scalar_manifest"),
        ("validity_mlp", "validity_mlp_manifest"),
    ):
        path = ROOT / config[key]
        linked[name] = {"path": config[key], "sha256": sha256_file(path)}
    manifest = {
        "schema_version": 1,
        "phase": 8,
        "component": "rule_baseline_tuning",
        "status": "PASS",
        "partition": config["partition"],
        "data": dataset.summary,
        "score_contracts": config["score_contracts"],
        "operating_points": tuned,
        "linked_learned_artifacts": linked,
        "config_sha256": sha256_file(CONFIG),
        "test_partition_accessed": False,
        "git": state,
        "completed_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "tuning.json", manifest)
    summary = {
        name: (
            value["selection"]["target_met"]
            if "selection" in value
            else value.get("all_targets_met")
        )
        for name, value in tuned.items()
    }
    print(json.dumps({"status": "PASS", "targets_met": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
