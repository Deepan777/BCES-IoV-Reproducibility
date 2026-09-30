#!/usr/bin/env python3
"""Independent raw-key, frozen-score, cutoff, and aggregate transfer replay."""

from __future__ import annotations

from collections import defaultdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.decision_surface import DecisionSurfaceNet, membership_score
from bces.utils.hashing import canonical_json_hash, sha256_file

REGISTRATION = ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_REGISTRATION.json"
SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v2"
MODELS = ROOT / "outputs/study_b/diverse_empty_transfer_models_dev_v1"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_teacher_transfer_analysis_dev_v1"
FAMILIES = ("surface_teacher", "scalar_ttl_teacher", "surface_no_teacher", "scalar_ttl_no_teacher")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def _reference_vector(batch):
    values = [float(value) for item in batch["object_features"] for value in item]
    values.extend(float(value) for value in batch["object_mask"])
    values.extend(float(value) for value in batch["reference_state"])
    values.extend(float(value) for value in batch["drift_scales"])
    values.extend(float(value) for item in batch["reference_path"] for value in item)
    values.extend(float(value) for value in batch["identifiers"])
    values.extend((float(batch["object_count"]), float(batch["occlusion_proxy"]),
                   float(batch["estimated_delay_s"]), float(batch["map_context_flags"])))
    return values


def _raw_unique(source_report):
    groups = defaultdict(list)
    for name, digest in sorted(source_report["artifact_sha256"].items()):
        path = SOURCE / name
        if sha256_file(path) != digest:
            raise AssertionError("raw artifact digest mismatch")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        ref = artifact["references"][0]["frozen_input"]
        ref_key = ref["feature_sha256"]
        vector = _reference_vector(ref["batch"])
        for point in artifact["points"]:
            key = canonical_json_hash({
                "reference_feature_sha256": ref_key,
                "normalized_drift": point["normalized_drift"],
                "cache_age_s": point["cache_age_s"],
            })
            groups[key].append({
                "key": key, "slot": f"{name}:{point['current_timestamp_ms']}",
                "seed": artifact["seed"], "reference_key": ref_key,
                "vector": vector, "drift": point["normalized_drift"],
                "valid": bool(point["valid"]), "point": point,
            })
    rows = []
    for key, members in sorted(groups.items()):
        if len({member["seed"] for member in members}) != 1 or len({member["valid"] for member in members}) != 1:
            raise AssertionError("conflicting or cross-seed receiver observables")
        rows.append(min(members, key=lambda item: item["slot"]))
    return rows


def _accepted(values, fraction):
    mask = np.zeros(len(values), dtype=bool)
    mask[np.argsort(-values, kind="stable")[:math.floor(fraction * len(values))]] = True
    return mask


def _counts(mask, valid):
    return {"accepted": int(mask.sum()), "valid_accepted": int((mask & valid).sum()),
            "invalid_accepted": int((mask & ~valid).sum())}


def main():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    model_report = json.loads((MODELS / "report.json").read_text(encoding="utf-8"))
    if (report["protocol_sha256"] != sha256_file(OUTPUT / "protocol.json")
            or protocol["registration_sha256"] != sha256_file(REGISTRATION)
            or protocol["source_report_sha256"] != sha256_file(SOURCE / "report.json")
            or protocol["model_report_sha256"] != sha256_file(MODELS / "report.json")
            or protocol["plan_sha256"] != sha256_file(
                ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_PLAN.md")
            or protocol["script_sha256"] != sha256_file(
                ROOT / "scripts/143_analyze_empty_teacher_transfer_dev.py")):
        raise AssertionError("analysis source/registration digest mismatch")
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise AssertionError("development registration source changed")
    if not source_report["summary"]["engineering_screen_passed"]:
        raise AssertionError("engineering gate did not pass")
    rows = _raw_unique(source_report)
    if len(rows) != report["new_unique_observables"]:
        raise AssertionError("unique observable count differs")
    x = np.asarray([row["vector"] for row in rows], dtype=np.float32)
    drift = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    if x.shape[1] != 448 or drift.shape[1] != 7 or len(np.unique(seeds)) != 64:
        raise AssertionError("transfer feature/seed shape differs")
    with gzip.open(OUTPUT / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        predictions = {row["observable_key"]: row for row in (json.loads(line) for line in handle)}
    if (sha256_file(OUTPUT / "predictions.jsonl.gz") != report["predictions_sha256"]
            or len(predictions) != len(rows)):
        raise AssertionError("prediction file incomplete")
    old_registration = json.loads((ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json")
                                  .read_text(encoding="utf-8"))
    limits = yaml.safe_load((ROOT / old_registration["config"]["validity"])
                            .read_text(encoding="utf-8"))
    summaries = {}
    all_scores = {}
    all_masks = {}
    cutoff_ties = {}
    for family in FAMILIES:
        checkpoint_path = MODELS / f"{family}.pt"
        if sha256_file(checkpoint_path) != model_report["models"][family]["checkpoint_sha256"]:
            raise AssertionError("model checkpoint changed")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        scalar = family.startswith("scalar_ttl")
        model = DecisionSurfaceNet(448, scalar=scalar)
        model.load_state_dict(saved["model_state"])
        model.eval()
        normal = np.clip((x - saved["mean"].numpy()) / saved["std"].numpy(), -10, 10).astype(np.float32)
        parts = []
        with torch.no_grad():
            for start in range(0, len(rows), 256):
                end = start + 256
                offsets, logits = model(torch.from_numpy(normal[start:end]))
                parts.append(membership_score(
                    offsets, logits, torch.from_numpy(drift[start:end]),
                    scalar=scalar).numpy())
        scores = np.concatenate(parts)
        all_scores[family] = scores
        masks = {str(fraction): _accepted(scores, fraction) for fraction in FRACTIONS}
        all_masks[family] = masks
        summaries[family] = {
            "literal_zero_threshold": _counts(scores >= 0, valid),
            "matched_coverage": {str(fraction): _counts(masks[str(fraction)], valid)
                                 for fraction in FRACTIONS},
        }
        for index, row in enumerate(rows):
            prediction = predictions[row["key"]]
            if (prediction["seed"] != row["seed"] or prediction["valid"] != row["valid"]
                    or prediction["reference_key"] != row["reference_key"]
                    or abs(prediction["score"][family] - scores[index]) > 1e-6
                    or any(prediction["accepted"][family][str(fraction)]
                           != bool(masks[str(fraction)][index]) for fraction in FRACTIONS)):
                raise AssertionError("saved prediction does not replay")
        order = np.argsort(-scores, kind="stable")
        k = math.floor(0.4 * len(scores))
        boundary = scores[order[k - 1]]
        cutoff_ties[family] = {
            "boundary_score": float(boundary),
            "equal_score_total": int((scores == boundary).sum()),
            "equal_score_accepted": int((masks["0.4"] & (scores == boundary)).sum()),
        }
    component_names = ("regret", "cached_risk", "trajectory_deviation")
    for index, row in enumerate(rows):
        point = row["point"]
        flags = {
            "regret": point["cost_regret"] > limits["max_cost_regret"] and not math.isclose(
                point["cost_regret"], limits["max_cost_regret"], rel_tol=1e-12, abs_tol=1e-12),
            "cached_risk": point["cached_risk"] > limits["max_cached_risk"] and not math.isclose(
                point["cached_risk"], limits["max_cached_risk"], rel_tol=1e-12, abs_tol=1e-12),
            "trajectory_deviation": point["trajectory_deviation_m"] > limits["max_trajectory_deviation_m"] and not math.isclose(
                point["trajectory_deviation_m"], limits["max_trajectory_deviation_m"], rel_tol=1e-12, abs_tol=1e-12),
        }
        if flags != predictions[row["key"]]["invalid_components"]:
            raise AssertionError("validity component differs")
    for family in FAMILIES:
        mask = all_masks[family]["0.4"]
        summaries[family]["invalid_components_at_40_percent"] = {
            key: sum(bool(mask[index] and predictions[row["key"]]["invalid_components"][key])
                     for index, row in enumerate(rows)) for key in component_names
        }
    if summaries != report["summary"]:
        raise AssertionError("aggregate counts differ")
    per_seed = {}
    for seed in np.unique(seeds):
        within = seeds == seed
        per_seed[str(seed)] = {family: _counts(all_masks[family]["0.4"] & within, valid)
                               for family in FAMILIES}
    if per_seed != report["per_seed_40_percent"]:
        raise AssertionError("per-seed counts differ")
    difference = np.asarray([
        per_seed[str(seed)]["surface_teacher"]["invalid_accepted"]
        - per_seed[str(seed)]["scalar_ttl_teacher"]["invalid_accepted"]
        for seed in np.unique(seeds)], dtype=float)
    invalid = np.asarray([per_seed[str(seed)]["surface_teacher"]["invalid_accepted"]
                          for seed in np.unique(seeds)], dtype=float)
    accepted = np.asarray([per_seed[str(seed)]["surface_teacher"]["accepted"]
                           for seed in np.unique(seeds)], dtype=float)
    rng = np.random.default_rng(20260925)
    draws = rng.integers(0, len(difference), size=(10000, len(difference)))
    diff_interval = np.quantile(difference[draws].sum(axis=1), [0.025, 0.975]).tolist()
    rate_interval = np.quantile(invalid[draws].sum(axis=1) / accepted[draws].sum(axis=1),
                                [0.025, 0.975]).tolist()
    boot = report["cluster_bootstrap_40_percent"]
    if (boot["surface_minus_ttl_invalid_count"]["percentile_95_interval"] != diff_interval
            or boot["surface_invalid_accept_rate"]["percentile_95_interval"] != rate_interval
            or boot["surface_minus_ttl_invalid_count"]["estimate"] != float(difference.sum())
            or not math.isclose(boot["surface_invalid_accept_rate"]["estimate"],
                                float(invalid.sum() / accepted.sum()), rel_tol=0, abs_tol=1e-15)):
        raise AssertionError("cluster bootstrap differs")
    gates = report["advancement_gate"]
    expected_gate = (
        summaries["surface_teacher"]["matched_coverage"]["0.4"]["invalid_accepted"]
        < summaries["scalar_ttl_teacher"]["matched_coverage"]["0.4"]["invalid_accepted"]
        and summaries["surface_teacher"]["matched_coverage"]["0.4"]["invalid_accepted"]
        < summaries["surface_no_teacher"]["matched_coverage"]["0.4"]["invalid_accepted"]
        and diff_interval[1] < 0 and rate_interval[1] < 0.05)
    if gates["candidate_for_later_closed_loop_work"] != expected_gate:
        raise AssertionError("advancement gate differs")
    print(json.dumps({"status": "PASS", "unique_observables": len(rows),
                      "cutoff_ties_40_percent": cutoff_ties,
                      "primary_invalid_accepts": {
                          family: summaries[family]["matched_coverage"]["0.4"]["invalid_accepted"]
                          for family in FAMILIES},
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
