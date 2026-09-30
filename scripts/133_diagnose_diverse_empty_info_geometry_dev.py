#!/usr/bin/env python3
"""Frozen development-only receiver-information and ideal-geometry audit."""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.decision_geometry import fit_zero_error_surface, fit_zero_error_ttl
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.utils.hashing import canonical_json_hash, sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_info_geometry_dev_v1"
SOURCE_SHA256 = "a61c0717767f74f888526cd8e5ac119ee6ecf040421368eef6ad4d27ee88dd2e"
PLAN = ROOT / "docs/STUDY_B_DIVERSE_EMPTY_INFO_GEOMETRY_DEV_V1_PLAN.md"
FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def _reference_vector(batch):
    values = [float(value) for row in batch["object_features"] for value in row]
    values.extend(float(value) for value in batch["object_mask"])
    values.extend(float(value) for value in batch["reference_state"])
    values.extend(float(value) for value in batch["drift_scales"])
    values.extend(float(value) for row in batch["reference_path"] for value in row)
    values.extend(float(value) for value in batch["identifiers"])
    values.extend((float(batch["object_count"]),
                   float(batch["occlusion_proxy"]),
                   float(batch["estimated_delay_s"]),
                   float(batch["map_context_flags"])))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite frozen reference feature")
    return values


def _load_unique(report):
    groups = defaultdict(list)
    for name, digest in sorted(report["artifact_sha256"].items()):
        path = SOURCE / name
        if sha256_file(path) != digest:
            raise RuntimeError("raw artifact changed: " + name)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if len(artifact["references"]) != 1:
            raise RuntimeError("reference missing")
        reference = artifact["references"][0]
        ref_key = reference["frozen_input"]["feature_sha256"]
        vector = _reference_vector(reference["frozen_input"]["batch"])
        for point in artifact["points"]:
            key = canonical_json_hash({
                "reference_feature_sha256": ref_key,
                "normalized_drift": point["normalized_drift"],
                "cache_age_s": point["cache_age_s"],
            })
            groups[key].append({
                "observable_key": key,
                "seed": artifact["seed"],
                "slot_id": f"{name}:{point['current_timestamp_ms']}",
                "reference_key": ref_key,
                "reference_vector": vector,
                "drift": point["normalized_drift"],
                "age_s": point["cache_age_s"],
                "valid": bool(point["valid"]),
            })
    rows = []
    for key, members in sorted(groups.items()):
        if len({row["seed"] for row in members}) != 1:
            raise RuntimeError("receiver observable shared across seed clusters")
        if len({row["valid"] for row in members}) != 1:
            raise RuntimeError("receiver observable has conflicting labels")
        representative = min(members, key=lambda row: row["slot_id"])
        rows.append({**representative,
                     "multiplicity": len(members),
                     "member_slot_ids": sorted(row["slot_id"] for row in members)})
    if len(rows) != report["summary"]["distinct_observable_keys"]:
        raise RuntimeError("distinct-observable count differs from source report")
    return rows


def _score_view(x, valid, seeds):
    oof = np.full(len(valid), np.nan, dtype=np.float64)
    fold_records = []
    for fold in range(8):
        validation = seeds % 8 == fold
        training = ~validation
        if not validation.any() or not training.any():
            raise RuntimeError("empty seed-disjoint fold")
        train_classes = np.unique(valid[training])
        if len(train_classes) < 2:
            raise RuntimeError("one-class training fold")
        model = ExtraTreesClassifier(
            n_estimators=256, min_samples_leaf=4, max_features="sqrt",
            random_state=20260925, n_jobs=1, class_weight="balanced")
        model.fit(x[training], valid[training])
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        oof[validation] = model.predict_proba(x[validation])[:, positive]
        fold_records.append({
            "fold": fold,
            "train_seed_count": len(np.unique(seeds[training])),
            "validation_seed_count": len(np.unique(seeds[validation])),
            "validation_rows": int(validation.sum()),
            "validation_valid": int(valid[validation].sum()),
            "validation_invalid": int((~valid[validation]).sum()),
            "both_classes": len(np.unique(valid[validation])) == 2,
        })
    if not np.isfinite(oof).all():
        raise RuntimeError("nonfinite or missing out-of-fold score")
    ranking = np.argsort(-oof, kind="stable")
    coverage = {}
    per_seed_at_40 = {}
    for fraction in FRACTIONS:
        accepted = np.zeros(len(valid), dtype=bool)
        accepted[ranking[:math.floor(fraction * len(valid))]] = True
        coverage[str(fraction)] = {
            "accepted": int(accepted.sum()),
            "invalid_accepted": int((accepted & ~valid).sum()),
            "valid_accepted": int((accepted & valid).sum()),
            "adverse_fraction": float((accepted & ~valid).sum() / accepted.sum()),
        }
        if fraction == 0.4:
            per_seed_at_40 = {
                str(seed): {
                    "accepted": int((accepted & (seeds == seed)).sum()),
                    "invalid_accepted": int((accepted & ~valid & (seeds == seed)).sum()),
                } for seed in sorted(np.unique(seeds))
            }
    return oof, {
        "auroc": float(roc_auc_score(valid, oof)),
        "average_precision": float(average_precision_score(valid, oof)),
        "brier": float(brier_score_loss(valid, oof)),
        "coverage": coverage,
        "per_seed_at_40_percent": per_seed_at_40,
        "folds": fold_records,
    }


def _geometry(rows):
    by_ref = defaultdict(list)
    for row in rows:
        by_ref[row["reference_key"]].append(row)
    results = []
    per_seed = defaultdict(lambda: {"surface": 0, "ttl": 0})
    for key, members in sorted(by_ref.items()):
        if len({row["seed"] for row in members}) != 1:
            raise RuntimeError("sender reference shared across seed clusters")
        drift = np.asarray([row["drift"] for row in members], dtype=np.float64)
        valid = np.asarray([row["valid"] for row in members], dtype=bool)
        ttl = fit_zero_error_ttl(drift[:, 6], valid)
        surface = fit_zero_error_surface(
            drift, valid, NORMAL_CODEBOOK_V1, time_limit=1.0)
        t_accept = np.asarray(ttl["accepted"], dtype=bool)
        s_accept = np.asarray(surface["accepted"], dtype=bool)
        if (t_accept & ~valid).any() or (s_accept & ~valid).any():
            raise AssertionError("oracle geometry accepted an invalid sample")
        seed = members[0]["seed"]
        per_seed[seed]["ttl"] += int(t_accept.sum())
        per_seed[seed]["surface"] += int(s_accept.sum())
        results.append({
            "reference_key": key,
            "seed": seed,
            "points": len(members),
            "valid": int(valid.sum()),
            "ttl_accepted_valid": int(t_accept.sum()),
            "surface_accepted_valid": int(s_accept.sum()),
            "surface_status": surface["status"],
            "surface_optimal": bool(surface["optimal"]),
            "surface_abstain": bool(surface["abstain"]),
        })
    return results, {
        "references": len(results),
        "ttl_accepted_valid": sum(row["ttl_accepted_valid"] for row in results),
        "surface_accepted_valid": sum(row["surface_accepted_valid"] for row in results),
        "surface_optimal_references": sum(row["surface_optimal"] for row in results),
        "surface_abstain_references": sum(row["surface_abstain"] for row in results),
        "per_seed": {str(seed): {
            **counts,
            "surface_minus_ttl": counts["surface"] - counts["ttl"],
        } for seed, counts in sorted(per_seed.items())},
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("frozen development source report changed")
    report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if not report["summary"]["engineering_screen_passed"]:
        raise RuntimeError("source engineering screen failed")
    rows = _load_unique(report)
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    age = np.asarray([[row["age_s"]] for row in rows], dtype=np.float32)
    reference = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    drift = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    if len(np.unique(seeds)) != 64 or len(rows) != 2688:
        raise RuntimeError("expected 64 independent seeds and 2688 observable rows")
    views = {
        "age_only": age,
        "frozen_reference_plus_age": np.column_stack((reference, age)),
        "frozen_reference_plus_full_drift": np.column_stack((reference, age, drift)),
    }
    information = {}
    predictions = []
    for view_name, x in views.items():
        oof, metrics = _score_view(x, valid, seeds)
        information[view_name] = metrics
        predictions.extend({
            "observable_key": row["observable_key"],
            "seed": row["seed"], "valid": row["valid"],
            "view": view_name, "oof_valid_probability": float(score),
        } for row, score in zip(rows, oof))
        print(json.dumps({"completed_information_view": view_name,
                          "auroc": metrics["auroc"]}), flush=True)
    geometry_rows, geometry = _geometry(rows)
    protocol = {
        "status": "DIVERSE_EMPTY_INFO_GEOMETRY_DEVELOPMENT_ONLY_V1",
        "plan_sha256": sha256_file(PLAN),
        "script_sha256": sha256_file(Path(__file__)),
        "source_report_sha256": SOURCE_SHA256,
        "normal_codebook": "NORMAL_CODEBOOK_V1",
        "milp_time_limit_s": 1.0,
        "unique_observable_rows": len(rows),
        "seed_folds": "seed modulo 8",
        "no_surface_or_ttl_model_fitted": True,
        "independent_confirmation": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    with gzip.open(OUTPUT / "predictions.jsonl.gz", "xt", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (OUTPUT / "geometry_references.json").write_text(
        json.dumps(geometry_rows, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": protocol["status"],
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifacts_sha256": {
            name: sha256_file(OUTPUT / name) for name in (
                "predictions.jsonl.gz", "geometry_references.json")
        },
        "distinct_rows": len(rows),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
        "information": information,
        "geometry": geometry,
        "not_predictive_surface_comparison": True,
        "not_independent_confirmation": True,
        "regret_theorem_certified": False,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"distinct_rows": len(rows),
                      "information": {name: {key: value for key, value in metrics.items()
                                             if key in ("auroc", "average_precision", "brier", "coverage")}
                                      for name, metrics in information.items()},
                      "geometry": {key: value for key, value in geometry.items()
                                   if key != "per_seed"},
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
