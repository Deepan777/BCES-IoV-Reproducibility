#!/usr/bin/env python3
"""Reload every fold checkpoint and replay equal-input OOF receiver decisions."""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.decision_surface import DecisionSurfaceNet, membership_score
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
DIAGNOSTIC = ROOT / "outputs/study_b/diverse_empty_info_geometry_dev_v1"
DIR = ROOT / "outputs/study_b/diverse_empty_surface_ttl_learning_dev_v1"
FAMILIES = ("surface", "scalar_ttl")


def _rows(source_report):
    path = ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("empty_features_for_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_unique(source_report)


def _matched(score, valid, fraction):
    accepted = np.zeros(len(score), dtype=bool)
    accepted[np.argsort(-score, kind="stable")[:math.floor(fraction * len(score))]] = True
    return accepted, {
        "accepted": int(accepted.sum()),
        "valid_accepted": int((accepted & valid).sum()),
        "invalid_accepted": int((accepted & ~valid).sum()),
    }


def main():
    protocol = json.loads((DIR / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((DIR / "report.json").read_text(encoding="utf-8"))
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if (sha256_file(DIR / "protocol.json") != report["protocol_sha256"]
            or sha256_file(SOURCE / "report.json") != protocol["source_report_sha256"]
            or sha256_file(DIAGNOSTIC / "report.json") != protocol["diagnostic_report_sha256"]
            or sha256_file(ROOT / "docs/STUDY_B_DIVERSE_EMPTY_SURFACE_TTL_LEARNING_DEV_V1_PLAN.md") != protocol["plan_sha256"]
            or sha256_file(ROOT / "scripts/135_train_diverse_empty_surface_ttl_dev.py") != protocol["script_sha256"]
            or sha256_file(ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py") != protocol["feature_source_sha256"]
            or sha256_file(ROOT / "bces/models/decision_surface.py") != protocol["decision_surface_source_sha256"]):
        raise AssertionError("learning protocol/source binding changed")
    rows = _rows(source)
    if len(rows) != 2688:
        raise AssertionError("distinct receiver-observable grid changed")
    x = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    drift = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    predictions = {}
    with gzip.open(DIR / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["observable_key"]
            if key in predictions:
                raise AssertionError("duplicate prediction key")
            predictions[key] = row
    if (sha256_file(DIR / "predictions.jsonl.gz") != report["predictions_sha256"]
            or set(predictions) != {row["observable_key"] for row in rows}):
        raise AssertionError("OOF prediction artifact incomplete")
    score = {family: np.full(len(rows), np.nan) for family in FAMILIES}
    summaries = {family: {"literal_zero_threshold": {
        "accepted": 0, "valid_accepted": 0, "invalid_accepted": 0},
        "matched_coverage": {
            str(fraction): {"accepted": 0, "valid_accepted": 0,
                            "invalid_accepted": 0}
            for fraction in (0.2, 0.4, 0.6, 0.8)}} for family in FAMILIES}
    paired_40 = {}
    parameter_counts = {}
    for fold in range(8):
        test = seeds % 8 == fold
        train = ~test
        indices = np.flatnonzero(test)
        for family in FAMILIES:
            name = f"{family}_fold{fold}"
            checkpoint = DIR / (name + ".pt")
            progress = DIR / (name + ".json")
            if (sha256_file(checkpoint) != report["checkpoint_sha256"][checkpoint.name]
                    or sha256_file(progress) != report["fold_progress_sha256"][progress.name]):
                raise AssertionError("checkpoint/progress digest mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (saved["family"] != family or saved["scalar"] != (family == "scalar_ttl")
                    or saved["fold"] != fold or saved["feature_count"] != 448
                    or saved["policy_hash"] != protocol["policy_hash"]):
                raise AssertionError("checkpoint policy/family mismatch")
            mean = x[train].mean(0)
            std = np.maximum(x[train].std(0), 0.01)
            if not np.allclose(saved["mean"].numpy(), mean, atol=0, rtol=0):
                raise AssertionError("checkpoint preprocessing mean leaked")
            if not np.allclose(saved["std"].numpy(), std, atol=0, rtol=0):
                raise AssertionError("checkpoint preprocessing std leaked")
            model = DecisionSurfaceNet(448, scalar=(family == "scalar_ttl"))
            model.load_state_dict(saved["model_state"])
            model.eval()
            parameter_counts[family] = trainable_parameter_count(model)
            normalized = np.clip((x[indices] - mean) / std, -10, 10).astype(np.float32)
            with torch.no_grad():
                offsets, logits = model(torch.from_numpy(normalized))
                calculated = membership_score(
                    offsets, logits, torch.from_numpy(drift[indices]),
                    scalar=model.scalar).numpy()
            fold_result = report["fold_results"][f"{family}:{fold}"]
            if (fold_result["parameter_count"] != parameter_counts[family]
                    or len(fold_result["history"]) != 60
                    or fold_result["test_rows"] != len(indices)
                    or fold_result["test_seed_count"] != len(np.unique(seeds[indices]))):
                raise AssertionError("fold model/history mismatch")
            saved_progress = json.loads(progress.read_text(encoding="utf-8"))
            if (saved_progress["test_indices"] != indices.tolist()
                    or saved_progress["fold_result"] != fold_result
                    or not np.allclose(calculated, saved_progress["score"], atol=1e-6, rtol=0)):
                raise AssertionError("checkpoint inference differs from saved OOF score")
            score[family][indices] = calculated
            literal = calculated >= 0
            literal_counts = {
                "accepted": int(literal.sum()),
                "valid_accepted": int((literal & valid[indices]).sum()),
                "invalid_accepted": int((literal & ~valid[indices]).sum()),
            }
            if fold_result["literal_zero_threshold"] != literal_counts:
                raise AssertionError("literal threshold count mismatch")
            for field, value in literal_counts.items():
                summaries[family]["literal_zero_threshold"][field] += value
            for fraction in (0.2, 0.4, 0.6, 0.8):
                _, counts = _matched(calculated, valid[indices], fraction)
                if fold_result["matched_coverage"][str(fraction)] != counts:
                    raise AssertionError("matched-coverage fold count mismatch")
                for field, value in counts.items():
                    summaries[family]["matched_coverage"][str(fraction)][field] += value
        accepted = {
            family: _matched(score[family][indices], valid[indices], 0.4)[0]
            for family in FAMILIES
        }
        for seed in np.unique(seeds[indices]):
            within = seeds[indices] == seed
            paired_40[str(seed)] = {
                family: {
                    "accepted": int((accepted[family] & within).sum()),
                    "invalid_accepted": int((accepted[family] & within & ~valid[indices]).sum()),
                } for family in FAMILIES
            }
    for index, row in enumerate(rows):
        saved = predictions[row["observable_key"]]
        if saved["seed"] != row["seed"] or saved["valid"] != row["valid"]:
            raise AssertionError("prediction provenance mismatch")
        for family in FAMILIES:
            if abs(saved["score"][family] - score[family][index]) > 1e-6:
                raise AssertionError("OOF prediction differs from checkpoint")
    if summaries != report["summary"] or paired_40 != report["paired_seed_40_percent"]:
        raise AssertionError("overall or paired-seed report mismatch")
    print(json.dumps({"status": "PASS", "rows": len(rows),
                      "parameter_counts": parameter_counts,
                      "summary": summaries,
                      "report_sha256": sha256_file(DIR / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
