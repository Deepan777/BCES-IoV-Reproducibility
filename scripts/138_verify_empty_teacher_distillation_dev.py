#!/usr/bin/env python3
"""Independently replay teacher scope, geometry, checkpoints, and OOF counts."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.decision_geometry import fit_zero_error_surface, fit_zero_error_ttl
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.models.decision_surface import DecisionSurfaceNet, membership_score
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_teacher_distillation_dev_v1"
FAMILIES = ("surface", "scalar_ttl")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def _rows(source):
    path = ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("empty_teacher_verify_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_unique(source)


def _matched(score, valid, fraction):
    accepted = np.zeros(len(score), dtype=bool)
    accepted[np.argsort(-score, kind="stable")[:math.floor(fraction * len(score))]] = True
    return accepted, {
        "accepted": int(accepted.sum()),
        "valid_accepted": int((accepted & valid).sum()),
        "invalid_accepted": int((accepted & ~valid).sum()),
    }


def main():
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "source_report_sha256": sha256_file(SOURCE / "report.json"),
        "baseline_report_sha256": sha256_file(
            ROOT / "outputs/study_b/diverse_empty_surface_ttl_learning_dev_v1/report.json"),
        "plan_sha256": sha256_file(ROOT / "docs/STUDY_B_EMPTY_TEACHER_DISTILLATION_DEV_V1_PLAN.md"),
        "script_sha256": sha256_file(ROOT / "scripts/137_train_empty_teacher_distillation_dev.py"),
        "feature_source_sha256": sha256_file(
            ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"),
        "model_source_sha256": sha256_file(ROOT / "bces/models/decision_surface.py"),
        "geometry_source_sha256": sha256_file(ROOT / "bces/evaluation/decision_geometry.py"),
    }
    if report["protocol_sha256"] != expected.pop("protocol_sha256"):
        raise AssertionError("protocol digest mismatch")
    for key, digest in expected.items():
        if protocol[key] != digest:
            raise AssertionError("source/plan digest mismatch: " + key)
    if (protocol["teacher_scope"] != "training_references_only"
            or not protocol["selection_biased_after_error_inspection"]
            or report["independent_confirmation"] or report["source_view_certified"]):
        raise AssertionError("exploratory scope misreported")
    rows = _rows(source)
    if len(rows) != 2688:
        raise AssertionError("unexpected observable count")
    x = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    z = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    ref_keys = [row["reference_key"] for row in rows]
    if len(np.unique(seeds)) != 64:
        raise AssertionError("seed cohort changed")
    with gzip.open(OUTPUT / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        saved_predictions = {entry["observable_key"]: entry
                             for entry in (json.loads(line) for line in handle)}
    if (sha256_file(OUTPUT / "predictions.jsonl.gz") != report["predictions_sha256"]
            or len(saved_predictions) != len(rows)):
        raise AssertionError("prediction artifact incomplete")
    score = {family: np.full(len(rows), np.nan) for family in FAMILIES}
    summaries = {family: {
        "literal_zero_threshold": {key: 0 for key in ("accepted", "valid_accepted", "invalid_accepted")},
        "matched_coverage": {str(fraction): {
            key: 0 for key in ("accepted", "valid_accepted", "invalid_accepted")}
            for fraction in FRACTIONS},
    } for family in FAMILIES}
    paired = {}
    fitted_cache = {}
    for fold in range(8):
        test = seeds % 8 == fold
        train = ~test
        indices = np.flatnonzero(test)
        train_refs = defaultdict(list)
        for index in np.flatnonzero(train):
            train_refs[ref_keys[index]].append(int(index))
        teacher_path = OUTPUT / f"teachers_fold{fold}.json"
        if sha256_file(teacher_path) != report["teacher_sha256"][teacher_path.name]:
            raise AssertionError("teacher audit digest mismatch")
        teacher_audit = json.loads(teacher_path.read_text(encoding="utf-8"))
        entries = teacher_audit["references"]
        if (teacher_audit["fold"] != fold or teacher_audit["scope"] != "training_references_only"
                or {entry["reference_key"] for entry in entries} != set(train_refs)
                or len(entries) != len(train_refs)):
            raise AssertionError("teacher used wrong reference set")
        for entry in entries:
            ref = entry["reference_key"]
            members = train_refs[ref]
            if (entry["training_rows"] != len(members)
                    or entry["training_seed"] != int(seeds[members[0]])
                    or entry["training_seed"] % 8 == fold):
                raise AssertionError("teacher train/test leakage")
            if ref not in fitted_cache:
                fitted_cache[ref] = (
                    fit_zero_error_surface(z[members], valid[members], NORMAL_CODEBOOK_V1,
                                           time_limit=1.0),
                    fit_zero_error_ttl(z[members, 6], valid[members]),
                )
            surface, ttl = fitted_cache[ref]
            if (entry["surface_abstain"] != surface["abstain"]
                    or entry["surface_status"] != surface["status"]
                    or entry["ttl_abstain"] != ttl["abstain"]
                    or entry["ttl_threshold"] != ttl["threshold"]):
                raise AssertionError("teacher fit replay differs")
            if surface["offsets"] is None:
                if entry["surface_offsets"] is not None:
                    raise AssertionError("abstaining surface has offsets")
            elif not np.allclose(entry["surface_offsets"], surface["offsets"], atol=1e-8, rtol=0):
                raise AssertionError("teacher surface offsets differ")
        mean = x[train].mean(axis=0)
        std = np.maximum(x[train].std(axis=0), 0.01)
        normalized = np.clip((x[indices] - mean) / std, -10, 10).astype(np.float32)
        for family in FAMILIES:
            name = f"{family}_fold{fold}"
            checkpoint_path = OUTPUT / (name + ".pt")
            progress_path = OUTPUT / (name + ".json")
            if (sha256_file(checkpoint_path) != report["checkpoint_sha256"][checkpoint_path.name]
                    or sha256_file(progress_path) != report["fold_progress_sha256"][progress_path.name]):
                raise AssertionError("checkpoint/progress digest mismatch")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (checkpoint["fold"] != fold or checkpoint["family"] != family
                    or checkpoint["scalar"] != (family == "scalar_ttl")
                    or checkpoint["training_teacher_sha256"] != sha256_file(teacher_path)
                    or checkpoint["source_report_sha256"] != protocol["source_report_sha256"]
                    or checkpoint["policy_hash"] != protocol["policy_hash"]
                    or not np.array_equal(checkpoint["mean"].numpy(), mean)
                    or not np.array_equal(checkpoint["std"].numpy(), std)):
                raise AssertionError("checkpoint provenance or train-only normalization differs")
            model = DecisionSurfaceNet(448, scalar=(family == "scalar_ttl"))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            with torch.no_grad():
                offsets, logits = model(torch.from_numpy(normalized))
                calculated = membership_score(
                    offsets, logits, torch.from_numpy(z[indices]), scalar=model.scalar).numpy()
            by_ref = defaultdict(list)
            for pos, index in enumerate(indices):
                by_ref[ref_keys[index]].append(pos)
            if any(not np.allclose(offsets.numpy()[positions], offsets.numpy()[positions[0]],
                                   atol=1e-6, rtol=0) for positions in by_ref.values()):
                raise AssertionError("sender offsets depend on receiver drift")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            fold_result = report["fold_results"][f"{family}:{fold}"]
            if (progress["test_indices"] != indices.tolist()
                    or progress["fold_result"] != fold_result
                    or not np.allclose(calculated, progress["score"], atol=1e-6, rtol=0)
                    or fold_result["parameter_count"] != trainable_parameter_count(model)
                    or fold_result["training_references"] != len(train_refs)
                    or len(fold_result["history"]) != 60):
                raise AssertionError("checkpoint prediction or fold record mismatch")
            score[family][indices] = calculated
            literal = calculated >= 0
            counts = {"accepted": int(literal.sum()),
                      "valid_accepted": int((literal & valid[indices]).sum()),
                      "invalid_accepted": int((literal & ~valid[indices]).sum())}
            if fold_result["literal_zero_threshold"] != counts:
                raise AssertionError("literal count mismatch")
            for key, count in counts.items():
                summaries[family]["literal_zero_threshold"][key] += count
            for fraction in FRACTIONS:
                _, counts = _matched(calculated, valid[indices], fraction)
                if fold_result["matched_coverage"][str(fraction)] != counts:
                    raise AssertionError("matched coverage count mismatch")
                for key, count in counts.items():
                    summaries[family]["matched_coverage"][str(fraction)][key] += count
        accepted = {family: _matched(score[family][indices], valid[indices], 0.4)[0]
                    for family in FAMILIES}
        for seed in np.unique(seeds[indices]):
            within = seeds[indices] == seed
            paired[str(seed)] = {family: {
                "accepted": int((accepted[family] & within).sum()),
                "invalid_accepted": int((accepted[family] & within & ~valid[indices]).sum()),
            } for family in FAMILIES}
    if any(not np.isfinite(values).all() for values in score.values()):
        raise AssertionError("missing held-out scores")
    for index, row in enumerate(rows):
        saved = saved_predictions[row["observable_key"]]
        if (saved["seed"] != row["seed"] or saved["valid"] != row["valid"]
                or saved["reference_key"] != row["reference_key"]):
            raise AssertionError("prediction provenance differs")
        if any(abs(saved["score"][family] - score[family][index]) > 1e-6
               for family in FAMILIES):
            raise AssertionError("prediction score differs")
    if summaries != report["summary"] or paired != report["paired_seed_40_percent"]:
        raise AssertionError("aggregate or paired result mismatch")
    print(json.dumps({"status": "PASS", "rows": len(rows), "fitted_references": len(fitted_cache),
                      "summary": summaries, "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
