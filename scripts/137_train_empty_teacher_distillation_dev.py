#!/usr/bin/env python3
"""Exploratory equal-input fitted-geometry teachers on opened development seeds."""

from __future__ import annotations

from collections import defaultdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.decision_geometry import fit_zero_error_surface, fit_zero_error_ttl
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.models.decision_surface import (
    CAP, GUARD, DecisionSurfaceNet, decision_surface_loss, membership_score,
)
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
BASELINE = ROOT / "outputs/study_b/diverse_empty_surface_ttl_learning_dev_v1/report.json"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_teacher_distillation_dev_v1"
PLAN = ROOT / "docs/STUDY_B_EMPTY_TEACHER_DISTILLATION_DEV_V1_PLAN.md"
SOURCE_SHA256 = "a61c0717767f74f888526cd8e5ac119ee6ecf040421368eef6ad4d27ee88dd2e"
BASELINE_SHA256 = "3ef670ab3255865708ed024e70e3355d8d7ec156ed3624230e4b526eaee4ac85"
FAMILIES = ("surface", "scalar_ttl")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
EPOCHS = 60
BATCH_SIZE = 128


def _load_rows(source_report):
    path = ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("empty_teacher_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_unique(source_report)


def _origin_labels(rows):
    origins = {}
    for row in rows:
        if row["age_s"] == 0.0:
            key = row["reference_key"]
            if key in origins and origins[key] != row["valid"]:
                raise RuntimeError("conflicting origin validity")
            origins[key] = row["valid"]
    if len(origins) != len({row["reference_key"] for row in rows}):
        raise RuntimeError("missing reference origin")
    return np.asarray([origins[row["reference_key"]] for row in rows], dtype=np.float32)


def _accepted_at_fraction(scores, fraction):
    accepted = np.zeros(len(scores), dtype=bool)
    accepted[np.argsort(-scores, kind="stable")[:math.floor(fraction * len(scores))]] = True
    return accepted


def _fold_teachers(fold, rows, drift, valid, seeds):
    train = seeds % 8 != fold
    refs = defaultdict(list)
    for index in np.flatnonzero(train):
        refs[rows[index]["reference_key"]].append(int(index))
    teacher = {
        "surface": np.zeros((len(rows), 16), dtype=np.float32),
        "scalar_ttl": np.zeros((len(rows), 1), dtype=np.float32),
    }
    audit = []
    for ref_key, indices in sorted(refs.items()):
        z, y = drift[indices], valid[indices]
        surface = fit_zero_error_surface(z, y, NORMAL_CODEBOOK_V1, time_limit=1.0)
        ttl = fit_zero_error_ttl(z[:, 6], y)
        if not surface["abstain"] and not surface["optimal"]:
            raise RuntimeError("nonoptimal fitted surface teacher")
        if not surface["abstain"]:
            teacher["surface"][indices] = np.minimum(
                CAP, np.asarray(surface["offsets"], dtype=np.float32) + GUARD)
        if not ttl["abstain"]:
            teacher["scalar_ttl"][indices, 0] = min(CAP, ttl["threshold"] + GUARD)
        audit.append({
            "reference_key": ref_key,
            "training_seed": int(seeds[indices[0]]),
            "training_rows": len(indices),
            "surface_abstain": bool(surface["abstain"]),
            "surface_status": surface["status"],
            "surface_offsets": surface["offsets"],
            "ttl_abstain": bool(ttl["abstain"]),
            "ttl_threshold": ttl["threshold"],
        })
    if any(entry["training_seed"] % 8 == fold for entry in audit):
        raise RuntimeError("held-out reference leaked into teacher fitting")
    path = OUTPUT / f"teachers_fold{fold}.json"
    path.write_text(json.dumps({"fold": fold, "scope": "training_references_only",
                                "references": audit}, sort_keys=True) + "\n", encoding="utf-8")
    return teacher, sha256_file(path), len(refs)


def _train_one(family, fold, x, drift, valid, origin, seeds, teacher, policy_hash):
    test = seeds % 8 == fold
    train = ~test
    mean = x[train].mean(axis=0)
    std = np.maximum(x[train].std(axis=0), 0.01)
    normalized = np.clip((x - mean) / std, -10, 10).astype(np.float32)
    seed = 20260925 + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DecisionSurfaceNet(x.shape[1], scalar=(family == "scalar_ttl"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.Generator().manual_seed(seed)
    train_indices = np.flatnonzero(train)
    features = torch.from_numpy(normalized)
    z = torch.from_numpy(drift)
    labels = torch.from_numpy(valid.astype(np.float32))
    origin_labels = torch.from_numpy(origin)
    targets = torch.from_numpy(teacher[family])
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=scheduler).numpy()
        total = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = torch.from_numpy(train_indices[permutation[start:start + BATCH_SIZE]])
            optimizer.zero_grad(set_to_none=True)
            offsets, logits = model(features[indices])
            loss = decision_surface_loss(
                offsets, logits, z[indices], labels[indices],
                origin_labels[indices], scalar=model.scalar, teacher=targets[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch, "training_loss": total / len(train_indices)})
    model.eval()
    with torch.no_grad():
        test_indices = np.flatnonzero(test)
        offsets, logits = model(features[test_indices])
        scores = membership_score(
            offsets, logits, z[test_indices], scalar=model.scalar).numpy()
        offsets_np = offsets.numpy()
    if not np.isfinite(scores).all() or not np.isfinite(offsets_np).all():
        raise RuntimeError("nonfinite prediction")
    by_ref = defaultdict(list)
    for position, index in enumerate(test_indices):
        by_ref[ROW_REFS[index]].append(position)
    for positions in by_ref.values():
        if not np.allclose(offsets_np[positions], offsets_np[positions[0]], atol=1e-6, rtol=0):
            raise RuntimeError("sender offset changed across receiver states")
    checkpoint = OUTPUT / f"{family}_fold{fold}.pt"
    torch.save({
        "model_state": model.state_dict(), "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std), "family": family, "scalar": model.scalar,
        "fold": fold, "feature_count": x.shape[1], "policy_hash": policy_hash,
        "source_report_sha256": SOURCE_SHA256,
        "training_teacher_sha256": sha256_file(OUTPUT / f"teachers_fold{fold}.json"),
    }, checkpoint)
    return scores, test_indices, history, sha256_file(checkpoint), trainable_parameter_count(model), len(by_ref)


def main():
    global ROW_REFS
    if sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("development labels changed")
    if sha256_file(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("no-teacher comparison changed")
    if OUTPUT.exists():
        raise RuntimeError("teacher output exists; preserve the first run")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    rows = _load_rows(source_report)
    if len(rows) != 2688:
        raise RuntimeError("unexpected observable count")
    x = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    drift = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    origin = _origin_labels(rows)
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    ROW_REFS = [row["reference_key"] for row in rows]
    if x.shape != (2688, 448) or drift.shape != (2688, 7) or len(np.unique(seeds)) != 64:
        raise RuntimeError("input feature/seed contract changed")
    policy_hashes = set()
    for name in sorted(source_report["artifact_sha256"]):
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        policy_hashes.add(artifact["references"][0]["policy_hash"])
    if len(policy_hashes) != 1:
        raise RuntimeError("multiple policy versions")
    policy_hash = policy_hashes.pop()
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True)
    protocol = {
        "status": "EXPLORATORY_OPENED_DEVELOPMENT_TEACHER_TEST_V1",
        "selection_biased_after_error_inspection": True,
        "plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
        "source_report_sha256": SOURCE_SHA256, "baseline_report_sha256": BASELINE_SHA256,
        "feature_source_sha256": sha256_file(ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"),
        "model_source_sha256": sha256_file(ROOT / "bces/models/decision_surface.py"),
        "geometry_source_sha256": sha256_file(ROOT / "bces/evaluation/decision_geometry.py"),
        "policy_hash": policy_hash, "feature_count": 448,
        "fold_rule": "seed modulo 8", "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "families": list(FAMILIES), "teacher_scope": "training_references_only",
        "teacher_loss_weight": 0.2, "source_view_certified": False,
        "independent_confirmation": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    scores = {family: np.full(len(rows), np.nan, dtype=float) for family in FAMILIES}
    fold_results = {}
    teacher_hashes = {}
    for fold in range(8):
        teacher, teacher_hash, n_training_refs = _fold_teachers(fold, rows, drift, valid, seeds)
        teacher_hashes[f"teachers_fold{fold}.json"] = teacher_hash
        for family in FAMILIES:
            prediction, indices, history, checkpoint_hash, parameters, n_test_refs = _train_one(
                family, fold, x, drift, valid, origin, seeds, teacher, policy_hash)
            scores[family][indices] = prediction
            literal = prediction >= 0
            fold_results[f"{family}:{fold}"] = {
                "fold": fold, "family": family,
                "test_seed_count": len(np.unique(seeds[indices])), "test_rows": len(indices),
                "training_references": n_training_refs, "test_references": n_test_refs,
                "checkpoint_sha256": checkpoint_hash, "parameter_count": parameters,
                "history": history,
                "literal_zero_threshold": {
                    "accepted": int(literal.sum()),
                    "valid_accepted": int((literal & valid[indices]).sum()),
                    "invalid_accepted": int((literal & ~valid[indices]).sum()),
                },
                "matched_coverage": {
                    str(fraction): {
                        "accepted": int(accept.sum()),
                        "valid_accepted": int((accept & valid[indices]).sum()),
                        "invalid_accepted": int((accept & ~valid[indices]).sum()),
                    } for fraction in FRACTIONS
                    for accept in [_accepted_at_fraction(prediction, fraction)]
                },
            }
            progress = OUTPUT / f"{family}_fold{fold}.json"
            progress.write_text(json.dumps({
                "test_indices": indices.tolist(), "score": prediction.tolist(),
                "fold_result": fold_results[f"{family}:{fold}"],
            }, separators=(",", ":")) + "\n", encoding="utf-8")
            print(json.dumps({"completed_fold": fold, "family": family,
                              "training_references": n_training_refs}), flush=True)
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("missing out-of-fold scores")
    prediction_path = OUTPUT / "predictions.jsonl.gz"
    with gzip.open(prediction_path, "xt", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                "observable_key": row["observable_key"], "seed": row["seed"],
                "valid": row["valid"], "reference_key": row["reference_key"],
                "score": {family: float(scores[family][index]) for family in FAMILIES},
            }, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {}
    for family in FAMILIES:
        literal = scores[family] >= 0
        summary[family] = {
            "literal_zero_threshold": {
                "accepted": int(literal.sum()),
                "valid_accepted": int((literal & valid).sum()),
                "invalid_accepted": int((literal & ~valid).sum()),
            },
            "matched_coverage": {
                str(fraction): {
                    key: sum(fold_results[f"{family}:{fold}"]["matched_coverage"][str(fraction)][key]
                             for fold in range(8))
                    for key in ("accepted", "valid_accepted", "invalid_accepted")
                } for fraction in FRACTIONS
            },
        }
    paired_40 = {}
    for fold in range(8):
        indices = np.flatnonzero(seeds % 8 == fold)
        accepted = {family: _accepted_at_fraction(scores[family][indices], 0.4)
                    for family in FAMILIES}
        for seed in np.unique(seeds[indices]):
            within = seeds[indices] == seed
            paired_40[str(seed)] = {
                family: {
                    "accepted": int((accepted[family] & within).sum()),
                    "invalid_accepted": int((accepted[family] & within & ~valid[indices]).sum()),
                } for family in FAMILIES
            }
    report = {
        "status": protocol["status"], "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "teacher_sha256": teacher_hashes,
        "checkpoint_sha256": {
            f"{family}_fold{fold}.pt": fold_results[f"{family}:{fold}"]["checkpoint_sha256"]
            for fold in range(8) for family in FAMILIES
        },
        "fold_progress_sha256": {
            f"{family}_fold{fold}.json": sha256_file(OUTPUT / f"{family}_fold{fold}.json")
            for fold in range(8) for family in FAMILIES
        },
        "predictions_sha256": sha256_file(prediction_path),
        "fold_results": fold_results, "summary": summary,
        "paired_seed_40_percent": paired_40,
        "extension_bytes_only": {"surface": 48, "scalar_ttl": 2},
        "full_communication_accounting_complete": False,
        "source_view_certified": False, "risk_calibrated": False,
        "independent_confirmation": False,
        "selection_biased_exploratory": True,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
