#!/usr/bin/env python3
"""Equal-input, seed-disjoint SurfaceNet and learned scalar-TTL development fit."""

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

from bces.models.decision_surface import (
    DecisionSurfaceNet, decision_surface_loss, membership_score,
)
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
DIAGNOSTIC = ROOT / "outputs/study_b/diverse_empty_info_geometry_dev_v1"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_surface_ttl_learning_dev_v1"
SOURCE_SHA256 = "a61c0717767f74f888526cd8e5ac119ee6ecf040421368eef6ad4d27ee88dd2e"
DIAGNOSTIC_SHA256 = "fdb5462d44c098c105a665c1244a7fa277036524127e6dfee7c1fc48b52e41d0"
PLAN = ROOT / "docs/STUDY_B_DIVERSE_EMPTY_SURFACE_TTL_LEARNING_DEV_V1_PLAN.md"
FAMILIES = ("surface", "scalar_ttl")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
EPOCHS = 60
BATCH_SIZE = 128


def _loader():
    source = ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("empty_information_source", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_unique


def _origin_labels(rows):
    origins = {}
    for row in rows:
        if row["age_s"] == 0.0:
            key = row["reference_key"]
            if key in origins and origins[key] != row["valid"]:
                raise RuntimeError("conflicting reference-origin validity")
            origins[key] = row["valid"]
    if len(origins) != len({row["reference_key"] for row in rows}):
        raise RuntimeError("reference missing a zero-age origin")
    return np.array([origins[row["reference_key"]] for row in rows],
                    dtype=np.float32)


def _train_one(family, fold, x, drift, valid, origin, seeds):
    test = seeds % 8 == fold
    train = ~test
    if not test.any() or not train.any():
        raise RuntimeError("empty fixed seed fold")
    mean = x[train].mean(axis=0)
    std = np.maximum(x[train].std(axis=0), 0.01)
    normalized = np.clip((x - mean) / std, -10, 10).astype(np.float32)
    seed = 20260925 + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DecisionSurfaceNet(x.shape[1], scalar=(family == "scalar_ttl"))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.Generator().manual_seed(seed)
    train_indices = np.flatnonzero(train)
    features = torch.from_numpy(normalized)
    z = torch.from_numpy(drift)
    labels = torch.from_numpy(valid.astype(np.float32))
    origin_labels = torch.from_numpy(origin)
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
                origin_labels[indices], scalar=model.scalar)
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
        score = membership_score(
            offsets, logits, z[test_indices], scalar=model.scalar).numpy()
        offsets_np = offsets.numpy()
    if not np.isfinite(score).all() or not np.isfinite(offsets_np).all():
        raise RuntimeError("nonfinite receiver score or sender offsets")
    references = defaultdict(list)
    for index, reference_key in zip(test_indices, (rows_ref[index] for index in test_indices)):
        references[reference_key].append(index)
    for members in references.values():
        locations = [int(np.where(test_indices == index)[0][0]) for index in members]
        if not np.allclose(offsets_np[locations], offsets_np[locations[0]], atol=1e-6, rtol=0):
            raise RuntimeError("sender offsets changed with receiver drift")
    checkpoint = OUTPUT / f"{family}_fold{fold}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "family": family,
        "scalar": model.scalar,
        "fold": fold,
        "feature_count": x.shape[1],
        "policy_hash": POLICY_HASH,
        "source_report_sha256": SOURCE_SHA256,
        "diagnostic_report_sha256": DIAGNOSTIC_SHA256,
    }, checkpoint)
    return {
        "score": score,
        "test_indices": test_indices,
        "history": history,
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": trainable_parameter_count(model),
        "invariant_reference_count": len(references),
    }


def _accepted_at_fraction(score, fraction):
    accepted = np.zeros(len(score), dtype=bool)
    order = np.argsort(-score, kind="stable")
    accepted[order[:math.floor(fraction * len(score))]] = True
    return accepted


def main():
    global rows_ref, POLICY_HASH
    if sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("source label report changed")
    if sha256_file(DIAGNOSTIC / "report.json") != DIAGNOSTIC_SHA256:
        raise RuntimeError("information/geometry diagnostic changed")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((DIAGNOSTIC / "report.json").read_text(encoding="utf-8"))
    if not source_report["summary"]["engineering_screen_passed"]:
        raise RuntimeError("source label gate did not pass")
    if (diagnostic["information"]["frozen_reference_plus_full_drift"]["auroc"] < 0.8
            or diagnostic["geometry"]["surface_accepted_valid"] <=
            diagnostic["geometry"]["ttl_accepted_valid"]
            or diagnostic["geometry"]["surface_optimal_references"] != 192):
        raise RuntimeError("pre-learning diagnostics do not justify model fit")
    rows = _loader()(source_report)
    if len(rows) != 2688:
        raise RuntimeError("distinct row count changed")
    policy_hashes = set()
    for name in sorted(source_report["artifact_sha256"]):
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        policy_hashes.add(artifact["references"][0]["policy_hash"])
    if len(policy_hashes) != 1:
        raise RuntimeError("multiple candidate policy hashes")
    POLICY_HASH = policy_hashes.pop()
    rows_ref = [row["reference_key"] for row in rows]
    x = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    drift = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    origin = _origin_labels(rows)
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    if x.shape != (2688, 448) or len(np.unique(seeds)) != 64:
        raise RuntimeError("frozen feature/seed contract changed")
    torch.set_num_threads(2)
    protocol = {
        "status": "EQUAL_INPUT_SURFACE_TTL_DEVELOPMENT_ONLY_V1",
        "plan_sha256": sha256_file(PLAN),
        "script_sha256": sha256_file(Path(__file__)),
        "source_report_sha256": SOURCE_SHA256,
        "diagnostic_report_sha256": DIAGNOSTIC_SHA256,
        "feature_source_sha256": sha256_file(
            ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"),
        "decision_surface_source_sha256": sha256_file(
            ROOT / "bces/models/decision_surface.py"),
        "feature_count": x.shape[1],
        "seed_folds": "seed modulo 8",
        "families": list(FAMILIES),
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "policy_hash": POLICY_HASH,
        "source_view_certified": False,
        "independent_confirmation": False,
    }
    if OUTPUT.exists():
        saved_protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
        if saved_protocol != protocol or (OUTPUT / "report.json").exists():
            raise RuntimeError("cannot resume changed or completed learning run")
    else:
        OUTPUT.mkdir(parents=True)
        (OUTPUT / "protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    scores = {family: np.full(len(rows), np.nan, dtype=float) for family in FAMILIES}
    fold_results = {}
    for fold in range(8):
        for family in FAMILIES:
            progress_path = OUTPUT / f"{family}_fold{fold}.json"
            if progress_path.exists():
                saved = json.loads(progress_path.read_text(encoding="utf-8"))
                indices = np.flatnonzero(seeds % 8 == fold)
                if saved["test_indices"] != indices.tolist():
                    raise RuntimeError("resumed fold identity changed")
                checkpoint_path = OUTPUT / f"{family}_fold{fold}.pt"
                if sha256_file(checkpoint_path) != saved["fold_result"]["checkpoint_sha256"]:
                    raise RuntimeError("resumed checkpoint changed")
                scores[family][indices] = np.asarray(saved["score"], dtype=float)
                fold_results[f"{family}:{fold}"] = saved["fold_result"]
                print(json.dumps({"resumed_fold": fold, "family": family}), flush=True)
                continue
            result = _train_one(family, fold, x, drift, valid, origin, seeds)
            scores[family][result["test_indices"]] = result["score"]
            indices = result["test_indices"]
            literal = result["score"] >= 0
            coverage = {}
            for fraction in FRACTIONS:
                accepted = _accepted_at_fraction(result["score"], fraction)
                coverage[str(fraction)] = {
                    "accepted": int(accepted.sum()),
                    "valid_accepted": int((accepted & valid[indices]).sum()),
                    "invalid_accepted": int((accepted & ~valid[indices]).sum()),
                }
            fold_results[f"{family}:{fold}"] = {
                "fold": fold, "family": family,
                "test_seed_count": len(np.unique(seeds[indices])),
                "test_rows": len(indices),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "parameter_count": result["parameter_count"],
                "invariant_reference_count": result["invariant_reference_count"],
                "history": result["history"],
                "literal_zero_threshold": {
                    "accepted": int(literal.sum()),
                    "valid_accepted": int((literal & valid[indices]).sum()),
                    "invalid_accepted": int((literal & ~valid[indices]).sum()),
                },
                "matched_coverage": coverage,
            }
            progress_path.write_text(json.dumps({
                "test_indices": indices.tolist(),
                "score": result["score"].tolist(),
                "fold_result": fold_results[f"{family}:{fold}"],
            }, separators=(",", ":")) + "\n", encoding="utf-8")
            print(json.dumps({"completed_fold": fold, "family": family,
                              "literal_accepts": int(literal.sum())}), flush=True)
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise RuntimeError("missing out-of-fold receiver scores")
    predictions = [{
        "observable_key": row["observable_key"],
        "seed": row["seed"], "valid": row["valid"],
        "reference_key": row["reference_key"],
        "score": {family: float(scores[family][index]) for family in FAMILIES},
    } for index, row in enumerate(rows)]
    with gzip.open(OUTPUT / "predictions.jsonl.gz", "xt", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True,
                                    separators=(",", ":")) + "\n")
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
                    "accepted": sum(fold_results[f"{family}:{fold}"]["matched_coverage"][str(fraction)]["accepted"]
                                    for fold in range(8)),
                    "valid_accepted": sum(fold_results[f"{family}:{fold}"]["matched_coverage"][str(fraction)]["valid_accepted"]
                                          for fold in range(8)),
                    "invalid_accepted": sum(fold_results[f"{family}:{fold}"]["matched_coverage"][str(fraction)]["invalid_accepted"]
                                            for fold in range(8)),
                } for fraction in FRACTIONS
            },
        }
    paired_40 = {}
    for fold in range(8):
        fold_mask = seeds % 8 == fold
        indices = np.flatnonzero(fold_mask)
        accepted = {
            family: _accepted_at_fraction(scores[family][indices], 0.4)
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
    report = {
        "status": protocol["status"],
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "checkpoint_sha256": {
            f"{family}_fold{fold}.pt": fold_results[f"{family}:{fold}"]["checkpoint_sha256"]
            for fold in range(8) for family in FAMILIES
        },
        "fold_progress_sha256": {
            f"{family}_fold{fold}.json": sha256_file(OUTPUT / f"{family}_fold{fold}.json")
            for fold in range(8) for family in FAMILIES
        },
        "predictions_sha256": sha256_file(OUTPUT / "predictions.jsonl.gz"),
        "fold_results": fold_results,
        "summary": summary,
        "paired_seed_40_percent": paired_40,
        "extension_bytes_only": {"surface": 48, "scalar_ttl": 2},
        "full_communication_accounting_complete": False,
        "source_view_certified": False,
        "risk_calibrated": False,
        "independent_confirmation": False,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
