#!/usr/bin/env python3
"""Freeze four old-cohort-only models before disjoint development labels exist."""

from __future__ import annotations

from collections import defaultdict
import gzip
import importlib.util
import json
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
from bces.models.decision_surface import CAP, GUARD, DecisionSurfaceNet, decision_surface_loss
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
EXPLORATORY = ROOT / "outputs/study_b/diverse_empty_teacher_distillation_dev_v1/report.json"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_transfer_models_dev_v1"
PLAN = ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_PLAN.md"
SOURCE_SHA256 = "a61c0717767f74f888526cd8e5ac119ee6ecf040421368eef6ad4d27ee88dd2e"
EXPLORATORY_SHA256 = "59a6ffab3c5ce35456b425a73a580cc1d41a92310bf8185f1317a328e286bd0e"
FAMILIES = ("surface_teacher", "scalar_ttl_teacher", "surface_no_teacher", "scalar_ttl_no_teacher")
EPOCHS = 60
BATCH_SIZE = 128
SEED = 20260925


def _rows(source_report):
    path = ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("empty_transfer_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_unique(source_report)


def _origins(rows):
    origin = {}
    for row in rows:
        if row["age_s"] == 0.0:
            key = row["reference_key"]
            if key in origin and origin[key] != row["valid"]:
                raise RuntimeError("conflicting origin label")
            origin[key] = row["valid"]
    if len(origin) != len({row["reference_key"] for row in rows}):
        raise RuntimeError("missing origin label")
    return np.asarray([origin[row["reference_key"]] for row in rows], dtype=np.float32)


def _teachers(rows, drift, valid):
    by_ref = defaultdict(list)
    for index, row in enumerate(rows):
        by_ref[row["reference_key"]].append(index)
    targets = {
        "surface": np.zeros((len(rows), 16), dtype=np.float32),
        "scalar_ttl": np.zeros((len(rows), 1), dtype=np.float32),
    }
    audit = []
    for ref, indices in sorted(by_ref.items()):
        z, y = drift[indices], valid[indices]
        surface = fit_zero_error_surface(z, y, NORMAL_CODEBOOK_V1, time_limit=1.0)
        ttl = fit_zero_error_ttl(z[:, 6], y)
        if not surface["abstain"] and not surface["optimal"]:
            raise RuntimeError("nonoptimal old-development teacher")
        if not surface["abstain"]:
            targets["surface"][indices] = np.minimum(
                CAP, np.asarray(surface["offsets"], dtype=np.float32) + GUARD)
        if not ttl["abstain"]:
            targets["scalar_ttl"][indices, 0] = min(CAP, ttl["threshold"] + GUARD)
        audit.append({
            "reference_key": ref, "training_rows": len(indices),
            "training_seed": int(rows[indices[0]]["seed"]),
            "surface_abstain": bool(surface["abstain"]),
            "surface_status": surface["status"],
            "surface_offsets": surface["offsets"],
            "ttl_abstain": bool(ttl["abstain"]),
            "ttl_threshold": ttl["threshold"],
        })
    path = OUTPUT / "old_development_teachers.json"
    path.write_text(json.dumps({"scope": "old_64_seed_development_only",
                                "references": audit}, sort_keys=True) + "\n", encoding="utf-8")
    return targets, sha256_file(path), len(by_ref)


def _train(family, features, drift, valid, origin, targets, policy_hash,
           teacher_sha256, mean, std):
    scalar = family.startswith("scalar_ttl")
    use_teacher = family.endswith("_teacher") and not family.endswith("no_teacher")
    reference_target = targets["scalar_ttl" if scalar else "surface"] if use_teacher else None
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = DecisionSurfaceNet(features.shape[1], scalar=scalar)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.Generator().manual_seed(SEED)
    x = torch.from_numpy(features)
    z = torch.from_numpy(drift)
    y = torch.from_numpy(valid.astype(np.float32))
    o = torch.from_numpy(origin)
    t = torch.from_numpy(reference_target) if reference_target is not None else None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(features), generator=scheduler)
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            offsets, logits = model(x[indices])
            loss = decision_surface_loss(
                offsets, logits, z[indices], y[indices], o[indices], scalar=scalar,
                teacher=t[indices] if t is not None else None)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch, "training_loss": total / len(features)})
    checkpoint = OUTPUT / f"{family}.pt"
    torch.save({
        "model_state": model.state_dict(), "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std), "family": family, "scalar": scalar,
        "use_teacher": use_teacher, "seed": SEED, "feature_count": 448,
        "policy_hash": policy_hash, "source_report_sha256": SOURCE_SHA256,
        "teacher_audit_sha256": teacher_sha256 if use_teacher else None,
        "training_rows": len(features),
    }, checkpoint)
    return {"checkpoint_sha256": sha256_file(checkpoint),
            "parameter_count": trainable_parameter_count(model), "history": history}


def main():
    if sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("old development labels changed")
    if sha256_file(EXPLORATORY) != EXPLORATORY_SHA256:
        raise RuntimeError("exploratory evidence changed")
    if OUTPUT.exists():
        raise RuntimeError("frozen-model output already exists")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    rows = _rows(source_report)
    x = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    drift = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    origin = _origins(rows)
    seeds = {row["seed"] for row in rows}
    if (x.shape != (2688, 448) or drift.shape != (2688, 7)
            or seeds != set(range(8701000, 8701064))):
        raise RuntimeError("old cohort feature/seed contract changed")
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
        "status": "FROZEN_OLD_DEVELOPMENT_MODELS_FOR_DISJOINT_DEV_TRANSFER_V1",
        "plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
        "feature_source_sha256": sha256_file(
            ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"),
        "model_source_sha256": sha256_file(ROOT / "bces/models/decision_surface.py"),
        "geometry_source_sha256": sha256_file(ROOT / "bces/evaluation/decision_geometry.py"),
        "source_report_sha256": SOURCE_SHA256,
        "exploratory_report_sha256": EXPLORATORY_SHA256,
        "train_seeds": sorted(seeds), "seed": SEED, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "families": list(FAMILIES),
        "policy_hash": policy_hash, "feature_count": 448,
        "new_cohort_accessed": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    targets, teacher_hash, n_refs = _teachers(rows, drift, valid)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 0.01)
    features = np.clip((x - mean) / std, -10, 10).astype(np.float32)
    results = {}
    for family in FAMILIES:
        results[family] = _train(family, features, drift, valid, origin,
                                 targets, policy_hash, teacher_hash, mean, std)
        print(json.dumps({"frozen_model": family,
                          "checkpoint_sha256": results[family]["checkpoint_sha256"]}),
              flush=True)
    report = {
        "status": protocol["status"], "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "teacher_audit_sha256": teacher_hash, "training_references": n_refs,
        "training_rows": len(rows), "models": results,
        "new_cohort_accessed": False,
        "independent_confirmation": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
