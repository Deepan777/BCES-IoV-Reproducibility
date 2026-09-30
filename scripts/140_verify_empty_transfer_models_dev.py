#!/usr/bin/env python3
"""Verify the old-cohort-only four-model freeze before disjoint labels exist."""

from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.decision_geometry import fit_zero_error_surface, fit_zero_error_ttl
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.models.decision_surface import DecisionSurfaceNet
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_transfer_models_dev_v1"
FAMILIES = ("surface_teacher", "scalar_ttl_teacher", "surface_no_teacher", "scalar_ttl_no_teacher")


def _rows(source):
    path = ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("empty_frozen_model_verify_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_unique(source)


def main():
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if sha256_file(OUTPUT / "protocol.json") != report["protocol_sha256"]:
        raise AssertionError("protocol digest mismatch")
    bindings = {
        "plan_sha256": ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_PLAN.md",
        "script_sha256": ROOT / "scripts/139_fit_empty_teacher_transfer_models_dev.py",
        "feature_source_sha256": ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py",
        "model_source_sha256": ROOT / "bces/models/decision_surface.py",
        "geometry_source_sha256": ROOT / "bces/evaluation/decision_geometry.py",
        "source_report_sha256": SOURCE / "report.json",
        "exploratory_report_sha256": ROOT / "outputs/study_b/diverse_empty_teacher_distillation_dev_v1/report.json",
    }
    if any(protocol[key] != sha256_file(path) for key, path in bindings.items()):
        raise AssertionError("frozen model source/plan changed")
    if (protocol["new_cohort_accessed"] or report["new_cohort_accessed"]
            or protocol["families"] != list(FAMILIES)
            or protocol["train_seeds"] != list(range(8701000, 8701064))):
        raise AssertionError("training cohort or scope changed")
    rows = _rows(source)
    x = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    z = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    y = np.asarray([row["valid"] for row in rows], dtype=bool)
    if x.shape != (2688, 448) or z.shape != (2688, 7):
        raise AssertionError("training feature shape changed")
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 0.01)
    audit_path = OUTPUT / "old_development_teachers.json"
    if sha256_file(audit_path) != report["teacher_audit_sha256"]:
        raise AssertionError("teacher audit digest mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["scope"] != "old_64_seed_development_only":
        raise AssertionError("teacher scope changed")
    by_ref = defaultdict(list)
    for index, row in enumerate(rows):
        by_ref[row["reference_key"]].append(index)
    entries = audit["references"]
    if {entry["reference_key"] for entry in entries} != set(by_ref) or len(entries) != len(by_ref):
        raise AssertionError("teacher reference set differs")
    for entry in entries:
        members = by_ref[entry["reference_key"]]
        if (entry["training_rows"] != len(members)
                or entry["training_seed"] != int(rows[members[0]]["seed"])):
            raise AssertionError("teacher reference provenance differs")
        surface = fit_zero_error_surface(z[members], y[members], NORMAL_CODEBOOK_V1,
                                         time_limit=1.0)
        ttl = fit_zero_error_ttl(z[members, 6], y[members])
        if (entry["surface_abstain"] != surface["abstain"]
                or entry["surface_status"] != surface["status"]
                or entry["ttl_abstain"] != ttl["abstain"]
                or entry["ttl_threshold"] != ttl["threshold"]):
            raise AssertionError("teacher fit differs")
        if surface["offsets"] is None:
            if entry["surface_offsets"] is not None:
                raise AssertionError("abstaining teacher has offsets")
        elif not np.allclose(entry["surface_offsets"], surface["offsets"], atol=1e-8, rtol=0):
            raise AssertionError("teacher offsets differ")
    if report["training_references"] != len(by_ref) or report["training_rows"] != len(rows):
        raise AssertionError("report training counts differ")
    for family in FAMILIES:
        path = OUTPUT / f"{family}.pt"
        if sha256_file(path) != report["models"][family]["checkpoint_sha256"]:
            raise AssertionError("frozen checkpoint digest mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        scalar = family.startswith("scalar_ttl")
        use_teacher = not family.endswith("no_teacher")
        if (checkpoint["family"] != family or checkpoint["scalar"] != scalar
                or checkpoint["use_teacher"] != use_teacher
                or checkpoint["seed"] != 20260925
                or checkpoint["feature_count"] != 448
                or checkpoint["training_rows"] != len(rows)
                or checkpoint["policy_hash"] != protocol["policy_hash"]
                or checkpoint["source_report_sha256"] != protocol["source_report_sha256"]
                or checkpoint["teacher_audit_sha256"] != (
                    report["teacher_audit_sha256"] if use_teacher else None)
                or not np.array_equal(checkpoint["mean"].numpy(), mean)
                or not np.array_equal(checkpoint["std"].numpy(), std)):
            raise AssertionError("checkpoint training contract differs")
        model = DecisionSurfaceNet(448, scalar=scalar)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        if (trainable_parameter_count(model) != report["models"][family]["parameter_count"]
                or len(report["models"][family]["history"]) != 60):
            raise AssertionError("model parameter/history mismatch")
        with torch.no_grad():
            normalized = np.clip((x[:20] - mean) / std, -10, 10).astype(np.float32)
            offsets, logits = model(torch.from_numpy(normalized))
        if (not torch.isfinite(offsets).all() or not torch.isfinite(logits).all()
                or offsets.shape != (20, 1 if scalar else 16)):
            raise AssertionError("checkpoint cannot infer valid offsets")
    print(json.dumps({"status": "PASS", "training_rows": len(rows),
                      "training_references": len(by_ref),
                      "models": list(FAMILIES),
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
