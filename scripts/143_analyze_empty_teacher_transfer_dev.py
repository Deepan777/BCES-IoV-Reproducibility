#!/usr/bin/env python3
"""Apply old-cohort-frozen four models to the disjoint development cohort."""

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
from bces.utils.hashing import sha256_file

REGISTRATION = ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_REGISTRATION.json"
SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v2"
MODELS = ROOT / "outputs/study_b/diverse_empty_transfer_models_dev_v1"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_teacher_transfer_analysis_dev_v1"
PLAN = ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_PLAN.md"
FAMILIES = ("surface_teacher", "scalar_ttl_teacher", "surface_no_teacher", "scalar_ttl_no_teacher")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260925


def _rows(source_report):
    path = ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"
    spec = importlib.util.spec_from_file_location("empty_transfer_new_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE = SOURCE
    return module._load_unique(source_report)


def _accepted(score, fraction):
    accept = np.zeros(len(score), dtype=bool)
    accept[np.argsort(-score, kind="stable")[:math.floor(fraction * len(score))]] = True
    return accept


def _counts(accept, valid):
    return {"accepted": int(accept.sum()),
            "valid_accepted": int((accept & valid).sum()),
            "invalid_accepted": int((accept & ~valid).sum())}


def _inference(x, drift, model_report, model_protocol):
    scores = {}
    for family in FAMILIES:
        checkpoint_path = MODELS / f"{family}.pt"
        if sha256_file(checkpoint_path) != model_report["models"][family]["checkpoint_sha256"]:
            raise RuntimeError("frozen checkpoint changed: " + family)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        scalar = family.startswith("scalar_ttl")
        if (saved["family"] != family or saved["scalar"] != scalar
                or saved["source_report_sha256"] != model_protocol["source_report_sha256"]
                or saved["policy_hash"] != model_protocol["policy_hash"]):
            raise RuntimeError("frozen model identity changed")
        mean, std = saved["mean"].numpy(), saved["std"].numpy()
        normalized = np.clip((x - mean) / std, -10, 10).astype(np.float32)
        model = DecisionSurfaceNet(448, scalar=scalar)
        model.load_state_dict(saved["model_state"])
        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(x), 256):
                end = start + 256
                offsets, logits = model(torch.from_numpy(normalized[start:end]))
                score_parts.append(membership_score(
                    offsets, logits, torch.from_numpy(drift[start:end]),
                    scalar=scalar).numpy())
        score = np.concatenate(score_parts)
        if not np.isfinite(score).all() or len(score) != len(x):
            raise RuntimeError("nonfinite or incomplete frozen inference")
        scores[family] = score
    return scores


def _components(rows, source_report, registration):
    old_registration = json.loads((ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json")
                                  .read_text(encoding="utf-8"))
    thresholds = yaml.safe_load((ROOT / old_registration["config"]["validity"])
                                .read_text(encoding="utf-8"))
    raw = {}
    for name, digest in source_report["artifact_sha256"].items():
        path = SOURCE / name
        if sha256_file(path) != digest:
            raise RuntimeError("raw label artifact changed")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        raw[name] = {point["current_timestamp_ms"]: point for point in artifact["points"]}
    flags = []
    for row in rows:
        name, timestamp_text = row["slot_id"].rsplit(":", 1)
        point = raw[name][int(timestamp_text)]
        def exceeds(value, threshold):
            return value > threshold and not math.isclose(
                value, threshold, rel_tol=1e-12, abs_tol=1e-12)
        component = {
            "regret": exceeds(point["cost_regret"], thresholds["max_cost_regret"]),
            "cached_risk": exceeds(point["cached_risk"], thresholds["max_cached_risk"]),
            "trajectory_deviation": exceeds(
                point["trajectory_deviation_m"], thresholds["max_trajectory_deviation_m"]),
        }
        if row["valid"] == any(component.values()):
            raise RuntimeError("validity components disagree with saved label")
        flags.append(component)
    return flags


def _bootstrap(accepted, valid, seeds):
    groups = np.unique(seeds)
    if len(groups) != 64:
        raise RuntimeError("cluster count changed")
    surface = accepted["surface_teacher"]
    ttl = accepted["scalar_ttl_teacher"]
    per_seed_difference = np.asarray([
        int((surface & ~valid & (seeds == seed)).sum())
        - int((ttl & ~valid & (seeds == seed)).sum())
        for seed in groups], dtype=float)
    per_seed_invalid = np.asarray([
        int((surface & ~valid & (seeds == seed)).sum()) for seed in groups], dtype=float)
    per_seed_accepted = np.asarray([
        int((surface & (seeds == seed)).sum()) for seed in groups], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draw = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
    difference = per_seed_difference[draw].sum(axis=1)
    denominator = per_seed_accepted[draw].sum(axis=1)
    if (denominator <= 0).any():
        raise RuntimeError("empty bootstrap acceptance sample")
    rate = per_seed_invalid[draw].sum(axis=1) / denominator
    return {
        "unit": "seed_cluster",
        "replicates": BOOTSTRAP_REPLICATES, "rng_seed": BOOTSTRAP_SEED,
        "surface_minus_ttl_invalid_count": {
            "estimate": float(per_seed_difference.sum()),
            "percentile_95_interval": np.quantile(difference, [0.025, 0.975]).tolist(),
        },
        "surface_invalid_accept_rate": {
            "estimate": float(per_seed_invalid.sum() / per_seed_accepted.sum()),
            "percentile_95_interval": np.quantile(rate, [0.025, 0.975]).tolist(),
        },
    }


def main():
    if OUTPUT.exists():
        raise RuntimeError("analysis output exists; preserve first disjoint evaluation")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered transfer source changed: " + name)
    if sha256_file(PLAN) != registration["source_sha256"][str(PLAN.relative_to(ROOT)).replace("\\", "/")]:
        raise RuntimeError("analysis plan changed")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    source_protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    model_report = json.loads((MODELS / "report.json").read_text(encoding="utf-8"))
    model_protocol = json.loads((MODELS / "protocol.json").read_text(encoding="utf-8"))
    if (source_protocol["seeds"] != registration["new_seeds"]
            or not source_report["summary"]["engineering_screen_passed"]
            or source_report["summary"]["traces"] != 576
            or source_report["summary"]["points"] != 3456
            or source_report["summary"]["abstentions"] != 0
            or source_report["summary"]["lane_jumps"] != 0
            or source_report["summary"]["pose_jumps"] != 0
            or source_report["summary"]["mixed_validity_seed_clusters"] < 20
            or model_protocol["new_cohort_accessed"]
            or model_report["new_cohort_accessed"]):
        raise RuntimeError("frozen cohort engineering/model scope gate failed")
    rows = _rows(source_report)
    x = np.asarray([row["reference_vector"] for row in rows], dtype=np.float32)
    drift = np.asarray([row["drift"] for row in rows], dtype=np.float32)
    valid = np.asarray([row["valid"] for row in rows], dtype=bool)
    seeds = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    if x.shape[1] != 448 or drift.shape[1] != 7 or len(np.unique(seeds)) != 64:
        raise RuntimeError("new cohort observable shape changed")
    flags = _components(rows, source_report, registration)
    scores = _inference(x, drift, model_report, model_protocol)
    accepted = {family: {str(fraction): _accepted(scores[family], fraction)
                         for fraction in FRACTIONS}
                for family in FAMILIES}
    summary = {}
    for family in FAMILIES:
        literal = scores[family] >= 0
        summary[family] = {
            "literal_zero_threshold": _counts(literal, valid),
            "matched_coverage": {
                str(fraction): _counts(accepted[family][str(fraction)], valid)
                for fraction in FRACTIONS
            },
            "invalid_components_at_40_percent": {
                key: sum(bool(accepted[family]["0.4"][index] and flags[index][key])
                         for index in range(len(rows)))
                for key in ("regret", "cached_risk", "trajectory_deviation")
            },
        }
    per_seed_40 = {}
    for seed in np.unique(seeds):
        mask = seeds == seed
        per_seed_40[str(seed)] = {
            family: _counts(accepted[family]["0.4"] & mask, valid)
            for family in FAMILIES
        }
    bootstrap = _bootstrap({family: accepted[family]["0.4"] for family in FAMILIES},
                           valid, seeds)
    surface = summary["surface_teacher"]["matched_coverage"]["0.4"]["invalid_accepted"]
    ttl = summary["scalar_ttl_teacher"]["matched_coverage"]["0.4"]["invalid_accepted"]
    no_teacher = summary["surface_no_teacher"]["matched_coverage"]["0.4"]["invalid_accepted"]
    gate = {
        "engineering": True,
        "surface_better_than_teacher_ttl_40": surface < ttl,
        "surface_better_than_no_teacher_surface_40": surface < no_teacher,
        "cluster_difference_upper_below_zero":
            bootstrap["surface_minus_ttl_invalid_count"]["percentile_95_interval"][1] < 0,
        "surface_cluster_rate_upper_below_5_percent":
            bootstrap["surface_invalid_accept_rate"]["percentile_95_interval"][1] < 0.05,
    }
    gate["candidate_for_later_closed_loop_work"] = all(gate.values())
    OUTPUT.mkdir(parents=True)
    protocol = {
        "status": "DISJOINT_DEVELOPMENT_TRANSFER_ANALYSIS_V1",
        "plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
        "registration_sha256": sha256_file(REGISTRATION),
        "source_report_sha256": sha256_file(SOURCE / "report.json"),
        "model_report_sha256": sha256_file(MODELS / "report.json"),
        "feature_source_sha256": sha256_file(ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py"),
        "model_source_sha256": sha256_file(ROOT / "bces/models/decision_surface.py"),
        "families": list(FAMILIES), "fractions": list(FRACTIONS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "threshold_selected_using_new_labels": False,
        "independent_confirmation": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    prediction_path = OUTPUT / "predictions.jsonl.gz"
    with gzip.open(prediction_path, "xt", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                "observable_key": row["observable_key"], "seed": row["seed"],
                "reference_key": row["reference_key"], "valid": row["valid"],
                "invalid_components": flags[index],
                "score": {family: float(scores[family][index]) for family in FAMILIES},
                "accepted": {family: {str(fraction): bool(accepted[family][str(fraction)][index])
                                      for fraction in FRACTIONS} for family in FAMILIES},
            }, sort_keys=True, separators=(",", ":")) + "\n")
    report = {
        "status": protocol["status"], "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "predictions_sha256": sha256_file(prediction_path),
        "new_seed_clusters": len(np.unique(seeds)), "new_unique_observables": len(rows),
        "label_balance": {"valid": int(valid.sum()), "invalid": int((~valid).sum())},
        "summary": summary, "per_seed_40_percent": per_seed_40,
        "cluster_bootstrap_40_percent": bootstrap, "advancement_gate": gate,
        "synthetic_ideal_view": True, "full_communication_accounting_complete": False,
        "risk_calibrated": False, "independent_confirmation": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "cluster_bootstrap_40_percent": bootstrap,
                      "advancement_gate": gate,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
