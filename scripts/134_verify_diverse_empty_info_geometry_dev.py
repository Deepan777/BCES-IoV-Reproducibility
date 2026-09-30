#!/usr/bin/env python3
"""Independent raw-label, score-metric, and ideal-geometry replay."""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.decision_geometry import fit_zero_error_surface, fit_zero_error_ttl
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.utils.hashing import canonical_json_hash, sha256_file

SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
RESULT = ROOT / "outputs/study_b/diverse_empty_info_geometry_dev_v1"


def main():
    protocol = json.loads((RESULT / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((RESULT / "report.json").read_text(encoding="utf-8"))
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if (sha256_file(RESULT / "protocol.json") != report["protocol_sha256"]
            or sha256_file(SOURCE / "report.json") != protocol["source_report_sha256"]
            or sha256_file(ROOT / "scripts/133_diagnose_diverse_empty_info_geometry_dev.py") != protocol["script_sha256"]
            or sha256_file(ROOT / "docs/STUDY_B_DIVERSE_EMPTY_INFO_GEOMETRY_DEV_V1_PLAN.md") != protocol["plan_sha256"]):
        raise AssertionError("frozen protocol/source binding changed")
    for name, digest in report["artifacts_sha256"].items():
        if sha256_file(RESULT / name) != digest:
            raise AssertionError("analysis artifact changed: " + name)
    groups = defaultdict(list)
    for name, digest in source["artifact_sha256"].items():
        path = SOURCE / name
        if sha256_file(path) != digest:
            raise AssertionError("raw development artifact changed: " + name)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        reference = artifact["references"][0]
        ref_key = reference["frozen_input"]["feature_sha256"]
        for point in artifact["points"]:
            key = canonical_json_hash({
                "reference_feature_sha256": ref_key,
                "normalized_drift": point["normalized_drift"],
                "cache_age_s": point["cache_age_s"],
            })
            groups[key].append({"seed": artifact["seed"],
                                "valid": bool(point["valid"]),
                                "ref_key": ref_key,
                                "drift": point["normalized_drift"]})
    unique = {}
    for key, members in groups.items():
        if len({row["seed"] for row in members}) != 1 or len({row["valid"] for row in members}) != 1:
            raise AssertionError("observable leakage or conflict")
        unique[key] = members[0]
    if len(unique) != report["distinct_rows"]:
        raise AssertionError("unique observable count wrong")
    by_view = defaultdict(dict)
    with gzip.open(RESULT / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row["observable_key"]
            if key not in unique or key in by_view[row["view"]]:
                raise AssertionError("missing or duplicate prediction identity")
            if (row["seed"], row["valid"]) != (unique[key]["seed"], unique[key]["valid"]):
                raise AssertionError("prediction label/provenance mismatch")
            score = row["oof_valid_probability"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise AssertionError("invalid predicted probability")
            by_view[row["view"]][key] = score
    expected_views = {"age_only", "frozen_reference_plus_age",
                      "frozen_reference_plus_full_drift"}
    if set(by_view) != expected_views or any(set(scores) != set(unique) for scores in by_view.values()):
        raise AssertionError("prediction view grid incomplete")
    keys = sorted(unique)
    valid = np.array([unique[key]["valid"] for key in keys], dtype=bool)
    seeds = np.array([unique[key]["seed"] for key in keys], dtype=int)
    for view, scores in by_view.items():
        probability = np.array([scores[key] for key in keys])
        metrics = report["information"][view]
        for field, recomputed in (
            ("auroc", roc_auc_score(valid, probability)),
            ("average_precision", average_precision_score(valid, probability)),
            ("brier", brier_score_loss(valid, probability)),
        ):
            if abs(metrics[field] - recomputed) > 1e-12:
                raise AssertionError("prediction metric mismatch: " + view + ":" + field)
        rank = np.argsort(-probability, kind="stable")
        for fraction in (0.2, 0.4, 0.6, 0.8):
            accepted = np.zeros(len(valid), dtype=bool)
            accepted[rank[:math.floor(fraction * len(valid))]] = True
            recorded = metrics["coverage"][str(fraction)]
            if (recorded["accepted"] != int(accepted.sum())
                    or recorded["invalid_accepted"] != int((accepted & ~valid).sum())
                    or recorded["valid_accepted"] != int((accepted & valid).sum())):
                raise AssertionError("matched-coverage count mismatch")
            if fraction == 0.4:
                for seed in np.unique(seeds):
                    recorded_seed = metrics["per_seed_at_40_percent"][str(seed)]
                    if recorded_seed != {
                            "accepted": int((accepted & (seeds == seed)).sum()),
                            "invalid_accepted": int((accepted & ~valid & (seeds == seed)).sum())}:
                        raise AssertionError("seed-cluster count mismatch")
        for fold in metrics["folds"]:
            mask = seeds % 8 == fold["fold"]
            if (fold["validation_seed_count"] != len(np.unique(seeds[mask]))
                    or fold["validation_rows"] != int(mask.sum())
                    or fold["validation_valid"] != int(valid[mask].sum())
                    or fold["validation_invalid"] != int((~valid[mask]).sum())):
                raise AssertionError("fold composition mismatch")
    by_ref = defaultdict(list)
    for row in unique.values():
        by_ref[row["ref_key"]].append(row)
    geometry_rows = json.loads((RESULT / "geometry_references.json").read_text(encoding="utf-8"))
    recorded = {row["reference_key"]: row for row in geometry_rows}
    if set(recorded) != set(by_ref):
        raise AssertionError("geometry reference grid incomplete")
    ttl_total = surface_total = optimal = abstain = 0
    per_seed = defaultdict(lambda: {"ttl": 0, "surface": 0})
    for ref_key, members in by_ref.items():
        drift = np.asarray([row["drift"] for row in members], dtype=float)
        labels = np.asarray([row["valid"] for row in members], dtype=bool)
        ttl = fit_zero_error_ttl(drift[:, 6], labels)
        surface = fit_zero_error_surface(drift, labels, NORMAL_CODEBOOK_V1, time_limit=1.0)
        t = int(sum(ttl["accepted"]))
        s = int(sum(surface["accepted"]))
        row = recorded[ref_key]
        if (row["points"] != len(members) or row["valid"] != int(labels.sum())
                or row["ttl_accepted_valid"] != t
                or row["surface_accepted_valid"] != s
                or row["surface_optimal"] != bool(surface["optimal"])
                or row["surface_abstain"] != bool(surface["abstain"])):
            raise AssertionError("ideal geometry replay mismatch")
        ttl_total += t
        surface_total += s
        optimal += bool(surface["optimal"])
        abstain += bool(surface["abstain"])
        per_seed[members[0]["seed"]]["ttl"] += t
        per_seed[members[0]["seed"]]["surface"] += s
    geometry = report["geometry"]
    if (geometry["references"] != len(by_ref)
            or geometry["ttl_accepted_valid"] != ttl_total
            or geometry["surface_accepted_valid"] != surface_total
            or geometry["surface_optimal_references"] != optimal
            or geometry["surface_abstain_references"] != abstain):
        raise AssertionError("geometry totals mismatch")
    for seed, counts in per_seed.items():
        if geometry["per_seed"][str(seed)] != {
                **counts, "surface_minus_ttl": counts["surface"] - counts["ttl"]}:
            raise AssertionError("geometry seed total mismatch")
    print(json.dumps({"status": "PASS", "unique_observables": len(unique),
                      "information_auroc": {view: report["information"][view]["auroc"]
                                            for view in sorted(expected_views)},
                      "ttl_accepted_valid": ttl_total,
                      "surface_accepted_valid": surface_total,
                      "optimal_references": optimal,
                      "report_sha256": sha256_file(RESULT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
