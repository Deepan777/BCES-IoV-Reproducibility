#!/usr/bin/env python3
"""Development-only representability test for a union of two expiry surfaces.

Each reference is fitted to its *own* labels, so this is an oracle geometry
diagnostic, never a deployable predictor or independent validation result.
The curvature-sign partition is specified in code, not chosen by outcomes.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.decision_geometry import fit_zero_error_surface
from bces.evaluation.study_b import load_development
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.utils.hashing import sha256_file

DATA = ROOT / "outputs/study_b/controlled_development_v2"
OUTPUT = ROOT / "outputs/study_b/two_region_geometry_development_v1"
SCENARIOS_PER_PARTITION = 8
CURVATURE_INDEX = 4
TIME_LIMIT_S = 1.0


def _fit(drift: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    result = fit_zero_error_surface(drift, valid, NORMAL_CODEBOOK_V1,
                                    time_limit=TIME_LIMIT_S)
    if result["abstain"]:
        return np.zeros(len(drift), dtype=bool), result
    offsets = np.asarray(result["offsets"], dtype=float)
    membership = (drift @ np.asarray(NORMAL_CODEBOOK_V1, dtype=float).T
                  <= offsets + 1e-10).all(axis=1)
    if np.any(membership & ~valid):
        raise AssertionError("sample-invalid point accepted")
    return membership, result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    refs, points, manifest = load_development(DATA)
    if manifest.get("original_test_accessed") is not False:
        raise PermissionError("test access prohibited")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        grouped[point["reference_id"]].append(point)
    rows = []
    for partition in ("train", "validation"):
        # Family is stored in each scene, not in reference IDs. Select it from
        # the immutable scenario artifacts without opening any holdout data.
        family_scenarios = []
        for name in sorted(manifest["artifact_sha256"]):
            if not name.startswith(partition + "_"):
                continue
            with gzip.open(DATA / name, "rt", encoding="utf-8") as handle:
                scene = json.load(handle)
            if scene["family"] == "straight_lead_braking":
                family_scenarios.append(str(scene["scenario_id"]))
        selected = sorted(family_scenarios)[:SCENARIOS_PER_PARTITION]
        if len(selected) != SCENARIOS_PER_PARTITION:
            raise RuntimeError("insufficient straight-braking development scenarios")
        for scenario in selected:
            key = next(key for key, ref in refs.items()
                       if str(ref["scenario_id"]) == scenario and ref["split"] == partition
                       and ref["behavior"] == "keep")
            items = grouped[key]
            drift = np.asarray([item["normalized_drift"] for item in items], dtype=float)
            valid = np.asarray([item["valid"] for item in items], dtype=bool)
            one, one_result = _fit(drift, valid)
            regions = []
            fit_status = []
            for sign in (-1, 1):
                desired = valid & ((drift[:, CURVATURE_INDEX] < 0) if sign == -1
                                   else (drift[:, CURVATURE_INDEX] >= 0))
                subset = desired | ~valid
                if not np.any(desired):
                    regions.append(np.zeros(len(drift), dtype=bool))
                    fit_status.append("no_valid_points_in_partition")
                    continue
                _, fitted = _fit(drift[subset], valid[subset])
                if fitted["abstain"]:
                    regions.append(np.zeros(len(drift), dtype=bool))
                else:
                    offsets = np.asarray(fitted["offsets"], dtype=float)
                    regions.append((drift @ np.asarray(NORMAL_CODEBOOK_V1, dtype=float).T
                                    <= offsets + 1e-10).all(axis=1))
                fit_status.append(fitted["status"])
            two = regions[0] | regions[1]
            if np.any(two & ~valid):
                raise AssertionError("union accepted a sample-invalid point")
            rows.append({"split": partition, "scenario_id": scenario,
                         "reference_id": key, "points": len(items),
                         "valid_points": int(valid.sum()),
                         "single_valid_accepted": int(one.sum()),
                         "two_region_valid_accepted": int(two.sum()),
                         "single_status": one_result["status"],
                         "two_region_status": fit_status,
                         "single_optimal": bool(one_result["optimal"]),
                         "not_predictive": True})
    summary = {}
    for partition in ("train", "validation"):
        group = [row for row in rows if row["split"] == partition]
        summary[partition] = {"references": len(group),
                              "valid_points": sum(row["valid_points"] for row in group),
                              "single_valid_accepted": sum(row["single_valid_accepted"] for row in group),
                              "two_region_valid_accepted": sum(row["two_region_valid_accepted"] for row in group),
                              "two_region_better_references": sum(row["two_region_valid_accepted"] > row["single_valid_accepted"] for row in group),
                              "two_region_worse_references": sum(row["two_region_valid_accepted"] < row["single_valid_accepted"] for row in group)}
    report = {"status": "DEVELOPMENT_ORACLE_GEOMETRY_ONLY",
              "source_manifest_sha256": sha256_file(DATA / "run_manifest.json"),
              "script_sha256": sha256_file(Path(__file__)),
              "partition_rule": "curvature drift <0 versus >=0; both regions exclude every sampled invalid point",
              "selection": "first 8 sorted straight-braking scenarios in train and validation; keep reference",
              "fit_time_limit_s": TIME_LIMIT_S, "summary": summary, "rows": rows,
              "confirmation_accessed": False, "deployable_policy": False,
              "claim": "finite-sample oracle representability only; no generalization or safety guarantee"}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
