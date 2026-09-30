#!/usr/bin/env python3
"""Fit and freeze drift scales using primary training intersections only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.oracle.observational_v2 import load_scene
from bces.oracle.scales import fit_drift_scales
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

REVIEW = ROOT / "data" / "manifests" / "v2xtraj_window_review_v1.json"
OUTPUT = ROOT / "data" / "manifests" / "drift_scales_v2.json"
RAW = ROOT / "data" / "raw" / "v2xtraj_primary"


def main() -> int:
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        expected = existing.pop("scales_sha256")
        if canonical_json_hash(existing) != expected:
            raise ValueError("existing immutable drift scales do not verify")
        if existing["source_file_sha256"] != sha256_file(REVIEW):
            raise ValueError("existing drift scales refer to a different source review")
        print(json.dumps({**existing, "scales_sha256": expected}, indent=2, sort_keys=True))
        return 0
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    expected = review.pop("review_sha256")
    if canonical_json_hash(review) != expected:
        raise ValueError("window review hash does not verify")
    records = [
        row for row in review["records"]
        if row.get("valid_windows", 0) > 0 and row["split"] == "train"
    ]
    trajectories = []
    for record in records:
        *_, states = load_scene(record, RAW)
        times = sorted(states)
        reference = states[times[0]]
        currents = tuple(
            states[reference.timestamp_ms + age_ms]
            for age_ms in range(200, 4001, 200)
            if reference.timestamp_ms + age_ms in states
        )
        trajectories.append((reference, currents))
    scales, fit = fit_drift_scales(trajectories)
    payload = {
        "schema_version": 2,
        "partition": "train",
        "test_partition_accessed": False,
        "scales": scales.as_tuple(),
        "scale_names": (
            "longitudinal_m", "lateral_m", "speed_mps", "heading_rad",
            "acceleration_mps2", "curvature_inv_m", "elapsed_s",
        ),
        "fit": fit,
        "source_review_sha256": expected,
        "source_file_sha256": sha256_file(REVIEW),
        "git": git_state(ROOT),
        "created_utc": utc_now(),
    }
    payload["scales_sha256"] = canonical_json_hash(payload)
    write_json_atomic(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
