#!/usr/bin/env python3
"""Outcome-blind integrity, AV-role, and mobility audit for UrbanIng labels."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import write_json_atomic

MANIFEST = ROOT / "data/manifests/urbaning_labels_v1.json"
PREFLIGHT = ROOT / "data/manifests/urbaning_preflight_v1.json"
OUT = ROOT / "outputs/study_b/urbaning_schema_audit_v1.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    receipt = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if receipt["status"] != "LABELS_ACQUIRED_NO_BCES_OUTCOME" or receipt["file_count"] != 34:
        raise RuntimeError("incomplete label receipt")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    lookup = {r["original_filename"]: r for r in preflight["selected_records"]}
    mapping_row = lookup["labels_av_track_ids.json"]
    mapping_path = ROOT / mapping_row["relative_path"]
    if sha256_file(mapping_path) != mapping_row["sha256"]:
        raise RuntimeError("AV-ID mapping changed")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    sequences = []
    statuses = Counter()
    for row in receipt["records"]:
        path = ROOT / row["relative_path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"provider-verified label changed: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem
        ids = mapping.get(name)
        problems = []
        if not isinstance(ids, dict) or not ids:
            problems.append("av_track_mapping_missing")
            ids = {}
        tracks = {}
        for tr in data.get("tracks", []):
            tid = tr.get("track_id")
            if tid in tracks:
                problems.append("duplicate_track_id")
            tracks[tid] = tr
            n = len(tr.get("timestamps", []))
            if n < 1 or any(len(tr.get(k, [])) != n for k in ("positions", "orientations")):
                problems.append("track_array_length_mismatch")
                continue
            if len(tr.get("dimensions", [])) not in (1, n):
                problems.append("dimension_array_length_mismatch")
            ts = tr["timestamps"]
            if any(not (0.08 <= b - a <= 0.12) for a, b in zip(ts, ts[1:])):
                problems.append("non_10hz_track")
            if any(not all(math.isfinite(float(v)) for v in xyz[:2]) for xyz in tr["positions"]):
                problems.append("invalid_position")
            if any(not math.isfinite(float(v)) for v in tr["orientations"]):
                problems.append("invalid_orientation")
        receivers = []
        for role, tid in sorted(ids.items()):
            tr = tracks.get(tid)
            if tr is None:
                receivers.append({"role": role, "track_id": tid, "status": "missing_av_track"})
                statuses["missing_av_track"] += 1
                continue
            xy = tr["positions"]
            displacement = math.hypot(float(xy[-1][0]) - float(xy[0][0]), float(xy[-1][1]) - float(xy[0][1]))
            cumulative = sum(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])) for a, b in zip(xy, xy[1:]))
            status = "mobile_eligible" if displacement >= 1.0 else "stationary_abstention"
            statuses[status] += 1
            receivers.append({"role": role, "track_id": tid, "status": status,
                              "samples": len(xy), "displacement_m": displacement,
                              "cumulative_motion_m": cumulative,
                              "first_timestamp": tr["timestamps"][0], "last_timestamp": tr["timestamps"][-1]})
        sequences.append({"sequence": name, "track_count": len(tracks), "problems": sorted(set(problems)), "receivers": receivers})
    result = {"status": "SCHEMA_AND_MOBILITY_ONLY_NO_BCES_OUTCOMES", "manifest_sha256": sha256_file(MANIFEST), "preflight_sha256": sha256_file(PREFLIGHT), "source_sha256": sha256_file(Path(__file__)), "sequence_count": len(sequences), "receiver_status_counts": dict(statuses), "sequences_with_problems": sum(bool(x["problems"]) for x in sequences), "sequences": sequences}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT, result)
    print(json.dumps({k: result[k] for k in ("sequence_count", "receiver_status_counts", "sequences_with_problems")}, indent=2))


if __name__ == "__main__":
    main()
