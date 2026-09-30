#!/usr/bin/env python3
"""Integrity and schema-only AV2 motion preflight; no policy scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file


MANIFEST = ROOT / "data/manifests/av2_motion_schema_preflight_v1.json"
DATA = ROOT / "data/external/av2_motion_schema_v1"
OUTPUT = ROOT / "outputs/study_b/av2_motion_schema_preflight_v1"
PREFIX = "datasets/av2/motion-forecasting/val/"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {item["key"]: item for item in manifest["files"]}
    if len(expected) != 6 or sum(item["bytes"] for item in expected.values()) != manifest["expected_total_bytes"]:
        raise ValueError("unexpected fixed acquisition manifest")

    file_audit = []
    for key, item in expected.items():
        if not key.startswith(PREFIX) or ".." in key:
            raise ValueError("unsafe manifest key")
        path = DATA / key[len(PREFIX):]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError("missing or wrong-sized source: " + key)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != item["etag"]:
            raise ValueError("source ETag mismatch: " + key)
        file_audit.append({"key": key, "bytes": path.stat().st_size,
                           "source_etag_md5": digest,
                           "sha256": sha256_file(path)})

    scenes = []
    for scene_dir in sorted(path for path in DATA.iterdir() if path.is_dir()):
        track_path = scene_dir / f"scenario_{scene_dir.name}.parquet"
        map_path = scene_dir / f"log_map_archive_{scene_dir.name}.json"
        table = pq.read_table(track_path)
        frame = table.to_pandas()
        map_data = json.loads(map_path.read_text(encoding="utf-8"))
        required = {"observed", "track_id", "object_type", "object_category",
                    "timestep", "position_x", "position_y", "heading",
                    "velocity_x", "velocity_y", "scenario_id"}
        if not required <= set(frame.columns):
            raise ValueError("missing required motion columns")
        if set(frame["scenario_id"]) != {scene_dir.name}:
            raise ValueError("scenario identity mismatch")
        lanes = map_data.get("lane_segments", {})
        lane_ids = {int(key) for key in lanes}
        invalid_links = sum(int(other) not in lane_ids
                            for lane in lanes.values()
                            for other in (lane.get("successors") or []) + (lane.get("predecessors") or []))
        av = frame[frame["track_id"] == "AV"]
        if av.empty:
            raise ValueError("no AV track")
        timesteps = sorted(int(value) for value in av["timestep"].unique())
        observed = sorted(int(value) for value in av.loc[av["observed"], "timestep"].unique())
        scene = {
            "scene_id": scene_dir.name,
            "rows": len(frame),
            "columns": list(frame.columns),
            "track_count": int(frame["track_id"].nunique()),
            "av_timesteps": len(timesteps),
            "av_first_timestep": timesteps[0],
            "av_last_timestep": timesteps[-1],
            "av_contiguous": timesteps == list(range(timesteps[0], timesteps[-1] + 1)),
            "av_observed_timesteps": len(observed),
            "av_observed_first": observed[0] if observed else None,
            "av_observed_last": observed[-1] if observed else None,
            "object_types": {str(key): int(value) for key, value in
                             frame.drop_duplicates("track_id")["object_type"].value_counts().items()},
            "motion_finite_fraction": float(pd.DataFrame({
                key: pd.to_numeric(frame[key], errors="coerce")
                for key in ("position_x", "position_y", "heading", "velocity_x", "velocity_y")
            }).notna().all(axis=1).mean()),
            "map_lane_segments": len(lanes),
            "map_successor_edges": sum(len(lane.get("successors") or []) for lane in lanes.values()),
            "map_invalid_predecessor_successor_links": invalid_links,
            "view_specific_visibility_present": False,
            "object_dimensions_present": False,
        }
        scenes.append(scene)

    report = {
        "status": "SCHEMA_ONLY_NO_POLICY_OUTCOMES",
        "manifest_sha256": sha256_file(MANIFEST),
        "source_sha256": sha256_file(Path(__file__)),
        "file_audit": file_audit,
        "scenes": scenes,
        "acquired_bytes": sum(item["bytes"] for item in file_audit),
        "trajectory_only_compatibility": all(
            scene["av_contiguous"] and scene["av_observed_timesteps"] >= 5
            and scene["map_lane_segments"] > 0 and scene["motion_finite_fraction"] == 1.0
            for scene in scenes),
        "source_faithful_v2x_compatibility": False,
        "limitations": [
            "single recorded ego viewpoint; no independent sender or RSU observations",
            "observed column denotes forecasting segment, not sensor visibility",
            "no measured object dimensions in motion parquet",
            "no BCES or learned-TTL outcome was computed",
        ],
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenes": len(scenes), "acquired_bytes": report["acquired_bytes"],
                      "trajectory_only_compatibility": report["trajectory_only_compatibility"],
                      "source_faithful_v2x_compatibility": False,
                      "av_observed_timesteps": [item["av_observed_timesteps"] for item in scenes],
                      "map_lane_segments": [item["map_lane_segments"] for item in scenes]}, indent=2))


if __name__ == "__main__":
    main()
