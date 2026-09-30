#!/usr/bin/env python3
"""Integrity and receiver/world schema audit without any BCES decisions."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.reproducibility import write_json_atomic

MANIFEST = ROOT / "data/manifests/cooperscene_annotations_v1.json"
OUT = ROOT / "outputs/study_b/cooperscene_schema_audit_v1.json"


def audit():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["file_count"] != 1680 or len(manifest["records"]) != 1680:
        raise ValueError("incomplete annotation manifest")
    per_scene = defaultdict(lambda: defaultdict(dict))
    keys = Counter()
    route_keys = Counter()
    parse_errors = []
    for record in manifest["records"]:
        path = ROOT / record["relative_path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        try:
            data = yaml.safe_load(raw)
        except Exception as exc:
            parse_errors.append({"path": record["relative_path"], "error": type(exc).__name__})
            continue
        if not isinstance(data, dict):
            parse_errors.append({"path": record["relative_path"], "error": "not_mapping"})
            continue
        keys.update(data.keys())
        route_keys.update(k for k in data if "route" in str(k).lower() or "map" in str(k).lower())
        per_scene[record["scene"]][record["frame"]][record["agent"]] = data
    takes = []
    for scene, frames in sorted(per_scene.items()):
        sorted_frames = sorted(frames)
        position = {agent: [] for agent in (0, 1)}
        object_counts = {agent: [] for agent in (0, 1)}
        shared_count = 0
        shared_location_differences = []
        missing_pose = Counter()
        for frame in sorted_frames:
            views = frames[frame]
            for agent in (0, 1):
                view = views.get(agent, {})
                pose = view.get("lidar_pose")
                if not isinstance(pose, list) or len(pose) != 6 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose):
                    missing_pose[agent] += 1
                else:
                    position[agent].append((frame, pose))
                object_counts[agent].append(len(view.get("vehicles", {})))
            first = views.get(0, {}).get("vehicles", {})
            second = views.get(1, {}).get("vehicles", {})
            for vid in first.keys() & second.keys():
                a, b = first[vid].get("location"), second[vid].get("location")
                if isinstance(a, list) and isinstance(b, list) and len(a) >= 2 and len(b) >= 2:
                    shared_count += 1
                    shared_location_differences.append(math.hypot(a[0] - b[0], a[1] - b[1]))
        movement = {}
        for agent in (0, 1):
            poses = position[agent]
            steps = [math.hypot(b[1][0] - a[1][0], b[1][1] - a[1][1])
                     for a, b in zip(poses, poses[1:]) if b[0] - a[0] == 1]
            movement[str(agent)] = {"pose_frames": len(poses),
                                    "net_xy_m": math.hypot(poses[-1][1][0] - poses[0][1][0],
                                                           poses[-1][1][1] - poses[0][1][1]) if poses else None,
                                    "maximum_step_xy_m": max(steps) if steps else None,
                                    "mean_step_xy_m": sum(steps) / len(steps) if steps else None,
                                    "mean_objects_per_frame": sum(object_counts[agent]) / max(len(object_counts[agent]), 1),
                                    "missing_pose_frames": missing_pose[agent]}
        takes.append({"scene": scene, "frames": len(sorted_frames),
                      "paired_views": sum(0 in v and 1 in v for v in frames.values()),
                      "consecutive_frame_ids": all(b - a == 1 for a, b in zip(sorted_frames, sorted_frames[1:])),
                      "movement": movement, "shared_object_id_observations": shared_count,
                      "maximum_shared_xy_discrepancy_m": max(shared_location_differences) if shared_location_differences else None,
                      "nonmatching_shared_xy_count": sum(v > 0.01 for v in shared_location_differences)})
    return {"status": "schema_only_no_bces_score_or_decision_label", "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "parsed_file_count": sum(len(v) for frames in per_scene.values() for v in frames.values()),
            "parse_errors": parse_errors, "top_level_key_counts": dict(keys), "map_or_route_key_counts": dict(route_keys),
            "take_count": len(takes), "takes": takes}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    result = audit()
    write_json_atomic(OUT, result)
    print(json.dumps({k: result[k] for k in ("status", "parsed_file_count", "take_count", "parse_errors", "map_or_route_key_counts")}, indent=2))
    for take in result["takes"]:
        print(take["scene"], take["frames"], take["movement"]["0"]["net_xy_m"],
              take["movement"]["1"]["net_xy_m"], take["shared_object_id_observations"],
              take["maximum_shared_xy_discrepancy_m"])


if __name__ == "__main__":
    main()
