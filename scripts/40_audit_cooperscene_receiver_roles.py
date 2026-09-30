#!/usr/bin/env python3
"""Outcome-blind all-agent motion/source audit for CooperScene windows."""
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

MANIFESTS = [ROOT / "data/manifests/cooperscene_annotations_v1.json",
             ROOT / "data/manifests/cooperscene_annotations_extra_agents_v1.json"]
OUT = ROOT / "outputs/study_b/cooperscene_receiver_role_audit_v1.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    all_records = []
    manifest_hashes = {}
    for path in MANIFESTS:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if manifest["file_count"] != 1680:
            raise ValueError(f"incomplete manifest: {path}")
        manifest_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        all_records.extend(manifest["records"])
    if len({r["relative_path"] for r in all_records}) != 3360:
        raise ValueError("duplicate or missing annotation paths")
    scenes = defaultdict(lambda: defaultdict(dict))
    key_counts = Counter()
    for record in all_records:
        path = ROOT / record["relative_path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        row = yaml.safe_load(raw)
        if not isinstance(row, dict):
            raise ValueError(f"invalid YAML root: {path}")
        key_counts.update(row.keys())
        scenes[record["scene"]][record["frame"]][record["agent"]] = row
    results = []
    for scene, frames in sorted(scenes.items()):
        frame_ids = sorted(frames)
        if len(frame_ids) != 60 or any(b - a != 1 for a, b in zip(frame_ids, frame_ids[1:])):
            raise ValueError(f"incomplete selected frame sequence: {scene}")
        agent_stats = {}
        for agent in range(4):
            poses = []
            counts = []
            for frame in frame_ids:
                record = frames[frame].get(agent)
                if record is None:
                    continue
                pose = record.get("lidar_pose")
                if isinstance(pose, list) and len(pose) == 6 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose):
                    poses.append((frame, pose))
                counts.append(len(record.get("vehicles", {})))
            movement = math.hypot(poses[-1][1][0] - poses[0][1][0],
                                  poses[-1][1][1] - poses[0][1][1]) if len(poses) == 60 else None
            agent_stats[str(agent)] = {"valid_pose_frames": len(poses), "net_displacement_m": movement,
                                       "mean_visible_vehicles": sum(counts) / len(counts) if counts else None}
        chosen = next((agent for agent in (1, 2, 3)
                       if agent_stats[str(agent)]["net_displacement_m"] is not None
                       and agent_stats[str(agent)]["net_displacement_m"] > 1.0), None)
        overlaps = 0
        mismatched = 0
        if chosen is not None:
            for views in frames.values():
                infra = views[0].get("vehicles", {})
                receiver = views[chosen].get("vehicles", {})
                for vid in infra.keys() & receiver.keys():
                    a = infra[vid].get("location")
                    b = receiver[vid].get("location")
                    if isinstance(a, list) and isinstance(b, list) and len(a) >= 2 and len(b) >= 2:
                        overlaps += 1
                        mismatched += math.hypot(a[0] - b[0], a[1] - b[1]) > 0.01
        results.append({"scene": scene, "frame_count": 60, "agent_stats": agent_stats,
                        "selected_receiver_agent": chosen,
                        "shared_object_observations": overlaps,
                        "shared_object_xy_mismatches_gt_1cm": mismatched})
    report = {"status": "SCHEMA_ONLY_NO_BCES_OUTCOMES", "manifest_sha256": manifest_hashes,
              "parsed_file_count": len(all_records), "take_count": len(results),
              "top_level_key_counts": dict(key_counts),
              "route_or_map_field_count": sum(v for k, v in key_counts.items() if "route" in str(k).lower() or "map" in str(k).lower()),
              "selected_mobile_takes": sum(r["selected_receiver_agent"] is not None for r in results),
              "takes": results}
    write_json_atomic(OUT, report)
    print(json.dumps({k: report[k] for k in ("status", "parsed_file_count", "take_count", "route_or_map_field_count", "selected_mobile_takes")}, indent=2))
    for take in results:
        print(take["scene"], take["selected_receiver_agent"],
              {k: round(v["net_displacement_m"], 3) if v["net_displacement_m"] is not None else None
               for k, v in take["agent_stats"].items()},
              take["shared_object_observations"], take["shared_object_xy_mismatches_gt_1cm"])


if __name__ == "__main__":
    main()
