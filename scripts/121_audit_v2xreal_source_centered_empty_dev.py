#!/usr/bin/env python3
"""Outcome-blind roadside-centered empty-view feasibility on validation scenes."""

from __future__ import annotations

from collections import defaultdict
import importlib.util
import json
import math
from pathlib import Path
import statistics
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_SCRIPT = ROOT / "scripts/120_audit_v2xreal_empty_view_feasibility_dev.py"
PREVIOUS_SCRIPT_SHA256 = "c5b7a40d0c9521daa12206be8b51c9c43be3a688d6dac9953bfa0ac9cc3a9f2a"
PREVIOUS_REPORT = ROOT / "outputs/study_b/v2xreal_empty_view_feasibility_val_v1.json"
PREVIOUS_REPORT_SHA256 = "9fada96d86d249bff4dd94977e389cf1a6359363ceade1b0a75824f67030973f"
OUTPUT = ROOT / "outputs/study_b/v2xreal_source_centered_empty_val_v1.json"
RADII_M = (30.0, 60.0, 120.0)


def finite_xy(frame, key):
    pose = frame.get(key)
    if (not isinstance(pose, list) or len(pose) < 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value)
                       for value in pose[:2])):
        raise RuntimeError("invalid source or receiver global pose")
    return float(pose[0]), float(pose[1])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    spec = importlib.util.spec_from_file_location("ego_centered_empty_audit", PREVIOUS_SCRIPT)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    for path, digest in ((PREVIOUS_SCRIPT, PREVIOUS_SCRIPT_SHA256),
                         (PREVIOUS_REPORT, PREVIOUS_REPORT_SHA256),
                         (previous.ARCHIVE, previous.ARCHIVE_SHA256),
                         (previous.SCHEMA, previous.SCHEMA_SHA256)):
        if previous.sha256(path) != digest:
            raise RuntimeError("frozen source changed: " + str(path))
    earlier = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    mobile = set(earlier["mobile_scenes"])
    if len(mobile) != 4:
        raise RuntimeError("unexpected mobile-scene cohort")
    rows = []
    with zipfile.ZipFile(previous.ARCHIVE) as archive:
        index = defaultdict(lambda: defaultdict(dict))
        for name in archive.namelist():
            if name.endswith(".yaml"):
                parts = name.split("/")
                if len(parts) == 4 and parts[1] in mobile and parts[2] in ("1", "-1"):
                    index[parts[1]][parts[2]][int(parts[3][:-5])] = name
        for scene in sorted(mobile):
            paired = sorted(set(index[scene]["1"]) & set(index[scene]["-1"]))
            for frame in paired:
                vehicle = yaml.safe_load(archive.read(index[scene]["1"][frame]))
                roadside = yaml.safe_load(archive.read(index[scene]["-1"][frame]))
                if vehicle.get("infra") is not False or roadside.get("infra") is not True:
                    raise RuntimeError("view role mismatch")
                receiver_xy = finite_xy(vehicle, "true_ego_pose")
                source_xy = finite_xy(roadside, "lidar_pose")
                source_to_receiver_m = math.dist(source_xy, receiver_xy)
                for radius in RADII_M:
                    source_objects, source_malformed = previous.objects_near(
                        roadside, source_xy, radius)
                    vehicle_objects, vehicle_malformed = previous.objects_near(
                        vehicle, source_xy, radius)
                    receiver_in_radius = source_to_receiver_m <= radius
                    rows.append({"scene": scene, "frame": frame, "radius_m": radius,
                                 "source_to_receiver_m": source_to_receiver_m,
                                 "roadside_objects": len(source_objects),
                                 "vehicle_objects": len(vehicle_objects),
                                 "roadside_empty": not source_objects,
                                 "receiver_inside_nominal_source_radius": receiver_in_radius,
                                 "empty_with_receiver_inside": not source_objects and receiver_in_radius,
                                 "vehicle_only_when_empty_and_receiver_inside":
                                     len(vehicle_objects - source_objects)
                                     if not source_objects and receiver_in_radius else 0,
                                 "malformed_roadside_objects": source_malformed,
                                 "malformed_vehicle_objects": vehicle_malformed})
            print(json.dumps({"audited_scene": scene, "paired_frames": len(paired)}), flush=True)
    if len(rows) != 479 * len(RADII_M):
        raise RuntimeError("paired mobile-frame cohort changed")
    summary = {}
    for radius in RADII_M:
        subset = [row for row in rows if row["radius_m"] == radius]
        relevant = [row for row in subset if row["empty_with_receiver_inside"]]
        by_scene = {}
        for scene in sorted(mobile):
            scene_rows = [row for row in subset if row["scene"] == scene]
            by_scene[scene] = {
                "paired_frames": len(scene_rows),
                "receiver_inside_nominal_source_radius_frames":
                    sum(row["receiver_inside_nominal_source_radius"] for row in scene_rows),
                "source_empty_frames": sum(row["roadside_empty"] for row in scene_rows),
                "empty_with_receiver_inside_frames":
                    sum(row["empty_with_receiver_inside"] for row in scene_rows),
                "relevant_empty_with_vehicle_only_object_frames":
                    sum(row["vehicle_only_when_empty_and_receiver_inside"] > 0
                        for row in scene_rows),
            }
        distances = [row["source_to_receiver_m"] for row in subset]
        summary[str(int(radius))] = {
            "paired_frames": len(subset),
            "source_to_receiver_distance_median_m": statistics.median(distances),
            "source_to_receiver_distance_p95_m": sorted(distances)[int(0.95 * (len(distances) - 1))],
            "receiver_inside_nominal_source_radius_frames":
                sum(row["receiver_inside_nominal_source_radius"] for row in subset),
            "source_empty_frames": sum(row["roadside_empty"] for row in subset),
            "empty_with_receiver_inside_frames": len(relevant),
            "relevant_empty_scene_count":
                sum(item["empty_with_receiver_inside_frames"] > 0
                    for item in by_scene.values()),
            "relevant_empty_with_vehicle_only_object_frames":
                sum(row["vehicle_only_when_empty_and_receiver_inside"] > 0
                    for row in relevant),
            "malformed_roadside_objects":
                sum(row["malformed_roadside_objects"] for row in subset),
            "by_scene": by_scene,
        }
    primary = summary["120"]
    gate = (primary["empty_with_receiver_inside_frames"] >= 20
            and primary["relevant_empty_scene_count"] >= 3
            and primary["relevant_empty_with_vehicle_only_object_frames"] == 0
            and primary["malformed_roadside_objects"] == 0)
    report = {"status": "VALIDATION_SOURCE_CENTERED_EMPTY_VIEW_FEASIBILITY_ONLY_V1",
              "script_sha256": previous.sha256(Path(__file__)),
              "prior_ego_centered_report_sha256": PREVIOUS_REPORT_SHA256,
              "archive_sha256": previous.ARCHIVE_SHA256,
              "mobile_scenes": sorted(mobile), "radii_m": list(RADII_M),
              "primary_radius_m": 120.0, "source_view_feasibility_screen": gate,
              "summary": summary, "rows": rows,
              "limitations": ["nominal radial distance is not a calibrated LiDAR field of view or line-of-sight polygon",
                              "receiver position inside radius does not prove its future conflict region is covered",
                              "paired YAML lists are annotations, not calibrated sensor detections or exhaustive world truth",
                              "no miss/emergence bound, ego route, BCES score, or driving outcome",
                              "479 frames nested inside four already-opened validation scenes"]}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screen": gate, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
