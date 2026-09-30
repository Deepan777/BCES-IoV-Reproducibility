#!/usr/bin/env python3
"""Outcome-blind paired-view empty-observation feasibility on validation scenes."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/v2x_real_lidar64/val.zip"
ARCHIVE_SHA256 = "e06e6525f31da009566ae553dce4cf34cc2c636f7a1162ed41809ee804fd9b7b"
SCHEMA = ROOT / "outputs/study_b/v2xreal_receiver_schema_audit_val_v1.json"
SCHEMA_SHA256 = "ef380bc3ed4e9ee33df85a9ceebe08eb3769a085cf5e00dd0017dd879d6a7062"
OUTPUT = ROOT / "outputs/study_b/v2xreal_empty_view_feasibility_val_v1.json"
RADII_M = (30.0, 60.0, 120.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def objects_near(frame: dict, ego_xy: tuple[float, float], radius_m: float):
    objects = frame.get("vehicles")
    if not isinstance(objects, dict):
        raise RuntimeError("missing per-view object dictionary")
    nearby = set()
    malformed = 0
    for identity, obj in objects.items():
        location = obj.get("location") if isinstance(obj, dict) else None
        if (not isinstance(location, list) or len(location) < 2
                or not all(isinstance(v, (int, float)) and math.isfinite(v)
                           for v in location[:2])):
            malformed += 1
            continue
        if math.dist(tuple(map(float, location[:2])), ego_xy) <= radius_m:
            nearby.add(str(identity))
    return nearby, malformed


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256(SCHEMA) != SCHEMA_SHA256 or sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise RuntimeError("independent validation source changed")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    if schema["status"] != "OUTCOME_BLIND_SOURCE_CONTRACT_AUDIT_NO_BCES_RESULTS":
        raise RuntimeError("paired-view source schema not established")
    mobile = {scene["base_scene"] for scene in schema["scenes"]
              if scene["paired_vehicle_roadside_frames"] > 0
              and (scene.get("ego_net_displacement_m") or 0) >= 10.0}
    if len(mobile) != 4:
        raise RuntimeError("unexpected mobile validation scene cohort")
    rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        index = defaultdict(lambda: defaultdict(dict))
        for name in archive.namelist():
            if not name.endswith(".yaml"):
                continue
            parts = name.split("/")
            if len(parts) != 4 or parts[1] not in mobile or parts[2] not in ("1", "-1"):
                continue
            index[parts[1]][parts[2]][int(parts[3][:-5])] = name
        for scene in sorted(mobile):
            paired = sorted(set(index[scene]["1"]) & set(index[scene]["-1"]))
            expected = next(s["paired_vehicle_roadside_frames"] for s in schema["scenes"]
                            if s["base_scene"] == scene)
            if len(paired) != expected:
                raise RuntimeError("paired-frame denominator differs from schema audit")
            for frame in paired:
                vehicle = yaml.safe_load(archive.read(index[scene]["1"][frame]))
                roadside = yaml.safe_load(archive.read(index[scene]["-1"][frame]))
                if vehicle.get("infra") is not False or roadside.get("infra") is not True:
                    raise RuntimeError("view role mismatch")
                pose = vehicle.get("true_ego_pose")
                if (not isinstance(pose, list) or len(pose) < 2
                        or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                   for v in pose[:2])):
                    raise RuntimeError("invalid vehicle global pose")
                ego_xy = float(pose[0]), float(pose[1])
                for radius in RADII_M:
                    rsu, rsu_bad = objects_near(roadside, ego_xy, radius)
                    car, car_bad = objects_near(vehicle, ego_xy, radius)
                    rows.append({"scene": scene, "frame": frame, "radius_m": radius,
                                 "roadside_objects": len(rsu), "vehicle_objects": len(car),
                                 "roadside_empty": not rsu,
                                 "vehicle_only_objects": len(car - rsu),
                                 "vehicle_only_when_roadside_empty": len(car - rsu) if not rsu else 0,
                                 "malformed_roadside_objects": rsu_bad,
                                 "malformed_vehicle_objects": car_bad})
            print(json.dumps({"audited_scene": scene, "paired_frames": len(paired)}), flush=True)
    if len(rows) != 479 * len(RADII_M):
        raise RuntimeError("fixed mobile paired-frame denominator changed")
    summary = {}
    for radius in RADII_M:
        subset = [row for row in rows if row["radius_m"] == radius]
        empty = [row for row in subset if row["roadside_empty"]]
        by_scene = {}
        for scene in sorted(mobile):
            scene_rows = [row for row in subset if row["scene"] == scene]
            by_scene[scene] = {"paired_frames": len(scene_rows),
                               "roadside_empty_frames":
                                   sum(row["roadside_empty"] for row in scene_rows),
                               "empty_with_vehicle_only_object_frames":
                                   sum(row["vehicle_only_when_roadside_empty"] > 0
                                       for row in scene_rows)}
        summary[str(int(radius))] = {"paired_frames": len(subset),
                                     "roadside_empty_frames": len(empty),
                                     "empty_scene_count":
                                         sum(item["roadside_empty_frames"] > 0
                                             for item in by_scene.values()),
                                     "empty_with_vehicle_only_object_frames":
                                         sum(row["vehicle_only_when_roadside_empty"] > 0
                                             for row in empty),
                                     "frames_with_any_vehicle_only_object":
                                         sum(row["vehicle_only_objects"] > 0
                                             for row in subset),
                                     "malformed_roadside_objects":
                                         sum(row["malformed_roadside_objects"] for row in subset),
                                     "malformed_vehicle_objects":
                                         sum(row["malformed_vehicle_objects"] for row in subset),
                                     "by_scene": by_scene}
    primary = summary["120"]
    gate = (primary["roadside_empty_frames"] >= 20
            and primary["empty_scene_count"] >= 3
            and primary["empty_with_vehicle_only_object_frames"] == 0
            and primary["malformed_roadside_objects"] == 0)
    report = {"status": "VALIDATION_VIEW_EMPTY_INFORMATION_FEASIBILITY_ONLY_V1",
              "script_sha256": sha256(Path(__file__)),
              "archive_sha256": ARCHIVE_SHA256, "schema_sha256": SCHEMA_SHA256,
              "mobile_scenes": sorted(mobile), "radii_m": list(RADII_M),
              "primary_radius_m": 120.0,
              "source_view_feasibility_screen": gate,
              "summary": summary, "rows": rows,
              "limitations": ["per-agent YAML object lists are annotations, not calibrated sensor detections",
                              "vehicle-only annotations are a necessary-condition proxy, not complete ground truth",
                              "no ego route or crossing conflict region is identified by this radial audit",
                              "empty-view prevalence is not a false-negative-rate estimate without true absence labels",
                              "frames are nested inside four validation acquisition scenes",
                              "no BCES, learned TTL, driving outcome, or communication-policy result"]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"screen": gate, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
