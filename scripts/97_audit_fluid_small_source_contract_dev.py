#!/usr/bin/env python3
"""Structural, outcome-blind audit of the opened FLUID TI development tranche."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/fluid_small_development_v1"
MANIFEST_SHA256 = "473d93fddab867a015e873ecbaa5d43c304d69b916a1f0d7160a206b38420c4c"
ARCHIVE_SHA256 = "e73ed4a0f2f400d447cf967f9173b8d4925174d49561e7a8408e1bb2be48bda3"
OUTPUT = ROOT / "outputs/study_b/fluid_small_source_contract_development_v1"
PREFIX = "derived_data/"
VIDEOS = ("20250529_video-1", "20250529_video-2", "20250529_video-3")


def reader(archive, name):
    return csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline=""))


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_video(archive, video):
    traj_name = PREFIX + f"traj/{video}_Traj.csv"
    route_name = PREFIX + f"route/{video}_Route.csv"
    signal_name = PREFIX + f"signal/{video}_signal.csv"
    tracks = {}
    last_time = {}
    dt = Counter()
    missing = Counter()
    types = Counter()
    rows = 0
    nonmonotone = 0
    real = 0
    required = ("frame", "id", "time", "cx_m", "cy_m", "vx", "vy",
                "yaw", "length_med", "width_med", "isReal")
    for row in reader(archive, traj_name):
        rows += 1
        if any(not row.get(key) for key in required):
            missing["required_field"] += 1
        if any(not finite_number(row.get(key)) for key in
               ("time", "cx_m", "cy_m", "vx", "vy", "yaw", "length_med", "width_med")):
            missing["nonfinite_numeric"] += 1
            continue
        key = row["id"]
        time_s = float(row["time"])
        real += row["isReal"] == "1"
        types[row["type"]] += 1
        if key in last_time:
            step_ms = round((time_s - last_time[key]) * 1000)
            dt[step_ms] += 1
            if step_ms <= 0:
                nonmonotone += 1
        last_time[key] = time_s
        if key not in tracks:
            tracks[key] = [time_s, time_s, 0]
        tracks[key][1] = time_s
        tracks[key][2] += 1
    csv.field_size_limit(10_000_000)
    route_rows, route_ids, turns, complete_route_times = 0, set(), Counter(), 0
    for row in reader(archive, route_name):
        route_rows += 1
        route_ids.add(row["id"])
        turns[row["turn"]] += 1
        complete_route_times += bool(row["in_time"] and row["out_time"])
    signal_rows, signal_groups, invalid_intervals = 0, set(), 0
    for row in reader(archive, signal_name):
        signal_rows += 1
        signal_groups.add((row["direction"], row["turn"]))
        if not finite_number(row["begin_time"]) or not finite_number(row["end_time"]):
            invalid_intervals += 1
        elif float(row["end_time"]) < float(row["begin_time"]):
            invalid_intervals += 1
    return {"video": video, "trajectory_rows": rows,
            "unique_tracks": len(tracks), "track_type_rows": dict(types),
            "real_detection_rows": real, "non_real_rows": rows - real,
            "required_field_missing_rows": missing["required_field"],
            "nonfinite_numeric_rows": missing["nonfinite_numeric"],
            "nonmonotone_within_track_steps": nonmonotone,
            "time_step_ms_counts": dict(sorted(dt.items())),
            "tracks_with_at_least_30_samples": sum(v[2] >= 30 for v in tracks.values()),
            "tracks_with_at_least_100_samples": sum(v[2] >= 100 for v in tracks.values()),
            "trajectory_first_time_s": min((v[0] for v in tracks.values()), default=None),
            "trajectory_last_time_s": max((v[1] for v in tracks.values()), default=None),
            "route_rows": route_rows, "unique_route_ids": len(route_ids),
            "tracks_with_route_row": len(set(tracks) & route_ids),
            "route_ids_without_track": len(route_ids - set(tracks)),
            "complete_route_time_rows": complete_route_times,
            "turn_counts": dict(turns),
            "signal_rows": signal_rows, "signal_groups": len(signal_groups),
            "invalid_signal_intervals": invalid_intervals}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    manifest_path = SOURCE / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("FLUID source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_path = SOURCE / "derived_data_64179079.zip"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("FLUID source archive changed")
    if manifest["doi"] != "10.6084/m9.figshare.29974954.v2":
        raise RuntimeError("wrong FLUID version")
    with zipfile.ZipFile(archive_path) as archive:
        files = set(archive.namelist())
        required = {PREFIX + f"traj/{v}_Traj.csv" for v in VIDEOS}
        required |= {PREFIX + f"route/{v}_Route.csv" for v in VIDEOS}
        required |= {PREFIX + f"signal/{v}_signal.csv" for v in VIDEOS}
        required |= {PREFIX + "map/TI.osm", PREFIX + "conflict/TI_conflict.csv"}
        if not required <= files:
            raise RuntimeError("required TI source files absent")
        videos = [audit_video(archive, video) for video in VIDEOS]
        conflict_rows, valid_conflict_ids = 0, 0
        for row in reader(archive, PREFIX + "conflict/TI_conflict.csv"):
            conflict_rows += 1
            valid_conflict_ids += bool(row["Car 1 ID"] and row["Car 2 ID"] and row["Scene"])
        osm = ET.fromstring(archive.read(PREFIX + "map/TI.osm"))
        map_counts = Counter(child.tag for child in osm)
        relation_types = Counter()
        for relation in osm.findall("relation"):
            tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
            relation_types[tags.get("type", "missing")] += 1
    result = {"status": "FLUID_TI_SMALL_DEVELOPMENT_SOURCE_CONTRACT_AUDIT",
              "source_manifest_sha256": MANIFEST_SHA256,
              "source_archive_sha256": ARCHIVE_SHA256,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "doi": manifest["doi"], "license": manifest["license"],
              "videos": videos, "conflict_rows": conflict_rows,
              "conflict_rows_with_nonempty_ids_and_scene": valid_conflict_ids,
              "osm_element_counts": dict(map_counts),
              "osm_relation_types": dict(relation_types),
              "receiver_view_present": False,
              "route_turn_and_exit_labels_are_future_derived": True,
              "not_policy_confirmation": True,
              "unopened_derived_archive_ids": manifest["unopened_derived_data_file_ids"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"videos": [{key: item[key] for key in
                                  ("video", "trajectory_rows", "unique_tracks",
                                   "tracks_with_at_least_30_samples", "route_rows",
                                   "tracks_with_route_row", "nonmonotone_within_track_steps",
                                   "required_field_missing_rows")}
                                 for item in videos],
                      "conflict_rows": conflict_rows,
                      "osm_element_counts": result["osm_element_counts"]}, indent=2))


if __name__ == "__main__":
    main()
