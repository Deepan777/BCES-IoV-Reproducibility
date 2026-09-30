#!/usr/bin/env python3
"""Independent raw-CSV replay of the frozen FLUID holdout selections/errors."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/fluid_cross_site_holdout_v1/derived_data_64179082.zip"
MANIFEST = ROOT / "data/external/fluid_cross_site_holdout_v1/manifest.json"
REPORT = ROOT / "outputs/study_b/fluid_cross_site_motion_holdout_v1/report.json"
REPORT_SHA256 = "bffea38f71849e581dfab4b08561047779428e6e4e9c5bf99b75d083f5acb309"
OUTPUT = ROOT / "outputs/study_b/fluid_cross_site_motion_holdout_v1/independent_replay_receipt.json"


def csv_rows(archive, name):
    return csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline=""))


def replay_video(archive, video):
    route = {row["id"]: row["turn"]
             for row in csv_rows(archive, f"derived_data/route/{video}_Route.csv")}
    tracks = {}
    for row in csv_rows(archive, f"derived_data/traj/{video}_Traj.csv"):
        if row["type"] == "car":
            tracks.setdefault(row["id"], {})[int(row["frame"])] = (
                float(row["time"]), float(row["cx_m"]), float(row["cy_m"]),
                row["isReal"] == "1")
    selected = {}
    for track_id, frames in tracks.items():
        middle = (min(frames) + max(frames)) / 2
        feasible = []
        for current in frames:
            if current - 3 not in frames or current + 30 not in frames:
                continue
            present = [frames.get(frame) for frame in range(current - 3, current + 31)]
            if any(point is None or not point[3] for point in present):
                continue
            now = frames[current][0]
            if any(abs(point[0] - (now + (frame - current) / 10)) > 0.0001
                   for frame, point in zip(range(current - 3, current + 31), present)):
                continue
            feasible.append(current)
        if not feasible:
            continue
        frame = min(feasible, key=lambda value: (abs(value - middle), value))
        past, now, future = frames[frame - 3], frames[frame], frames[frame + 30]
        projected_x = now[1] + 10 * (now[1] - past[1])
        projected_y = now[2] + 10 * (now[2] - past[2])
        selected[track_id] = (frame,
                              math.dist((projected_x, projected_y), (future[1], future[2])),
                              route.get(track_id))
    return selected


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if hashlib.sha256(REPORT.read_bytes()).hexdigest() != REPORT_SHA256:
        raise RuntimeError("frozen holdout report changed")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != manifest["sha256"]:
        raise RuntimeError("source archive changed")
    radius = report["fixed_radius_m"]
    verified_rows = 0
    verified_hits = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        for cohort in report["cohorts"]:
            video = cohort["video"]
            raw = replay_video(archive, video)
            reported = {row["track_id"]: row for row in cohort["windows"]}
            if raw.keys() != reported.keys():
                raise RuntimeError("selected track identities differ for " + video)
            if len(raw) != cohort["eligible_car_tracks"]:
                raise RuntimeError("eligible count differs for " + video)
            for track_id, (frame, error, turn) in raw.items():
                row = reported[track_id]
                if (row["selected_frame"] != frame
                        or abs(row["error_3s_m"] - error) > 1e-8
                        or row["future_derived_turn_label"] != turn):
                    raise RuntimeError("selected frame/error/turn mismatch")
                verified_rows += 1
                verified_hits += error <= radius
    if (verified_rows != report["overall"]["n"]
            or verified_hits != report["overall"]["covered"]):
        raise RuntimeError("independent aggregate coverage mismatch")
    receipt = {"status": "INDEPENDENT_RAW_CSV_REPLAY_PASSED",
               "source_sha256": manifest["sha256"],
               "frozen_report_sha256": REPORT_SHA256,
               "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "videos": len(report["cohorts"]),
               "verified_selected_track_windows": verified_rows,
               "verified_covered": verified_hits,
               "selected_frames_errors_and_route_labels_match": True}
    OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
