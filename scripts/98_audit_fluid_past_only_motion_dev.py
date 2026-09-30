#!/usr/bin/env python3
"""Past-only 3-s FLUID motion-error audit with a video-held-out test."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/fluid_small_development_v1/derived_data_64179079.zip"
SOURCE_SHA256 = "e73ed4a0f2f400d447cf967f9173b8d4925174d49561e7a8408e1bb2be48bda3"
OUTPUT = ROOT / "outputs/study_b/fluid_past_only_motion_development_v1"
VIDEOS = ("20250529_video-1", "20250529_video-2", "20250529_video-3")
HISTORY_FRAMES = 3
FUTURE_FRAMES = 30
FRAME_PERIOD_S = 0.1
PRIMARY_TYPE = "car"
COVERAGE_TARGET = 0.95


def rows(archive, name):
    return csv.DictReader(io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline=""))


def quantile_nearest_rank(values, level):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(level * len(ordered)) - 1)]


def route_turns(archive, video):
    csv.field_size_limit(10_000_000)
    filename = f"derived_data/route/{video}_Route.csv"
    return {row["id"]: row["turn"] for row in rows(archive, filename)}


def select_windows(archive, video):
    turns = route_turns(archive, video)
    tracks = {}
    for row in rows(archive, f"derived_data/traj/{video}_Traj.csv"):
        if row["type"] != PRIMARY_TYPE:
            continue
        key = row["id"]
        frame = int(row["frame"])
        tracks.setdefault(key, {})[frame] = (
            float(row["time"]), float(row["cx_m"]), float(row["cy_m"]),
            int(row["isReal"]), float(row["yaw"]))
    selected = []
    for track_id, samples in sorted(tracks.items(), key=lambda item: int(item[0])):
        if not samples:
            continue
        middle = (min(samples) + max(samples)) / 2
        eligible = []
        for frame in samples:
            window = [samples.get(frame + offset)
                      for offset in range(-HISTORY_FRAMES, FUTURE_FRAMES + 1)]
            if any(item is None or item[3] != 1 for item in window):
                continue
            t0 = samples[frame][0]
            if any(abs(item[0] - (t0 + offset * FRAME_PERIOD_S)) > 1e-4
                   for offset, item in zip(range(-HISTORY_FRAMES, FUTURE_FRAMES + 1), window)):
                continue
            eligible.append(frame)
        if not eligible:
            continue
        frame = min(eligible, key=lambda value: (abs(value - middle), value))
        past, current, future = (samples[frame - HISTORY_FRAMES],
                                 samples[frame], samples[frame + FUTURE_FRAMES])
        vx = (current[1] - past[1]) / (HISTORY_FRAMES * FRAME_PERIOD_S)
        vy = (current[2] - past[2]) / (HISTORY_FRAMES * FRAME_PERIOD_S)
        x_pred = current[1] + vx * FUTURE_FRAMES * FRAME_PERIOD_S
        y_pred = current[2] + vy * FUTURE_FRAMES * FRAME_PERIOD_S
        error = math.hypot(x_pred - future[1], y_pred - future[2])
        yaw_change = abs((future[4] - current[4] + math.pi) % (2 * math.pi) - math.pi)
        selected.append({"video": video, "track_id": track_id,
                         "selected_frame": frame, "current_time_s": current[0],
                         "history_start_time_s": past[0],
                         "future_label_time_s": future[0],
                         "past_only_velocity_mps": [vx, vy],
                         "error_3s_m": error, "yaw_change_3s_rad": yaw_change,
                         "future_derived_turn_label": turns.get(track_id)})
    return {"video": video, "car_tracks": len(tracks),
            "eligible_car_tracks": len(selected),
            "excluded_car_tracks": len(tracks) - len(selected),
            "windows": selected}


def summarize(values):
    return {"n": len(values), "mean": sum(values) / len(values) if values else None,
            "p50_nearest_rank": quantile_nearest_rank(values, 0.5),
            "p90_nearest_rank": quantile_nearest_rank(values, 0.9),
            "p95_nearest_rank": quantile_nearest_rank(values, 0.95),
            "max": max(values) if values else None}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("FLUID source archive changed")
    with zipfile.ZipFile(SOURCE) as archive:
        cohorts = [select_windows(archive, video) for video in VIDEOS]
    calibration = [row["error_3s_m"] for cohort in cohorts[:2] for row in cohort["windows"]]
    test = cohorts[2]["windows"]
    if not calibration or not test:
        raise RuntimeError("empty calibration or held-out video cohort")
    ordered = sorted(calibration)
    rank = math.ceil((len(ordered) + 1) * COVERAGE_TARGET)
    radius = ordered[rank - 1] if rank <= len(ordered) else math.inf
    hits = [row["error_3s_m"] <= radius for row in test]
    by_turn = {}
    for label in sorted({row["future_derived_turn_label"] or "missing" for row in test}):
        group = [row for row in test if (row["future_derived_turn_label"] or "missing") == label]
        by_turn[label] = {"n": len(group), "covered": sum(row["error_3s_m"] <= radius for row in group),
                          "coverage": sum(row["error_3s_m"] <= radius for row in group) / len(group),
                          "error_summary": summarize([row["error_3s_m"] for row in group])}
    report = {"status": "FLUID_PAST_ONLY_MOTION_DEVELOPMENT_VIDEO_HELD_OUT",
              "source_archive_sha256": SOURCE_SHA256,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "primary_agent_type": PRIMARY_TYPE,
              "history_duration_s": HISTORY_FRAMES * FRAME_PERIOD_S,
              "future_horizon_s": FUTURE_FRAMES * FRAME_PERIOD_S,
              "selection": "one full-real-detection 34-frame window per car track, eligible frame nearest track midpoint, ties earlier frame",
              "prediction": "constant velocity from positions at t-0.3 and t; no future or route feature",
              "calibration_videos": list(VIDEOS[:2]),
              "test_video": VIDEOS[2],
              "coverage_target": COVERAGE_TARGET,
              "calibration_rank": rank,
              "calibration_radius_m": radius,
              "calibration_summary": summarize(calibration),
              "test_summary": summarize([row["error_3s_m"] for row in test]),
              "test_covered": sum(hits), "test_total": len(hits),
              "test_empirical_coverage": sum(hits) / len(hits),
              "test_by_future_turn_label": by_turn,
              "cohorts": cohorts,
              "claims": ["one held-out video at the same TI intersection/date, not cross-site confirmation",
                         "per-track windows can be correlated through shared traffic; nominal quantile coverage not guaranteed",
                         "recorded global drone positions are not a receiver-specific observation stream",
                         "turn labels derive from future route and are used only for subgroup reporting",
                         "position tracks may have offline processing even when isReal=1",
                         "this does not validate BCES decision-regret or communication benefit"],
              "not_policy_confirmation": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"calibration_tracks": len(calibration),
                      "calibration_radius_m": radius,
                      "held_out_tracks": len(test),
                      "held_out_coverage": report["test_empirical_coverage"],
                      "held_out_error_summary": report["test_summary"],
                      "held_out_by_turn": by_turn}, indent=2))


if __name__ == "__main__":
    main()
