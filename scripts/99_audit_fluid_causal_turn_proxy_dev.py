#!/usr/bin/env python3
"""Development-only audit of a causal past-heading-change motion proxy."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "data/external/fluid_small_development_v1/derived_data_64179079.zip"
SOURCE_SHA256 = "e73ed4a0f2f400d447cf967f9173b8d4925174d49561e7a8408e1bb2be48bda3"
SELECTOR = ROOT / "scripts/98_audit_fluid_past_only_motion_dev.py"
SELECTOR_SHA256 = "b479be1b7686c0c0ea1581a5c679c794db8c92632f593e4ea238a498c9a9a460"
OUTPUT = ROOT / "outputs/study_b/fluid_causal_turn_proxy_development_v1"
CALIBRATION_VIDEO = "20250529_video-1"
VALIDATION_VIDEO = "20250529_video-2"
RESERVED_VIDEO = "20250529_video-3"
HISTORY_FRAMES = 6
SEGMENT_FRAMES = 3
MIN_SEGMENT_DISTANCE_M = 0.30
TURN_THRESHOLD_RAD = 0.10
TARGET_COVERAGE = 0.95
MIN_BIN_CALIBRATION_N = 20


def load_selector():
    if hashlib.sha256(SELECTOR.read_bytes()).hexdigest() != SELECTOR_SHA256:
        raise RuntimeError("frozen window selector changed")
    spec = importlib.util.spec_from_file_location("fluid_window_selector", SELECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load window selector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_positions(archive, video, selected):
    wanted = {row["track_id"]: {row["selected_frame"] - HISTORY_FRAMES,
                                 row["selected_frame"] - SEGMENT_FRAMES,
                                 row["selected_frame"]}
              for row in selected}
    points = {}
    filename = f"derived_data/traj/{video}_Traj.csv"
    reader = csv.DictReader(io.TextIOWrapper(archive.open(filename), encoding="utf-8-sig", newline=""))
    for row in reader:
        track_id = row["id"]
        if track_id not in wanted:
            continue
        frame = int(row["frame"])
        if frame in wanted[track_id]:
            points[(track_id, frame)] = (float(row["cx_m"]), float(row["cy_m"]),
                                         float(row["time"]), row["isReal"] == "1")
    return points


def classify(row, points):
    key, frame = row["track_id"], row["selected_frame"]
    try:
        early = points[(key, frame - HISTORY_FRAMES)]
        middle = points[(key, frame - SEGMENT_FRAMES)]
        current = points[(key, frame)]
    except KeyError:
        return None
    if not all(point[3] for point in (early, middle, current)):
        return None
    if abs((middle[2] - early[2]) - 0.3) > 1e-4 or abs((current[2] - middle[2]) - 0.3) > 1e-4:
        return None
    dx1, dy1 = middle[0] - early[0], middle[1] - early[1]
    dx2, dy2 = current[0] - middle[0], current[1] - middle[1]
    distance1, distance2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
    if min(distance1, distance2) < MIN_SEGMENT_DISTANCE_M:
        group, angle = "low_speed", None
    else:
        first, second = math.atan2(dy1, dx1), math.atan2(dy2, dx2)
        angle = abs((second - first + math.pi) % (2 * math.pi) - math.pi)
        group = "past_turning" if angle >= TURN_THRESHOLD_RAD else "past_straight"
    return {"video": row["video"], "track_id": key, "frame": frame,
            "causal_group": group, "past_heading_change_rad": angle,
            "past_segment_distances_m": [distance1, distance2],
            "error_3s_m": row["error_3s_m"],
            "future_derived_turn_label": row["future_derived_turn_label"]}


def upper_quantile(values):
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil((len(ordered) + 1) * TARGET_COVERAGE)
    return ordered[rank - 1] if rank <= len(ordered) else math.inf


def evaluate(rows, global_radius, group_radii):
    groups = {}
    for group in sorted({row["causal_group"] for row in rows}):
        subset = [row for row in rows if row["causal_group"] == group]
        radius = group_radii[group]
        groups[group] = {"n": len(subset), "radius_m": radius,
                         "global_covered": sum(row["error_3s_m"] <= global_radius for row in subset),
                         "group_covered": sum(row["error_3s_m"] <= radius for row in subset),
                         "error_p95_m": upper_quantile([row["error_3s_m"] for row in subset])}
    total = len(rows)
    global_hits = sum(row["error_3s_m"] <= global_radius for row in rows)
    group_hits = sum(row["error_3s_m"] <= group_radii[row["causal_group"]] for row in rows)
    mean_group_radius = sum(group_radii[row["causal_group"]] for row in rows) / total
    by_turn = {}
    for label in sorted({row["future_derived_turn_label"] or "missing" for row in rows}):
        subset = [row for row in rows if (row["future_derived_turn_label"] or "missing") == label]
        by_turn[label] = {"n": len(subset),
                          "global_covered": sum(row["error_3s_m"] <= global_radius for row in subset),
                          "group_covered": sum(row["error_3s_m"] <= group_radii[row["causal_group"]]
                                               for row in subset)}
    return {"n": total, "global_covered": global_hits,
            "global_coverage": global_hits / total,
            "group_covered": group_hits,
            "group_coverage": group_hits / total,
            "mean_assigned_group_radius_m": mean_group_radius,
            "causal_groups": groups, "future_turn_subgroups_label_only": by_turn}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("FLUID source archive changed")
    selector = load_selector()
    cohorts = []
    with zipfile.ZipFile(SOURCE) as archive:
        for video in (CALIBRATION_VIDEO, VALIDATION_VIDEO):
            selected = selector.select_windows(archive, video)["windows"]
            points = sample_positions(archive, video, selected)
            classified = [classify(row, points) for row in selected]
            valid = [row for row in classified if row is not None]
            cohorts.append({"video": video, "original_windows": len(selected),
                            "causal_feature_available": len(valid),
                            "causal_feature_missing": len(selected) - len(valid),
                            "rows": valid})
    calibration, validation = cohorts[0]["rows"], cohorts[1]["rows"]
    if not calibration or not validation:
        raise RuntimeError("empty development calibration/validation cohort")
    global_radius = upper_quantile([row["error_3s_m"] for row in calibration])
    group_radii, group_calibration_n = {}, {}
    for group in ("low_speed", "past_straight", "past_turning"):
        errors = [row["error_3s_m"] for row in calibration if row["causal_group"] == group]
        group_calibration_n[group] = len(errors)
        group_radii[group] = (upper_quantile(errors) if len(errors) >= MIN_BIN_CALIBRATION_N
                              else global_radius)
    report = {"status": "FLUID_CAUSAL_TURN_PROXY_DEVELOPMENT_ONLY",
              "source_archive_sha256": SOURCE_SHA256,
              "selector_sha256": SELECTOR_SHA256,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "calibration_video": CALIBRATION_VIDEO,
              "validation_video": VALIDATION_VIDEO,
              "reserved_video_not_read": RESERVED_VIDEO,
              "causal_feature": "two 0.3-s displacement headings from t-0.6,t-0.3,t; low-speed if either segment <0.30 m; past-turning if absolute heading change >=0.10 rad",
              "target_coverage": TARGET_COVERAGE,
              "minimum_bin_calibration_n": MIN_BIN_CALIBRATION_N,
              "global_radius_m": global_radius,
              "group_radii_m": group_radii,
              "group_calibration_n": group_calibration_n,
              "validation": evaluate(validation, global_radius, group_radii),
              "cohorts": cohorts,
              "not_external_confirmation": True,
              "not_receiver_observation_stream": True,
              "future_turn_labels_only_for_reporting": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"global_radius_m": global_radius,
                      "group_radii_m": group_radii,
                      "group_calibration_n": group_calibration_n,
                      "validation": report["validation"]}, indent=2))


if __name__ == "__main__":
    main()
