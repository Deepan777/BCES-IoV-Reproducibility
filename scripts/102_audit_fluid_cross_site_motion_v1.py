#!/usr/bin/env python3
"""Frozen cross-archive FLUID motion-envelope test; no model refitting."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_DIR = ROOT / "data/external/fluid_cross_site_holdout_v1"
SOURCE = SOURCE_DIR / "derived_data_64179082.zip"
MANIFEST = SOURCE_DIR / "manifest.json"
DEVELOPMENT_REPORT = ROOT / "outputs/study_b/fluid_past_only_motion_development_v1/report.json"
DEVELOPMENT_REPORT_SHA256 = "6bd0eeaac3eefa215578e159a3f2845d83eb7595e8f991e0fda182758bc223d5"
SELECTOR = ROOT / "scripts/98_audit_fluid_past_only_motion_dev.py"
SELECTOR_SHA256 = "b479be1b7686c0c0ea1581a5c679c794db8c92632f593e4ea238a498c9a9a460"
OUTPUT = ROOT / "outputs/study_b/fluid_cross_site_motion_holdout_v1"
EXPECTED_FILE_ID = 64179082
EXPECTED_SIZE = 96_533_946
MIN_VIDEOS = 2
MIN_CARS = 200
MIN_TURN_CARS = 30
OVERALL_COVERAGE_GATE = 0.95
TURN_COVERAGE_GATE = 0.90


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_selector():
    if digest(SELECTOR) != SELECTOR_SHA256:
        raise RuntimeError("frozen TI selector changed")
    spec = importlib.util.spec_from_file_location("fluid_frozen_selector", SELECTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen selector")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def turn_group(value):
    normalized = (value or "").strip().lower()
    if normalized in ("l", "left"):
        return "left"
    if normalized in ("r", "right"):
        return "right"
    if normalized in ("s", "straight"):
        return "straight"
    return "other_or_missing"


def summary(rows, radius, selector):
    errors = [row["error_3s_m"] for row in rows]
    covered = sum(error <= radius for error in errors)
    return {"n": len(rows), "covered": covered,
            "coverage": covered / len(rows) if rows else None,
            "error_m": selector.summarize(errors)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if digest(DEVELOPMENT_REPORT) != DEVELOPMENT_REPORT_SHA256:
        raise RuntimeError("TI calibration report changed")
    development = json.loads(DEVELOPMENT_REPORT.read_text(encoding="utf-8"))
    if development["calibration_videos"] != ["20250529_video-1", "20250529_video-2"]:
        raise RuntimeError("unexpected TI calibration split")
    radius = development["calibration_radius_m"]
    if not (9.65 < radius < 9.66):
        raise RuntimeError("unexpected frozen TI radius")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest["file_id"] != EXPECTED_FILE_ID
            or manifest["official_bytes"] != EXPECTED_SIZE
            or SOURCE.stat().st_size != EXPECTED_SIZE
            or digest(SOURCE) != manifest["sha256"]):
        raise RuntimeError("selected cross-site archive identity/integrity mismatch")
    selector = load_selector()
    with zipfile.ZipFile(SOURCE) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate archive members")
        videos = sorted(name.removeprefix("derived_data/traj/").removesuffix("_Traj.csv")
                        for name in names
                        if name.startswith("derived_data/traj/") and name.endswith("_Traj.csv"))
        if not videos:
            raise RuntimeError("no trajectory recordings in selected archive")
        if any(video in development["calibration_videos"] + [development["test_video"]]
               for video in videos):
            raise RuntimeError("cross-site archive reuses opened TI video identity")
        for video in videos:
            if f"derived_data/route/{video}_Route.csv" not in names:
                raise RuntimeError("matching route table missing for " + video)
        cohorts = []
        for video in videos:
            cohorts.append(selector.select_windows(archive, video))
    rows = [row for cohort in cohorts for row in cohort["windows"]]
    by_turn = {group: summary([row for row in rows
                               if turn_group(row["future_derived_turn_label"]) == group],
                              radius, selector)
               for group in ("left", "right", "straight", "other_or_missing")}
    gate = {"at_least_two_recordings": len(videos) >= MIN_VIDEOS,
            "at_least_200_eligible_car_tracks": len(rows) >= MIN_CARS,
            "at_least_30_left": by_turn["left"]["n"] >= MIN_TURN_CARS,
            "at_least_30_right": by_turn["right"]["n"] >= MIN_TURN_CARS,
            "overall_empirical_coverage_at_least_95pct":
                bool(rows) and summary(rows, radius, selector)["coverage"] >= OVERALL_COVERAGE_GATE,
            "left_empirical_coverage_at_least_90pct":
                by_turn["left"]["n"] >= MIN_TURN_CARS
                and by_turn["left"]["coverage"] >= TURN_COVERAGE_GATE,
            "right_empirical_coverage_at_least_90pct":
                by_turn["right"]["n"] >= MIN_TURN_CARS
                and by_turn["right"]["coverage"] >= TURN_COVERAGE_GATE}
    report = {"status": "FLUID_CROSS_SITE_MOTION_HOLDOUT_V1",
              "source_file_id": EXPECTED_FILE_ID,
              "source_archive_sha256": manifest["sha256"],
              "source_manifest_sha256": digest(MANIFEST),
              "script_sha256": digest(Path(__file__)),
              "frozen_selector_sha256": SELECTOR_SHA256,
              "calibration_report_sha256": DEVELOPMENT_REPORT_SHA256,
              "calibration": "TI videos 1 and 2 only; TI video 3 was a prior within-site diagnostic",
              "fixed_radius_m": radius,
              "history_s": 0.3, "future_horizon_s": 3.0,
              "selector": "one all-real contiguous 34-frame car window nearest track midpoint",
              "predictor": "constant velocity from positions at t-0.3 s and t only",
              "future_route_turn_is_label_only": True,
              "videos": videos,
              "per_video": {cohort["video"]: summary(cohort["windows"], radius, selector)
                            for cohort in cohorts},
              "overall": summary(rows, radius, selector),
              "by_future_turn_label": by_turn,
              "prewritten_gates": gate,
              "all_prewritten_gates_passed": all(gate.values()),
              "limitations": ["global drone-derived offline tracks, not sender/receiver observations",
                              "one window per car does not remove within-video traffic dependence",
                              "empirical coverage is descriptive, not a distribution-free conditional guarantee",
                              "motion coverage does not establish BCES decision regret, safety, or byte utility"],
              "cohorts": cohorts}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                        encoding="utf-8")
    print(json.dumps({"videos": len(videos), "overall": report["overall"],
                      "by_turn": by_turn, "gates": gate}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
