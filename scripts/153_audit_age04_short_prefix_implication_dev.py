#!/usr/bin/env python3
"""Triangle-inequality transfer of a 0.2-s prefix bound to 0.4-s age."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

prior = importlib.import_module("scripts.122_audit_empty_disk_reachability_dev")
PLAN = ROOT / "docs/STUDY_B_AGE04_SHORT_PREFIX_IMPLICATION_DEV_V1_PLAN.md"
PREFIX = ROOT / "outputs/study_b/empty_view_horizon_radius_dev_v1/report.json"
PREFIX_SHA256 = "ce5dd55f6b560214972f3c25d4f811d45e008cb59015d246d58d43c66a10987c"
ORIGINAL = ROOT / "outputs/study_b/empty_disk_reachability_development_v1/report.json"
ORIGINAL_SHA256 = "5ec3dbea32a51255a544b7ce34de58cda7cbfefe33a7513fe061c8ee88104f33"
OUTPUT = ROOT / "outputs/study_b/age04_short_prefix_implication_dev_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(row):
    return f"{row['seed']}/{row['timing']}/{row['condition']}/{row['timestamp_ms']}"


def xy(row):
    values = row["ego_xy_m"]
    if len(values) != 2 or not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("bad source center")
    return float(values[0]), float(values[1])


def main():
    if OUTPUT.exists():
        raise RuntimeError("output exists; preserve first run")
    for path, expected in ((PREFIX, PREFIX_SHA256), (ORIGINAL, ORIGINAL_SHA256),
                           (prior.BASE / "report.json", prior.BASE_REPORT_SHA256),
                           (prior.REGISTRATION, prior.REGISTRATION_SHA256)):
        if digest(path) != expected:
            raise RuntimeError("frozen input changed: " + str(path))
    prefix = json.loads(PREFIX.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    baseline = json.loads((prior.BASE / "report.json").read_text(encoding="utf-8"))
    if original["summary"]["0.4"]["approach_states"] != 568 or original["summary"]["0.4"]["conditional_clear_states"] != 328:
        raise RuntimeError("original 0.4-s result changed")
    newer = {row["key"]: row for row in prefix["rows"]}
    if len(newer) != 600:
        raise RuntimeError("newer path-prefix denominator changed")
    older = {}
    for row in original["rows"]:
        if row["age_s"] != 0.4:
            continue
        k = row_key(row)
        if k in older or k not in newer or not row["candidate_available"]:
            raise RuntimeError("older row not uniquely paired with a planned newer row")
        older[k] = row
    if len(older) != 568:
        raise RuntimeError("older source key set changed")
    centers = {}
    branch_names = sorted(name for name in baseline["artifact_sha256"]
                          if name.endswith("_absent_negative_evidence_periodic_payload.json.gz"))
    if len(branch_names) != 32:
        raise RuntimeError("branch count changed")
    for name in branch_names:
        path = prior.BASE / name
        if digest(path) != baseline["artifact_sha256"][name]:
            raise RuntimeError("branch hash changed")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        seed = int(name.split("_")[0])
        timing = "near" if "_near_" in name else "timing_control"
        condition = "nominal" if "_nominal_" in name else "impaired"
        visibility = {item["timestamp_ms"]: item for item in branch["visibility_audit"]}
        for sample in branch["rows"]:
            timestamp = sample["timestamp_ms"]
            k = f"{seed}/{timing}/{condition}/{timestamp}"
            if k not in older:
                continue
            if k in centers:
                raise RuntimeError("duplicate branch row key")
            old_source = visibility.get(timestamp - 400)
            new_source = visibility.get(timestamp - 200)
            if old_source is None or new_source is None:
                raise RuntimeError("source timestamp missing")
            if old_source["target_xy_m"] is not None or new_source["target_xy_m"] is not None:
                raise RuntimeError("source was not empty")
            centers[k] = (xy(old_source), xy(new_source))
    if set(centers) != set(older):
        raise RuntimeError("paired source centers incomplete")
    rows = []
    for k, old_row in sorted(older.items()):
        old_center, new_center = centers[k]
        shift = math.dist(old_center, new_center)
        if shift > 4.0 + 1e-7:
            raise RuntimeError("source-center displacement violated 20-m/s bound")
        required_new = newer[k]["required_radius_m"]["0.4"]
        upper = required_new + shift + 20.0 * 0.2
        rows.append({"key": k, "seed": old_row["seed"],
                     "source_center_shift_m": shift,
                     "newer_age_02_required_radius_h04_m": required_new,
                     "older_age_04_required_radius_h04_upper_m": upper,
                     "strictly_clears_40m_by_upper_bound": upper < 40.0})
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    shifts = [row["source_center_shift_m"] for row in rows]
    uppers = [row["older_age_04_required_radius_h04_upper_m"] for row in rows]
    summary = {"paired_states": len(rows),
               "upper_bound_clears_40m": sum(row["strictly_clears_40m_by_upper_bound"] for row in rows),
               "source_center_shift_m": {"min": min(shifts), "median": statistics.median(shifts), "max": max(shifts)},
               "required_radius_upper_m": {"min": min(uppers), "median": statistics.median(uppers), "max": max(uppers)},
               "by_seed": {str(seed): {"states": len(group),
                                      "clear_by_upper_bound": sum(row["strictly_clears_40m_by_upper_bound"]
                                                                  for row in group)}
                           for seed, group in sorted(by_seed.items())}}
    OUTPUT.mkdir(parents=True)
    protocol = {"status": "AGE04_SHORT_PREFIX_IMPLICATION_DEV_V1",
                "plan_sha256": digest(PLAN), "script_sha256": digest(Path(__file__)),
                "prefix_report_sha256": PREFIX_SHA256,
                "original_reachability_report_sha256": ORIGINAL_SHA256,
                "baseline_report_sha256": prior.BASE_REPORT_SHA256,
                "registration_sha256": prior.REGISTRATION_SHA256,
                "actor_speed_bound_mps": 20.0,
                "age_increment_s": 0.2, "prefix_horizon_s": 0.4,
                "hypothetical_complete_radius_m": 40.0,
                "independent_confirmation": False, "real_source_certified": False}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {"status": protocol["status"], "protocol_sha256": digest(OUTPUT / "protocol.json"),
              "summary": summary, "rows": rows,
              "scope": "conditional_collision_prefix_triangle_inequality_only",
              "independent_confirmation": False, "real_source_certified": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "report_sha256": digest(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
