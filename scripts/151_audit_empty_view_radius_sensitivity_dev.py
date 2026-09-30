#!/usr/bin/env python3
"""Exact radius rescaling of an already-opened conditional clearance report."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/study_b/empty_disk_reachability_development_v1/report.json"
SOURCE_SHA256 = "5ec3dbea32a51255a544b7ce34de58cda7cbfefe33a7513fe061c8ee88104f33"
PLAN = ROOT / "docs/STUDY_B_EMPTY_VIEW_RADIUS_SENSITIVITY_DEV_V1_PLAN.md"
OUTPUT = ROOT / "outputs/study_b/empty_view_radius_sensitivity_dev_v1"
RADII_M = (20, 40, 60, 80, 100, 120)
AGES_S = (0.2, 0.4)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows, age, radius):
    subset = [row for row in rows if row["age_s"] == age]
    by_seed = defaultdict(list)
    required = []
    for row in subset:
        slack = row["minimum_slack_m"]
        clear = bool(row["candidate_available"] and slack is not None
                     and slack - (120.0 - radius) > 0.0)
        by_seed[int(row["seed"])].append(clear)
        if slack is not None:
            required.append(120.0 - slack)
        if radius == 120 and clear != row["continuous_conditional_clearance"]:
            raise RuntimeError("saved 120-m clearance flag not reproduced")
    if not subset or len(by_seed) != 8:
        raise RuntimeError("unexpected development denominator")
    count = sum(sum(values) for values in by_seed.values())
    sorted_required = sorted(required)
    return {
        "states": len(subset), "clear_states": count,
        "clear_fraction": count / len(subset),
        "seeds_at_least_70pct_clear": sum(sum(values) / len(values) >= 0.7
                                            for values in by_seed.values()),
        "by_seed": {str(seed): {"states": len(values), "clear_states": sum(values),
                                "fraction": sum(values) / len(values)}
                    for seed, values in sorted(by_seed.items())},
        "minimum_required_radius_m_over_available_candidates": {
            "minimum": min(sorted_required),
            "median": statistics.median(sorted_required),
            "maximum": max(sorted_required),
        },
    }


def main():
    if OUTPUT.exists():
        raise RuntimeError("output exists; preserve first run")
    if digest(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("frozen reachability report changed")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source["assumptions"]["complete_empty_ego_centered_source_disk_radius_m"] != 120.0:
        raise RuntimeError("expected original 120-m source radius")
    rows = source["rows"]
    results = {}
    for age in AGES_S:
        results[str(age)] = {str(radius): summary(rows, age, radius)
                             for radius in RADII_M}
        old = source["summary"][str(age)]
        check = results[str(age)]["120"]
        if (check["states"] != old["approach_states"]
                or check["clear_states"] != old["conditional_clear_states"]
                or check["seeds_at_least_70pct_clear"] != old["seeds_at_least_70pct_clear"]):
            raise RuntimeError("saved 120-m summary not reproduced")
    OUTPUT.mkdir(parents=True)
    protocol = {"status": "EMPTY_VIEW_RADIUS_SENSITIVITY_DEV_V1",
                "plan_sha256": digest(PLAN), "script_sha256": digest(Path(__file__)),
                "source_report_sha256": SOURCE_SHA256,
                "radii_m": list(RADII_M), "ages_s": list(AGES_S),
                "rescaling_identity": "new_slack=old_120m_slack-(120-new_radius)",
                "independent_confirmation": False, "real_source_view_certified": False}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {"status": protocol["status"],
              "protocol_sha256": digest(OUTPUT / "protocol.json"),
              "results": results,
              "scope": "ideal_ego_centered_empty_disk_geometry_sensitivity_only",
              "independent_confirmation": False, "real_source_view_certified": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": {age: {radius: {"n": item["states"],
                                                       "clear": item["clear_states"]}
                                          for radius, item in radii.items()}
                                 for age, radii in results.items()},
                      "report_sha256": digest(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
