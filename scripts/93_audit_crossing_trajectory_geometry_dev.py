#!/usr/bin/env python3
"""Independent oriented-box replay of saved development trajectories.

Uses corner projections, not the planner's rectangle-overlap implementation.
Linear interpolation is an approximation between 0.2-s SUMO samples and
cannot turn modeled overlap into an observed real-world collision.
"""

from __future__ import annotations

from collections import Counter
import gzip
import json
import math
from pathlib import Path

from bces.utils.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
SOURCE_REPORT_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
OUTPUT = ROOT / "outputs/study_b/recovery_crossing_trajectory_audit_v1"
STEP_MS = 20


def corners(x, y, heading, length, width):
    fx, fy = math.cos(heading), math.sin(heading)
    sx, sy = -fy, fx
    return tuple((x + a * length / 2 * fx + b * width / 2 * sx,
                  y + a * length / 2 * fy + b * width / 2 * sy)
                 for a in (-1, 1) for b in (-1, 1))


def independent_penetration(a, b):
    """Minimum signed SAT overlap of independently constructed corners."""
    ac = corners(*a)
    bc = corners(*b)
    axes = []
    for box in (ac, bc):
        for i, j in ((0, 2), (0, 1)):
            dx, dy = box[j][0] - box[i][0], box[j][1] - box[i][1]
            length = math.hypot(dx, dy)
            axes.append((-dy / length, dx / length))
    penetration = math.inf
    for ux, uy in axes:
        aa = [x * ux + y * uy for x, y in ac]
        bb = [x * ux + y * uy for x, y in bc]
        depth = min(max(aa), max(bb)) - max(min(aa), min(bb))
        if depth < 0:
            return depth
        penetration = min(penetration, depth)
    return penetration


def actor_heading(positions, timestamp):
    times = sorted(positions)
    index = times.index(timestamp)
    before = times[max(0, index - 1)]
    after = times[min(len(times) - 1, index + 1)]
    if before == after:
        return None
    dx = positions[after][0] - positions[before][0]
    dy = positions[after][1] - positions[before][1]
    if math.hypot(dx, dy) < 1e-8:
        return None
    return math.atan2(dy, dx)


def interpolate_angle(a, b, fraction):
    delta = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + fraction * delta


def branch_audit(branch):
    actor = branch["initial_contract"]["snapshot"]["vehicles"]["cross"]
    actor_length, actor_width = actor["length"], actor["width"]
    actor_xy = {item["timestamp_ms"]: item["target_xy_m"]
                for item in branch["visibility_audit"] if item["target_xy_m"] is not None}
    rows = {row["timestamp_ms"]: row for row in branch["rows"] if "receiver" in row}
    shared = sorted(set(rows) & set(actor_xy))
    if not shared:
        raise RuntimeError("no aligned actor/ego states")
    sampled_positive, sampled_replayed = set(), set()
    fine_positive_times = set()
    depths = []
    for timestamp in shared:
        row = rows[timestamp]
        if row["events"]["geometric_overlap_ids"]:
            sampled_positive.add(timestamp)
        heading = actor_heading(actor_xy, timestamp)
        if heading is None:
            continue
        ego = row["receiver"]
        a = (ego["x_m"], ego["y_m"], ego["heading_rad"],
             ego["length_m"], ego["width_m"])
        b = (*actor_xy[timestamp], heading, actor_length, actor_width)
        penetration = independent_penetration(a, b)
        if penetration >= 0:
            sampled_replayed.add(timestamp)
            fine_positive_times.add(timestamp)
            depths.append(penetration)
    for first, second in zip(shared, shared[1:]):
        if second - first != 200:
            continue
        left, right = rows[first]["receiver"], rows[second]["receiver"]
        left_heading, right_heading = actor_heading(actor_xy, first), actor_heading(actor_xy, second)
        if left_heading is None or right_heading is None:
            continue
        for timestamp in range(first + STEP_MS, second, STEP_MS):
            fraction = (timestamp - first) / (second - first)
            ego = (left["x_m"] + fraction * (right["x_m"] - left["x_m"]),
                   left["y_m"] + fraction * (right["y_m"] - left["y_m"]),
                   interpolate_angle(left["heading_rad"], right["heading_rad"], fraction),
                   left["length_m"], left["width_m"])
            actor_box = (actor_xy[first][0] + fraction * (actor_xy[second][0] - actor_xy[first][0]),
                         actor_xy[first][1] + fraction * (actor_xy[second][1] - actor_xy[first][1]),
                         interpolate_angle(left_heading, right_heading, fraction),
                         actor_length, actor_width)
            penetration = independent_penetration(ego, actor_box)
            if penetration >= 0:
                fine_positive_times.add(timestamp)
                depths.append(penetration)
    return {"source_sampled_overlap_ticks": sorted(sampled_positive),
            "independent_sampled_overlap_ticks": sorted(sampled_replayed),
            "interpolated_overlap_sample_count": len(fine_positive_times),
            "interpolated_overlap_first_ms": min(fine_positive_times) if fine_positive_times else None,
            "interpolated_overlap_last_ms": max(fine_positive_times) if fine_positive_times else None,
            "interpolated_max_min_axis_penetration_m": max(depths) if depths else None,
            "sumo_ego_collision_ticks": branch["counts"].get("sumo_ego_collision_ticks", 0),
            "aligned_0p2s_states": len(shared)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_REPORT_SHA256:
        raise RuntimeError("development source report changed")
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    if source["branches"] != 128 or len(source["artifact_sha256"]) != 128:
        raise RuntimeError("wrong source cohort")
    results = []
    for name, digest in sorted(source["artifact_sha256"].items()):
        if sha256_file(SOURCE / name) != digest:
            raise RuntimeError("source branch changed: " + name)
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        results.append({"source_artifact": name, **branch_audit(branch)})
    mismatches = [row["source_artifact"] for row in results
                  if row["source_sampled_overlap_ticks"] != row["independent_sampled_overlap_ticks"]]
    positives = [row for row in results if row["interpolated_overlap_sample_count"]]
    unsampled = [row["source_artifact"] for row in positives
                 if not row["source_sampled_overlap_ticks"]]
    summary = {"status": "DEVELOPMENT_ONLY_INDEPENDENT_GEOMETRY_REPLAY",
               "source_report_sha256": SOURCE_REPORT_SHA256,
               "script_sha256": sha256_file(Path(__file__)),
               "interpolation_step_ms": STEP_MS,
               "branches": len(results),
               "source_sampled_positive_branches": sum(bool(row["source_sampled_overlap_ticks"]) for row in results),
               "independent_sampled_positive_branches": sum(bool(row["independent_sampled_overlap_ticks"]) for row in results),
               "interpolated_positive_branches": len(positives),
               "sampled_tick_mismatches": mismatches,
               "interpolated_only_positive_branches": unsampled,
               "sumo_ego_collision_branches": sum(bool(row["sumo_ego_collision_ticks"]) for row in results),
               "rows": results,
               "not_real_world_validation": True, "not_confirmatory": True,
               "manuscript_allowed": False}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("branches", "source_sampled_positive_branches",
                       "independent_sampled_positive_branches",
                       "interpolated_positive_branches", "sampled_tick_mismatches",
                       "interpolated_only_positive_branches", "sumo_ego_collision_branches")}, indent=2))


if __name__ == "__main__":
    main()
