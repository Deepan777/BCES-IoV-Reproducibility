#!/usr/bin/env python3
"""Outcome-blind AV2 trajectory input/route/label availability audit."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file


DATA = ROOT / "data/external/av2_motion_schema_v1"
OUTPUT = ROOT / "outputs/study_b/av2_causal_contract_development_v3"
REFERENCES = (10, 20, 30, 35)
OFFSETS = (5, 10)
RADIUS_M = 120.0
FUTURE_FRAMES = 20
DYNAMIC = {"vehicle", "bus", "motorcyclist", "cyclist", "pedestrian"}


def angle_difference(a: float, b: float) -> float:
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def lane_match(ego: dict, lanes: dict) -> dict:
    x, y = float(ego["position_x"]), float(ego["position_y"])
    heading = float(ego["heading"])
    candidates = []
    for key, lane in lanes.items():
        if lane.get("lane_type") != "VEHICLE" or lane.get("is_intersection"):
            continue
        points = lane.get("centerline") or []
        best = None
        for first, second in zip(points, points[1:]):
            dx, dy = second["x"] - first["x"], second["y"] - first["y"]
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                continue
            tangent = math.atan2(dy, dx)
            if angle_difference(heading, tangent) > math.pi / 6:
                continue
            fraction = max(0.0, min(1.0, ((x - first["x"]) * dx +
                                          (y - first["y"]) * dy) / length_sq))
            distance = math.hypot(x - first["x"] - fraction * dx,
                                  y - first["y"] - fraction * dy)
            best = distance if best is None else min(best, distance)
        if best is not None:
            candidates.append((best, key, lane))
    candidates.sort(key=lambda item: item[0])
    if not candidates or candidates[0][0] > 1.6:
        return {"status": "no_aligned_lane"}
    distance, key, lane = candidates[0]
    if len(candidates) > 1 and candidates[1][0] - distance < 1.0:
        return {"status": "ambiguous_lane"}
    successors = lane.get("successors") or []
    if len(successors) != 1 or str(successors[0]) not in lanes:
        return {"status": "route_successor_unknown"}
    return {"status": "unique_keep_corridor", "lane_id": str(key),
            "lane_distance_m": distance}


def actor_ids_at(by_track: dict, t: int, ego: dict) -> set[str]:
    result = set()
    for track_id, states in by_track.items():
        if track_id == "AV" or t not in states:
            continue
        row = states[t]
        if str(row["object_type"]).lower() not in DYNAMIC:
            continue
        if math.hypot(float(row["position_x"] - ego["position_x"]),
                      float(row["position_y"] - ego["position_y"])) <= RADIUS_M:
            result.add(track_id)
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_audit = ROOT / "outputs/study_b/av2_motion_schema_preflight_v1/report.json"
    schema = json.loads(source_audit.read_text(encoding="utf-8"))
    if schema["status"] != "SCHEMA_ONLY_NO_POLICY_OUTCOMES" or len(schema["scenes"]) != 3:
        raise ValueError("fixed schema preflight missing")
    assets = {item["key"].split("/")[-1]: item["sha256"] for item in schema["file_audit"]}
    rows = []
    for scene in schema["scenes"]:
        scene_id = scene["scene_id"]
        directory = DATA / scene_id
        track_path = directory / f"scenario_{scene_id}.parquet"
        map_path = directory / f"log_map_archive_{scene_id}.json"
        if sha256_file(track_path) != assets[track_path.name] or sha256_file(map_path) != assets[map_path.name]:
            raise ValueError("source asset hash changed")
        frame = pq.read_table(track_path).to_pandas()
        map_data = json.loads(map_path.read_text(encoding="utf-8"))
        by_track = {str(track_id): {int(item["timestep"]): item
                                    for item in group.to_dict("records")}
                    for track_id, group in frame.groupby("track_id", sort=False)}
        av = by_track.get("AV", {})
        for reference in REFERENCES:
            for offset in OFFSETS:
                decision = reference + offset
                row = {"scene_id": scene_id, "reference": reference,
                       "decision": decision, "offset_frames": offset}
                required_ego = set(range(reference - 2, decision + 1))
                if (not required_ego <= av.keys() or
                    any(not bool(av[t]["observed"]) for t in required_ego)):
                    row["status"] = "ego_observed_history_missing"
                    rows.append(row)
                    continue
                route = lane_match(av[reference], map_data["lane_segments"])
                row["route_status"] = route["status"]
                if route["status"] != "unique_keep_corridor":
                    row["status"] = route["status"]
                    rows.append(row)
                    continue
                reference_actors = actor_ids_at(by_track, reference, av[reference])
                decision_actors = actor_ids_at(by_track, decision, av[decision])
                row["reference_actor_count"] = len(reference_actors)
                row["decision_actor_count"] = len(decision_actors)
                if not reference_actors:
                    row["status"] = "no_reference_remote_actors"
                elif any(not {reference - 2, reference - 1, reference} <= by_track[track].keys()
                         for track in reference_actors):
                    row["status"] = "reference_actor_history_missing"
                elif any(not {decision - 2, decision - 1, decision} <= by_track[track].keys()
                         for track in decision_actors):
                    row["status"] = "decision_actor_history_missing"
                elif not set(range(decision + 1, decision + FUTURE_FRAMES + 1)) <= av.keys():
                    row["status"] = "future_ego_absent"
                else:
                    label_times = set(range(decision + 2, decision + FUTURE_FRAMES + 1, 2))
                    missing = sum(not label_times <= by_track[track].keys()
                                  for track in decision_actors)
                    row["future_actor_tracks_incomplete"] = missing
                    row["status"] = ("complete_structural_contract" if missing == 0
                                     else "future_actor_tracks_incomplete")
                rows.append(row)
    report = {
        "status": "DEVELOPMENT_CAUSAL_CONTRACT_ONLY_NO_POLICY_OUTCOMES",
        "source_schema_report_sha256": sha256_file(source_audit),
        "source_sha256": sha256_file(Path(__file__)),
        "candidate_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "scene_status_counts": {scene["scene_id"]: dict(Counter(
            row["status"] for row in rows if row["scene_id"] == scene["scene_id"]))
                                for scene in schema["scenes"]},
        "rows": rows,
        "scientific_success": False,
        "manuscript_allowed": False,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(rows),
                      "status_counts": report["status_counts"]}, indent=2))


if __name__ == "__main__":
    main()


