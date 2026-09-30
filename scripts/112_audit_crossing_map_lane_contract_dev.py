#!/usr/bin/env python3
"""Causal map-lane ambiguity preflight on opened crossing prefixes."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file


NET = ROOT / "sumo/registered_grid_v1.net.xml"
NET_SHA256 = "476cd09dd964e2f55258e9730f96bbdc0048ba4c2dd357016141e6a386127cfb"
SOURCE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
SOURCE_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
OUTPUT = ROOT / "outputs/study_b/crossing_map_lane_contract_development_v1"
MAX_LATERAL_DISTANCE_M = 2.5
MAX_HEADING_DIFFERENCE_RAD = 0.35


def polyline(shape):
    return tuple(tuple(map(float, item.split(","))) for item in shape.split())


def nearest_segment(x, y, points):
    best = None
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= 0:
            continue
        fraction = max(0.0, min(1.0,
                                ((x - start[0]) * dx + (y - start[1]) * dy) / length_sq))
        distance = math.hypot(x - start[0] - fraction * dx,
                              y - start[1] - fraction * dy)
        heading = math.atan2(dy, dx)
        if best is None or distance < best[0]:
            best = distance, heading
    if best is None:
        raise RuntimeError("map lane has no nonzero segment")
    return best


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(NET) != NET_SHA256 or sha256_file(SOURCE / "report.json") != SOURCE_SHA256:
        raise RuntimeError("registered map or development prefix source changed")
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    root = ET.parse(NET).getroot()
    lanes = {}
    for edge in root.findall("edge"):
        if edge.get("function") == "internal":
            continue
        for lane in edge.findall("lane"):
            lanes[lane.attrib["id"]] = {"edge": edge.attrib["id"],
                                          "index": lane.attrib["index"],
                                          "shape": polyline(lane.attrib["shape"])}
    connections = {}
    for conn in root.findall("connection"):
        key = (conn.attrib["from"], conn.attrib["fromLane"])
        connections.setdefault(key, []).append({"to_edge": conn.attrib["to"],
                                                   "to_lane_index": conn.attrib["toLane"],
                                                   "via_internal_lane": conn.attrib.get("via"),
                                                   "turn_code": conn.attrib.get("dir")})
    rows = []
    for seed in sorted({row["seed"] for row in source["rows"]}):
        name = f"{seed}_near_nominal_original_periodic_payload.json.gz"
        if sha256_file(SOURCE / name) != source["artifact_sha256"][name]:
            raise RuntimeError("prefix branch changed: " + name)
        with gzip.open(SOURCE / name, "rt", encoding="utf-8") as handle:
            branch = json.load(handle)
        actor = branch["initial_contract"]["snapshot"]["vehicles"]["cross"]
        heading = math.radians(90.0 - actor["heading"])
        x = actor["position"][0] - 0.5 * actor["length"] * math.cos(heading)
        y = actor["position"][1] - 0.5 * actor["length"] * math.sin(heading)
        matches = []
        for lane_id, lane in lanes.items():
            distance, lane_heading = nearest_segment(x, y, lane["shape"])
            heading_error = abs((heading - lane_heading + math.pi) % (2 * math.pi) - math.pi)
            if (distance <= MAX_LATERAL_DISTANCE_M
                    and heading_error <= MAX_HEADING_DIFFERENCE_RAD):
                matches.append({"lane_id": lane_id, "lateral_distance_m": distance,
                                "heading_error_rad": heading_error,
                                "successors": connections.get((lane["edge"], lane["index"]), [])})
        rows.append({"seed": seed, "reference_time_ms": 4000,
                     "sender_position_xy_m": [x, y],
                     "sender_heading_rad": heading,
                     "matching_lanes": sorted(matches, key=lambda item: item["lane_id"]),
                     "actual_future_route_not_read": True})
    structural_gate = (len(rows) == 8
                       and all(len(row["matching_lanes"]) == 1
                               and 1 <= len(row["matching_lanes"][0]["successors"]) <= 3
                               for row in rows))
    result = {"status": "CAUSAL_MAP_LANE_CONTRACT_DEVELOPMENT_ONLY_V1",
              "net_sha256": NET_SHA256, "prefix_report_sha256": SOURCE_SHA256,
              "script_sha256": sha256_file(Path(__file__)),
              "max_lateral_distance_m": MAX_LATERAL_DISTANCE_M,
              "max_heading_difference_rad": MAX_HEADING_DIFFERENCE_RAD,
              "eligible_prefixes": len(rows),
              "structural_lane_and_successor_gate": structural_gate,
              "rows": rows,
              "limitations": ["ideal sender position/heading from simulator prefix, not recorded receiver packet",
                              "lane consistency is an unvalidated motion assumption",
                              "future route intentionally not used; all map successors retained",
                              "this structural result does not prove reachable-tube containment or robust-action feasibility"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prefixes": len(rows),
                      "structural_gate": structural_gate,
                      "matched_lanes": [row["matching_lanes"][0]["lane_id"]
                                        if len(row["matching_lanes"]) == 1 else None for row in rows],
                      "successor_counts": [len(row["matching_lanes"][0]["successors"])
                                           if len(row["matching_lanes"]) == 1 else None for row in rows]},
                     indent=2))


if __name__ == "__main__":
    main()
