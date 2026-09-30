#!/usr/bin/env python3
"""Independent artifact, identity, prefix, and raw-label replay for dev v1."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.oracle.planner import PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.oracle.world import WorldObject
from bces.simulation.controlled_contract import ego_from_sample, make_message
from bces.simulation.development_empty_decision_labels import (
    empty_decision_pair, empty_reference_contract,
)
from bces.simulation.development_empty_view_input import DevelopmentEmptyViewPremise
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner
from bces.simulation.development_visibility_map import visible_local_objects_with_map
from bces.utils.hashing import canonical_json_hash, sha256_file
from scripts._negative_evidence_freshness_support import BLOCKERS

DIR = ROOT / "outputs/study_b/stateless_empty_labels_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"


def observed(frames, timestamp, config, seed, view_id):
    frame = frames[str(timestamp)]
    ego = ego_from_sample(frame["receiver"])
    world = tuple(WorldObject(**item) for item in frame["objects"])
    local = visible_local_objects_with_map(
        world, ego, blockers=BLOCKERS, range_m=config["local_range_m"])
    cooperative = tuple(item for item in world
                        if math.hypot(item.x_m - ego.x_m, item.y_m - ego.y_m)
                        <= config["cooperative_range_m"])
    message = make_message(cooperative, timestamp, message_id=seed)
    premise = None if message.objects else DevelopmentEmptyViewPremise(
        hashlib.sha256(message.encode()).hexdigest(), message.sender_id,
        timestamp, view_id, True)
    return ego, local, message, premise


def main():
    protocol = json.loads((DIR / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((DIR / "report.json").read_text(encoding="utf-8"))
    if sha256_file(DIR / "protocol.json") != report["protocol_sha256"]:
        raise AssertionError("protocol hash mismatch")
    if sha256_file(ROOT / "scripts/127_generate_stateless_empty_labels_dev.py") != protocol["script_sha256"]:
        raise AssertionError("generation source hash mismatch")
    if sha256_file(ROOT / "docs/STUDY_B_STATELESS_EMPTY_LABEL_GENERATION_DEV_V1_PLAN.md") != protocol["plan_sha256"]:
        raise AssertionError("pre-execution plan hash mismatch")
    if sha256_file(REGISTRATION) != protocol["registration_sha256"]:
        raise AssertionError("registered confirmation changed")
    if sha256_file(ROOT / "outputs/study_b/stateless_empty_planner_development_v1/report.json") != protocol["pilot_report_sha256"]:
        raise AssertionError("source pilot report changed")
    for name, digest in protocol["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise AssertionError("bound development source changed: " + name)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise AssertionError("registered source changed: " + name)
    config = dict(registration["config"])
    planner_config = PlannerConfig.from_yaml(ROOT / config["planner"])
    thresholds_data = yaml.safe_load((ROOT / config["validity"]).read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(*(
        thresholds_data[key] for key in (
            "max_cost_regret", "max_cached_risk", "max_trajectory_deviation_m")))
    scales = DriftScales(*config["drift_scales"])
    expected = {
        f"{seed}_{timing}_{presence}_{continuation:+.0f}.json.gz"
        for seed in protocol["seeds"]
        for timing in protocol["timings"]
        for presence in protocol["presence"]
        for continuation in protocol["continuations_mps2"]
    }
    actual = {path.name for path in DIR.glob("*.json.gz")}
    if actual != expected or set(report["artifact_sha256"]) != expected:
        raise AssertionError("trace identity grid incomplete or has extras")
    rows = []
    prefixes = {}
    for name in sorted(expected):
        path = DIR / name
        if sha256_file(path) != report["artifact_sha256"][name]:
            raise AssertionError("artifact hash changed: " + name)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        seed = artifact["seed"]
        timing = artifact["timing"]
        presence = artifact["presence"]
        continuation = artifact["continuation_mps2"]
        if name != f"{seed}_{timing}_{presence}_{continuation:+.0f}.json.gz":
            raise AssertionError("artifact identity mismatch")
        if artifact["lane_jump_violations"] or artifact["abstentions"]:
            raise AssertionError("generator reported failed engineering or abstention")
        frames = artifact["frames"]
        if set(map(int, frames)) != set(range(200, 8001, 200)):
            raise AssertionError("trace timeline incomplete")
        previous = None
        for timestamp in sorted(map(int, frames)):
            receiver = frames[str(timestamp)]["receiver"]
            if receiver is None:
                raise AssertionError("receiver disappeared")
            if receiver["timestamp_ms"] != timestamp:
                raise AssertionError("receiver time mismatch")
            if previous is not None:
                distance = math.hypot(receiver["x_m"] - previous["x_m"],
                                      receiver["y_m"] - previous["y_m"])
                speed_bound = max(receiver["speed_mps"], previous["speed_mps"]) * 0.2 + 0.5
                if distance > speed_bound + 1e-8:
                    raise AssertionError("uncommanded pose jump in saved trace")
            previous = receiver
        prefix_key = (seed, timing, presence)
        prefix_digest = canonical_json_hash({str(t): frames[str(t)]
                                             for t in range(200, 3001, 200)})
        if prefix_key in prefixes and prefixes[prefix_key] != prefix_digest:
            raise AssertionError("acceleration continuations changed the common prefix")
        prefixes[prefix_key] = prefix_digest
        planner = DevelopmentStatelessEmptyPlanner(planner_config)
        ego, local, packet, premise = observed(
            frames, 3000, config, seed, protocol["view_id"])
        ref, query, cached = empty_reference_contract(
            scenario=str(seed), split="development", behavior="keep",
            timestamp=3000, ego=ego, local=local, message=packet,
            empty_premise=premise, planner=planner, scales=scales)
        if len(artifact["references"]) != 1 or canonical_json_hash(ref) != canonical_json_hash(artifact["references"][0]):
            raise AssertionError("reference replay mismatch")
        if len(artifact["points"]) != len(protocol["ages_s"]):
            raise AssertionError("point grid incomplete")
        world = {int(t): tuple(WorldObject(**item) for item in frame["objects"])
                 for t, frame in frames.items()}
        for age, saved in zip(protocol["ages_s"], artifact["points"]):
            timestamp = 3000 + round(age * 1000)
            now_ego, now_local, fresh, fresh_premise = observed(
                frames, timestamp, config, seed, protocol["view_id"])
            point = empty_decision_pair(
                reference=ref, query=query, cached_reference=cached,
                ego=now_ego, local=now_local, fresh_message=fresh,
                fresh_empty_premise=fresh_premise, future_world=world,
                planner=planner, thresholds=thresholds)
            if canonical_json_hash(point) != canonical_json_hash(saved):
                raise AssertionError("point replay mismatch: " + name + ":" + str(age))
        rows.append({
            "name": name, "seed": seed, "timing": timing,
            "presence": presence, "continuation_mps2": continuation,
            "points": len(artifact["points"]), "abstentions": 0,
            "lane_jump_violations": 0,
            "cached_expired_fresh_empty": sum(
                point["cached_policy_selection_sidecar"]["stop_hold_fallback"]
                and point["fresh_policy_selection_sidecar"]["fresh_empty_used"]
                for point in artifact["points"]),
            "physical_payload_points": sum(
                point["fresh_status"] == "objects" for point in artifact["points"]),
        })
    if sorted(rows, key=lambda r: r["name"]) != sorted(report["rows"], key=lambda r: r["name"]):
        raise AssertionError("report rows disagree with raw artifacts")
    summary = report["summary"]
    if (summary["traces"] != len(rows)
            or summary["points"] != sum(row["points"] for row in rows)
            or summary["abstentions"] != 0
            or summary["lane_jump_violations"] != 0
            or summary["absent_near_expired_vs_fresh_pairs"] != sum(
                row["cached_expired_fresh_empty"] for row in rows
                if row["presence"] == "absent" and row["timing"] == "near")
            or summary["present_physical_payload_points"] != sum(
                row["physical_payload_points"] for row in rows
                if row["presence"] == "present")
            or summary["seed_clusters"] != len(protocol["seeds"])
            or not summary["engineering_screen_passed"]):
        raise AssertionError("report summary failed raw replay")
    print(json.dumps({"status": "PASS", "traces": len(rows),
                      "points": summary["points"],
                      "absent_near_expired_vs_fresh_pairs":
                      summary["absent_near_expired_vs_fresh_pairs"],
                      "report_sha256": sha256_file(DIR / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
