#!/usr/bin/env python3
"""Independent hash, identity, prefix, pose, and full-label replay of dev cohort."""

from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
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

DIR = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"


def _parameters(seed):
    rng = np.random.default_rng(seed)
    return {
        "ego_speed_mps": float(rng.uniform(9.5, 15.5)),
        "near_position_m": float(rng.uniform(0.0, 22.0)),
        "control_position_m": float(rng.uniform(35.0, 45.0)),
        "actor_speed_mps": float(rng.uniform(8.0, 11.0)),
    }


def _observed(frames, timestamp, *, config, seed, view_id):
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
        raise AssertionError("protocol changed")
    if sha256_file(ROOT / "scripts/131_generate_diverse_empty_labels_dev.py") != protocol["script_sha256"]:
        raise AssertionError("generator source changed")
    if sha256_file(ROOT / "docs/STUDY_B_DIVERSE_EMPTY_LABELS_DEV_V1_PLAN.md") != protocol["plan_sha256"]:
        raise AssertionError("pre-execution plan changed")
    if sha256_file(REGISTRATION) != protocol["registration_sha256"]:
        raise AssertionError("registered confirmation changed")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise AssertionError("registered source changed: " + name)
    for name, digest in protocol["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise AssertionError("development source changed: " + name)
    expected = {
        f"{seed}_{stratum}_{continuation:+.0f}.json.gz"
        for seed in protocol["seeds"]
        for stratum in protocol["strata"]
        for continuation in protocol["continuations_mps2"]
    }
    if ({path.name for path in DIR.glob("*.json.gz")} != expected
            or set(report["artifact_sha256"]) != expected):
        raise AssertionError("576-trace artifact grid incomplete")
    config = registration["config"]
    planner_config = PlannerConfig.from_yaml(ROOT / config["planner"])
    thresholds_data = yaml.safe_load((ROOT / config["validity"]).read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(*(
        thresholds_data[key] for key in (
            "max_cost_regret", "max_cached_risk", "max_trajectory_deviation_m")))
    scales = DriftScales(*config["drift_scales"])
    summary_records = []
    prefixes = {}
    slot_ids = set()
    observable_keys = {}
    reference_keys = set()
    per_seed = defaultdict(lambda: {"valid": 0, "invalid": 0})
    for name in sorted(expected):
        path = DIR / name
        if sha256_file(path) != report["artifact_sha256"][name]:
            raise AssertionError("artifact hash changed: " + name)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        seed = artifact["seed"]
        stratum = artifact["stratum"]
        continuation = artifact["continuation_mps2"]
        if name != f"{seed}_{stratum}_{continuation:+.0f}.json.gz":
            raise AssertionError("artifact identity mismatch")
        if artifact["parameters"] != _parameters(seed):
            raise AssertionError("seed parameters changed")
        if artifact["lane_jump_violations"] or artifact["pose_jump_count"] or artifact["abstentions"]:
            raise AssertionError("generator recorded engineering failure/abstention")
        frames = artifact["frames"]
        if set(map(int, frames)) != set(range(200, 8001, 200)):
            raise AssertionError("trace timeline incomplete")
        previous = None
        for timestamp in sorted(map(int, frames)):
            receiver = frames[str(timestamp)]["receiver"]
            if receiver is None or receiver["timestamp_ms"] != timestamp:
                raise AssertionError("receiver missing or time-mismatched")
            if previous is not None:
                distance = math.hypot(receiver["x_m"] - previous["x_m"],
                                      receiver["y_m"] - previous["y_m"])
                limit = max(receiver["speed_mps"], previous["speed_mps"]) * 0.2 + 0.5
                if distance > limit + 1e-8:
                    raise AssertionError("uncommanded ego pose jump")
            previous = receiver
        key = (seed, stratum)
        prefix = canonical_json_hash({str(t): frames[str(t)]
                                      for t in range(200, 3001, 200)})
        if key in prefixes and prefixes[key] != prefix:
            raise AssertionError("receiver continuation changed common prefix")
        prefixes[key] = prefix
        ego, local, message, premise = _observed(
            frames, 3000, config=config, seed=seed,
            view_id=protocol["view_id"])
        planner = DevelopmentStatelessEmptyPlanner(planner_config)
        ref, query, cached = empty_reference_contract(
            scenario=str(seed), split="diverse_development", behavior="keep",
            timestamp=3000, ego=ego, local=local, message=message,
            empty_premise=premise, planner=planner, scales=scales)
        if len(artifact["references"]) != 1 or canonical_json_hash(ref) != canonical_json_hash(artifact["references"][0]):
            raise AssertionError("reference replay mismatch")
        feature_key = ref["frozen_input"]["feature_sha256"]
        reference_keys.add(feature_key)
        if len(artifact["points"]) != len(protocol["ages_s"]):
            raise AssertionError("age grid incomplete")
        future = {int(t): tuple(WorldObject(**item) for item in frame["objects"])
                  for t, frame in frames.items()}
        for age, saved in zip(protocol["ages_s"], artifact["points"]):
            timestamp = 3000 + round(age * 1000)
            now_ego, now_local, fresh, fresh_premise = _observed(
                frames, timestamp, config=config, seed=seed,
                view_id=protocol["view_id"])
            point = empty_decision_pair(
                reference=ref, query=query, cached_reference=cached,
                ego=now_ego, local=now_local, fresh_message=fresh,
                fresh_empty_premise=fresh_premise, future_world=future,
                planner=planner, thresholds=thresholds)
            if canonical_json_hash(point) != canonical_json_hash(saved):
                raise AssertionError("point replay mismatch: " + name + ":" + str(age))
            slot_id = f"{name}:{timestamp}"
            if slot_id in slot_ids:
                raise AssertionError("duplicate global slot")
            slot_ids.add(slot_id)
            observable = canonical_json_hash({
                "reference_feature_sha256": feature_key,
                "normalized_drift": point["normalized_drift"],
                "cache_age_s": point["cache_age_s"],
            })
            if observable in observable_keys and observable_keys[observable] != (seed, point["valid"]):
                raise AssertionError("observable collision or label conflict")
            observable_keys[observable] = (seed, point["valid"])
            per_seed[seed]["valid" if point["valid"] else "invalid"] += 1
        summary_records.append({
            "name": name, "seed": seed, "stratum": stratum,
            "continuation_mps2": continuation,
            "points": len(artifact["points"]),
            "invalid": sum(not point["valid"] for point in artifact["points"]),
            "abstentions": 0, "lane_jumps": 0, "pose_jumps": 0,
        })
    if sorted(summary_records, key=lambda item: item["name"]) != sorted(
            report["records"], key=lambda item: item["name"]):
        raise AssertionError("report records differ from raw artifacts")
    summary = report["summary"]
    mixed = sum(bool(row["valid"] and row["invalid"]) for row in per_seed.values())
    if (len(summary_records) != summary["traces"]
            or len(summary_records) != 576
            or len(slot_ids) != summary["points"]
            or len(reference_keys) != summary["distinct_reference_features"]
            or len(observable_keys) != summary["distinct_observable_keys"]
            or mixed != summary["mixed_validity_seed_clusters"]
            or not summary["engineering_screen_passed"]):
        raise AssertionError("summary mismatch")
    for seed, counts in per_seed.items():
        if summary["per_seed"][str(seed)] != {
                **counts, "mixed": bool(counts["valid"] and counts["invalid"])}:
            raise AssertionError("seed-cluster balance mismatch")
    print(json.dumps({"status": "PASS", "traces": len(summary_records),
                      "points": len(slot_ids), "unique_observables": len(observable_keys),
                      "mixed_seed_clusters": mixed,
                      "report_sha256": sha256_file(DIR / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
