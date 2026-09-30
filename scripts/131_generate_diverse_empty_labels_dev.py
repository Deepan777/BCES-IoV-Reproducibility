#!/usr/bin/env python3
"""Diverse, resumable development-only labels for the stateless policy."""

from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import importlib.util
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
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import canonical_json_hash, sha256_file

PLAN = ROOT / "docs/STUDY_B_DIVERSE_EMPTY_LABELS_DEV_V1_PLAN.md"
BASE_RUNNER = ROOT / "scripts/127_generate_stateless_empty_labels_dev.py"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_labels_development_v1"
SEEDS = tuple(range(8701000, 8701064))
STRATA = ("absent", "present_near", "present_control")
CONTINUATIONS = (-1.0, 0.0, 1.0)
AGES = (0.0, 0.2, 0.4, 0.8, 1.2, 2.0)


def _base_module():
    spec = importlib.util.spec_from_file_location("empty_labels_pilot_source", BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parameters(seed):
    rng = np.random.default_rng(seed)
    return {
        "ego_speed_mps": float(rng.uniform(9.5, 15.5)),
        "near_position_m": float(rng.uniform(0.0, 22.0)),
        "control_position_m": float(rng.uniform(35.0, 45.0)),
        "actor_speed_mps": float(rng.uniform(8.0, 11.0)),
    }


def _pose_jumps(frames, step_s):
    count = 0
    previous = None
    for timestamp in sorted(frames):
        current = frames[timestamp]["receiver"]
        if current is None:
            previous = None
            continue
        if previous is not None:
            distance = math.hypot(current["x_m"] - previous["x_m"],
                                  current["y_m"] - previous["y_m"])
            bound = max(current["speed_mps"], previous["speed_mps"]) * step_s + 0.5
            count += distance > bound + 1e-8
        previous = current
    return count


def _create_artifact(base, *, seed, stratum, continuation, config, planner_config,
                     thresholds, scales):
    parameters = _parameters(seed)
    position = (parameters["control_position_m"] if stratum == "present_control"
                else parameters["near_position_m"])
    spec = replace(scenario_spec("unprotected_crossing"),
                   actor_depart_position_m=position,
                   actor_depart_speed_mps=parameters["actor_speed_mps"],
                   hidden_from_local=False)
    presence = "absent" if stratum == "absent" else "present"
    frames, lane_jumps = base._trace(
        seed=seed, spec=spec, speed_mps=parameters["ego_speed_mps"],
        continuation_mps2=continuation, presence=presence, config=config)
    planner = DevelopmentStatelessEmptyPlanner(planner_config)
    references, points, abstentions = [], [], []
    try:
        ego, local, packet, premise = base._observed(frames, 3000, config, seed)
        reference, query, cached = base.empty_reference_contract(
            scenario=str(seed), split="diverse_development", behavior="keep",
            timestamp=3000, ego=ego, local=local, message=packet,
            empty_premise=premise, planner=planner, scales=scales)
        references.append(reference)
    except (ValueError, RuntimeError, KeyError) as exc:
        abstentions.append({"age_s": None, "reason": f"{type(exc).__name__}:{exc}"})
        reference = None
    if reference is not None:
        future = {t: f["objects"] for t, f in frames.items()}
        for age_s in AGES:
            timestamp = 3000 + round(age_s * 1000)
            try:
                ego, local, fresh, fresh_premise = base._observed(
                    frames, timestamp, config, seed)
                points.append(base.empty_decision_pair(
                    reference=reference, query=query, cached_reference=cached,
                    ego=ego, local=local, fresh_message=fresh,
                    fresh_empty_premise=fresh_premise, future_world=future,
                    planner=planner, thresholds=thresholds))
            except (ValueError, RuntimeError, KeyError) as exc:
                abstentions.append({"age_s": age_s,
                                     "reason": f"{type(exc).__name__}:{exc}"})
    return {
        "seed": seed, "stratum": stratum,
        "continuation_mps2": continuation,
        "parameters": parameters,
        "scenario_spec": asdict(spec),
        "lane_jump_violations": lane_jumps,
        "pose_jump_count": _pose_jumps(frames, config["step_s"]),
        "references": references, "points": points,
        "abstentions": abstentions,
        "frames": {str(t): {
            "receiver": f["receiver"],
            "objects": [asdict(obj) for obj in f["objects"]],
        } for t, f in frames.items()},
    }


def main():
    if sha256_file(REGISTRATION) != REGISTRATION_SHA256:
        raise RuntimeError("registered confirmation changed")
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    base = _base_module()
    config = dict(registration["config"])
    config["reference_time_s"] = 3.0
    planner_config = PlannerConfig.from_yaml(ROOT / config["planner"])
    config["horizon_s"] = planner_config.horizon_s
    if config["cooperative_range_m"] < 120.0:
        raise RuntimeError("source-view range below development premise")
    thresholds_data = yaml.safe_load((ROOT / config["validity"]).read_text(encoding="utf-8"))
    thresholds = ValidityThresholds(*(
        thresholds_data[key] for key in (
            "max_cost_regret", "max_cached_risk", "max_trajectory_deviation_m")))
    scales = DriftScales(*config["drift_scales"])
    bound_sources = (
        "scripts/127_generate_stateless_empty_labels_dev.py",
        "bces/simulation/development_empty_view_input.py",
        "bces/simulation/development_empty_decision_labels.py",
        "bces/simulation/development_stateless_empty_planner.py",
        "bces/simulation/development_visibility_map.py",
        "bces/simulation/development_map_loop.py",
    )
    protocol = {
        "status": "DIVERSE_EMPTY_LABELS_DEVELOPMENT_ONLY_V1",
        "plan_sha256": sha256_file(PLAN),
        "script_sha256": sha256_file(Path(__file__)),
        "source_sha256": {name: sha256_file(ROOT / name) for name in bound_sources},
        "registration_sha256": REGISTRATION_SHA256,
        "seeds": list(SEEDS), "strata": list(STRATA),
        "continuations_mps2": list(CONTINUATIONS),
        "ages_s": list(AGES),
        "expected_traces": 576,
        "maximum_points": 3456,
        "view_id": base.VIEW_ID,
        "synthetic_ideal_view": True,
        "independent_confirmation": False,
        "no_model_fitted": True,
    }
    if OUTPUT.exists():
        saved = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
        if saved != protocol or (OUTPUT / "report.json").exists():
            raise RuntimeError("cannot resume changed or completed cohort")
    else:
        OUTPUT.mkdir(parents=True)
        (OUTPUT / "protocol.json").write_text(
            json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    artifacts = {}
    records = []
    all_observables = {}
    all_slots = set()
    reference_features = set()
    prefix_by_stratum = {}
    for index, seed in enumerate(SEEDS):
        for stratum in STRATA:
            for continuation in CONTINUATIONS:
                name = f"{seed}_{stratum}_{continuation:+.0f}.json.gz"
                path = OUTPUT / name
                if path.exists():
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        artifact = json.load(handle)
                else:
                    artifact = _create_artifact(
                        base, seed=seed, stratum=stratum,
                        continuation=continuation, config=config,
                        planner_config=planner_config, thresholds=thresholds,
                        scales=scales)
                    with gzip.open(path, "xt", encoding="utf-8") as handle:
                        json.dump(artifact, handle, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False)
                if (artifact["seed"], artifact["stratum"],
                        artifact["continuation_mps2"]) != (seed, stratum, continuation):
                    raise RuntimeError("artifact identity mismatch")
                artifacts[name] = sha256_file(path)
                frames = artifact["frames"]
                prefix = canonical_json_hash({str(t): frames[str(t)]
                                              for t in range(200, 3001, 200)})
                key = (seed, stratum)
                if key in prefix_by_stratum and prefix_by_stratum[key] != prefix:
                    raise RuntimeError("continuations changed the common prefix")
                prefix_by_stratum[key] = prefix
                if artifact["references"]:
                    ref = artifact["references"][0]
                    feature_key = ref["frozen_input"]["feature_sha256"]
                    reference_features.add(feature_key)
                else:
                    feature_key = None
                for point in artifact["points"]:
                    slot_id = f"{name}:{point['current_timestamp_ms']}"
                    if slot_id in all_slots:
                        raise RuntimeError("duplicate global slot identity")
                    all_slots.add(slot_id)
                    key = canonical_json_hash({
                        "reference_feature_sha256": feature_key,
                        "normalized_drift": point["normalized_drift"],
                        "cache_age_s": point["cache_age_s"],
                    })
                    if key in all_observables and all_observables[key][0] != point["valid"]:
                        raise RuntimeError("identical observables have conflicting labels")
                    if key in all_observables and all_observables[key][1] != seed:
                        raise RuntimeError("identical observables cross new seed groups")
                    all_observables[key] = (point["valid"], seed)
                records.append({
                    "name": name, "seed": seed, "stratum": stratum,
                    "continuation_mps2": continuation,
                    "points": len(artifact["points"]),
                    "invalid": sum(not point["valid"] for point in artifact["points"]),
                    "abstentions": len(artifact["abstentions"]),
                    "lane_jumps": len(artifact["lane_jump_violations"]),
                    "pose_jumps": artifact["pose_jump_count"],
                })
        print(json.dumps({"completed_seed_clusters": index + 1,
                          "total_seed_clusters": len(SEEDS)}), flush=True)
    per_seed = {}
    for seed in SEEDS:
        rows = [row for row in records if row["seed"] == seed]
        valid = sum(row["points"] - row["invalid"] for row in rows)
        invalid = sum(row["invalid"] for row in rows)
        per_seed[str(seed)] = {"valid": valid, "invalid": invalid,
                              "mixed": bool(valid and invalid)}
    summary = {
        "traces": len(records), "points": sum(row["points"] for row in records),
        "abstentions": sum(row["abstentions"] for row in records),
        "lane_jumps": sum(row["lane_jumps"] for row in records),
        "pose_jumps": sum(row["pose_jumps"] for row in records),
        "unique_slot_ids": len(all_slots),
        "distinct_reference_features": len(reference_features),
        "distinct_observable_keys": len(all_observables),
        "mixed_validity_seed_clusters": sum(value["mixed"] for value in per_seed.values()),
        "per_seed": per_seed,
    }
    summary["engineering_screen_passed"] = bool(
        summary["traces"] == 576 and summary["lane_jumps"] == 0
        and summary["pose_jumps"] == 0
        and summary["unique_slot_ids"] == summary["points"]
        and summary["mixed_validity_seed_clusters"] >= 20)
    report = {
        "status": protocol["status"],
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "artifact_sha256": artifacts,
        "records": records, "summary": summary,
        "independent_confirmation": False,
        "source_view_certified": False,
        "no_model_fitted": True,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary,
                      "report_sha256": sha256_file(OUTPUT / "report.json")},
                     indent=2))


if __name__ == "__main__":
    main()
