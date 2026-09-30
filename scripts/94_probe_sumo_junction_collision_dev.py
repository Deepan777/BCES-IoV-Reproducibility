#!/usr/bin/env python3
"""Development-only A/B probe of SUMO's junction collision-reporting flag."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.network.events import NetworkCondition
from bces.simulation import development_map_loop
from bces.simulation.controlled_contract import compare_snapshots
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.simulation.sumo_adapter import load_traci, sumo_binary
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/sumo_junction_collision_probe_dev_v1"
SOURCE = ROOT / "outputs/study_b/recovery_crossing_driving_development_v1"
SOURCE_REPORT_SHA256 = "13196de214f47cd38b9cd321b2337d77ab46b414b1a22ae33f53ff500da21d9c"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
SEED = 8600501
METHODS = ("periodic_payload", "local_only")
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)


def start_with_junction_check(*, network, label, seed, step_s, state=None):
    command = [str(sumo_binary()), "-n", str(network), "--step-length", str(step_s),
               "--seed", str(seed), "--no-step-log", "true", "--no-warnings", "true",
               "--time-to-teleport", "-1", "--collision.action", "warn",
               "--collision.check-junctions", "true",
               "--save-state.rng", "true", "--save-state.transportables", "true",
               "--save-state.precision", "12", "--thread-rngs", "1"]
    if state is not None:
        command.extend(["--load-state", str(state)])
    traci = load_traci()
    traci.start(command, label=label)
    return traci.getConnection(label)


@contextmanager
def junction_collision_enabled():
    original = development_map_loop.start_controlled
    try:
        development_map_loop.start_controlled = start_with_junction_check
        yield
    finally:
        development_map_loop.start_controlled = original


def compare_rows(original, rerun):
    first = {row["timestamp_ms"]: row for row in original["rows"] if "receiver" in row}
    second = {row["timestamp_ms"]: row for row in rerun["rows"] if "receiver" in row}
    if set(first) != set(second):
        return {"same_timestamps": False, "maximum_center_error_m": None,
                "maximum_speed_error_mps": None}
    center_errors = [math.hypot(first[t]["receiver"]["x_m"] - second[t]["receiver"]["x_m"],
                                first[t]["receiver"]["y_m"] - second[t]["receiver"]["y_m"])
                     for t in first]
    speed_errors = [abs(first[t]["receiver"]["speed_mps"] - second[t]["receiver"]["speed_mps"])
                    for t in first]
    return {"same_timestamps": True,
            "maximum_center_error_m": max(center_errors, default=0.0),
            "maximum_speed_error_mps": max(speed_errors, default=0.0)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE / "report.json") != SOURCE_REPORT_SHA256:
        raise RuntimeError("source development report changed")
    source = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    selected = [item for item in source["rows"]
                if item["seed"] == SEED and item["timing"] == "near"
                and item["condition"] == "nominal" and item["planner"] == "original"]
    if len(selected) != 1:
        raise RuntimeError("wrong source scenario selection")
    selected = selected[0]
    config = dict(registration["config"])
    config["reference_time_s"] = 4.0
    network = NetworkCondition(**registration["conditions"]["nominal"])
    spec = replace(scenario_spec("unprotected_crossing"),
                   actor_depart_position_m=selected["actor_depart_position_m"],
                   actor_depart_speed_mps=9.0, hidden_from_local=False)
    parameters = {"benchmark_stratum": "prefix_eta_solver_development_v1",
                  "ego_initial_speed_mps": selected["ego_initial_speed_mps"],
                  "ego_post_reference_acceleration_mps2": 0.0,
                  "braking_event_delay_s": 0.2}
    factory = lambda _seed, _config: (spec, parameters)
    protocol = {"status": "FROZEN_DEVELOPMENT_SUMO_JUNCTION_COLLISION_PROBE",
                "registration_sha256": sha256_file(REGISTRATION),
                "source_report_sha256": SOURCE_REPORT_SHA256,
                "runner_sha256": sha256_file(Path(__file__)),
                "development_loop_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
                "seed": SEED, "condition": "nominal", "planner": "original",
                "timing": "near", "methods": list(METHODS),
                "only_simulator_flag_change": "--collision.check-junctions true",
                "expected": "same checkpoint, timestamps, ego pose and speed to 1e-6; local-only has SUMO junction collision tick and periodic does not",
                "not_new_policy_or_confirmation": True}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    comparisons, artifacts = [], {}
    for method in METHODS:
        old_name = f"{SEED}_near_nominal_original_{method}.json.gz"
        if sha256_file(SOURCE / old_name) != source["artifact_sha256"][old_name]:
            raise RuntimeError("source branch changed: " + old_name)
        with gzip.open(SOURCE / old_name, "rt", encoding="utf-8") as handle:
            old = json.load(handle)
        with junction_collision_enabled():
            new = development_map_loop.run_branch(
                seed=SEED, method=method, config=config, condition=network,
                network_seed=SEED + 970000,
                horizon_s=registration["horizon_s"], scenario_factory=factory,
                map_blockers=BLOCKERS)
        filename = f"{SEED}_near_nominal_original_{method}_junction_check.json.gz"
        with gzip.open(OUTPUT / filename, "xt", encoding="utf-8") as handle:
            json.dump(new, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        artifacts[filename] = sha256_file(OUTPUT / filename)
        prefix_equal = compare_snapshots(old["initial_contract"]["snapshot"],
                                         new["initial_contract"]["snapshot"],
                                         tolerance=1e-6)["equal_within_tolerance"]
        comparisons.append({"method": method, "original_artifact": old_name,
                            "new_artifact": filename,
                            "prefix_equal": prefix_equal,
                            **compare_rows(old, new),
                            "original_sumo_collision_times_ms": [row["timestamp_ms"] for row in old["rows"]
                                                                  if "ego" in row["sumo_colliding_vehicle_ids"]],
                            "new_sumo_collision_times_ms": [row["timestamp_ms"] for row in new["rows"]
                                                             if "ego" in row["sumo_colliding_vehicle_ids"]],
                            "original_geometric_overlap_times_ms": [row["timestamp_ms"] for row in old["rows"]
                                                                     if "receiver" in row and row["events"]["geometric_overlap_ids"]],
                            "new_geometric_overlap_times_ms": [row["timestamp_ms"] for row in new["rows"]
                                                                if "receiver" in row and row["events"]["geometric_overlap_ids"]],
                            "new_actuation_contract_passed": new["actuation_contract_passed"]})
    expected = all(row["prefix_equal"] and row["same_timestamps"]
                   and row["maximum_center_error_m"] <= 1e-6
                   and row["maximum_speed_error_mps"] <= 1e-6
                   and row["new_actuation_contract_passed"] for row in comparisons)
    local = next(row for row in comparisons if row["method"] == "local_only")
    periodic = next(row for row in comparisons if row["method"] == "periodic_payload")
    report = {"status": "DEVELOPMENT_SUMO_JUNCTION_COLLISION_PROBE",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "comparisons": comparisons,
              "trajectory_replay_match": expected,
              "junction_collision_reporting_explained": bool(expected
                   and local["new_sumo_collision_times_ms"]
                   and not periodic["new_sumo_collision_times_ms"]),
              "not_confirmatory": True, "manuscript_allowed": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"trajectory_replay_match": report["trajectory_replay_match"],
                      "junction_collision_reporting_explained": report["junction_collision_reporting_explained"],
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
