#!/usr/bin/env python3
"""Outcome-blind SUMO prefix solver for crossing arrival-time construction."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.network.events import NetworkCondition
from bces.simulation.development_map_loop import run_branch
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/prefix_eta_solver_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)
SPEEDS = (11.0, 12.0, 13.0, 14.0)
REPLICATES = (0, 1)
TARGET_DELTA_S = 0.20
POSITION_LOW_M = 0.0
POSITION_HIGH_M = 30.0
TOLERANCE_S = 0.10


def linear_position(low_position: float, low_delta: float,
                    high_position: float, high_delta: float,
                    target: float) -> float | None:
    """Interpolate only inside a monotone, finite bracket; never extrapolate."""
    values = (low_position, low_delta, high_position, high_delta, target)
    if (not all(math.isfinite(value) for value in values)
            or low_position >= high_position or low_delta >= high_delta
            or not low_delta <= target <= high_delta):
        return None
    return low_position + (target - low_delta) * (high_position - low_position) / (high_delta - low_delta)


def probe(seed: int, speed: float, position: float, index: int,
          config: dict, condition: NetworkCondition) -> tuple[dict, dict]:
    spec = replace(scenario_spec("unprotected_crossing"),
                   actor_depart_position_m=position,
                   actor_depart_speed_mps=9.0, hidden_from_local=False)
    parameters = {"benchmark_stratum": "prefix_eta_solver_development_v1",
                  "ego_initial_speed_mps": speed,
                  "ego_post_reference_acceleration_mps2": 0.0,
                  "braking_event_delay_s": 0.2}
    branch = run_branch(seed=seed, method="local_only", config=config,
                        condition=condition, network_seed=seed + 940000 + index,
                        horizon_s=0.0,
                        scenario_factory=lambda _seed, _config: (spec, parameters),
                        map_blockers=BLOCKERS)
    if branch["rows"] or branch["duration_s"] != 0.0:
        raise RuntimeError("prefix probe unexpectedly ran a decision branch")
    initial = branch["initial_contract"]
    vehicles = initial["snapshot"]["vehicles"]
    row = {"seed": seed, "ego_speed_setting_mps": speed,
           "actor_depart_position_m": position, "probe_index": index,
           "prefix_time_s": initial["snapshot"]["time_s"],
           "ego_present": "ego" in vehicles, "actor_present": "cross" in vehicles}
    if "ego" in vehicles and "cross" in vehicles:
        ego, actor = vehicles["ego"], vehicles["cross"]
        row.update({"ego_lane": ego["lane"], "actor_lane": actor["lane"],
                    "ego_xy_m": ego["position"], "actor_xy_m": actor["position"],
                    "ego_speed_mps": ego["speed"], "actor_speed_mps": actor["speed"]})
        if ego["speed"] > 0 and actor["speed"] > 0:
            ex, ey = ego["position"]
            ax, ay = actor["position"]
            ego_eta = (ay - ey) / ego["speed"]
            actor_eta = (ex - ax) / actor["speed"]
            row.update({"ego_eta_s": ego_eta, "actor_eta_s": actor_eta,
                        "signed_eta_difference_s": ego_eta - actor_eta})
    return row, initial


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    protocol = {
        "status": "FROZEN_PREFIX_ONLY_DEVELOPMENT_ENGINEERING",
        "registration_sha256": sha256_file(REGISTRATION),
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility_map.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "seeds": [8600200 + 2 * i + replicate
                  for i, _ in enumerate(SPEEDS) for replicate in REPLICATES],
        "speeds_mps": list(SPEEDS), "replicates": list(REPLICATES),
        "target_signed_eta_difference_s": TARGET_DELTA_S,
        "position_bracket_m": [POSITION_LOW_M, POSITION_HIGH_M],
        "solver": "probe fixed 0 and 30 m positions at each seed/speed; if finite monotone deltas bracket +0.20 s, linearly interpolate one final position and run one final prefix probe; never select by post-branch outcome",
        "success_rule": "at least 6/8 final prefixes on A1B1 with absolute signed ETA error <=0.10 s, spanning at least 3 ego speeds",
        "all_failed_brackets_and_absent_actors_retained": True,
        "no_policy_outcomes": True, "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    condition = NetworkCondition(**registration["conditions"]["nominal"])
    rows, artifacts = [], {}
    for speed_index, speed in enumerate(SPEEDS):
        for replicate in REPLICATES:
            seed = 8600200 + 2 * speed_index + replicate
            probes = []
            for index, position in enumerate((POSITION_LOW_M, POSITION_HIGH_M)):
                row, initial = probe(seed, speed, position, index,
                                     registration["config"], condition)
                probes.append(row)
                name = f"{seed}_prefix_{index}.json"
                (OUTPUT / name).write_text(json.dumps(initial, sort_keys=True,
                                                     separators=(",", ":")) + "\n", encoding="utf-8")
                artifacts[name] = sha256_file(OUTPUT / name)
            low, high = probes
            low_delta, high_delta = (item.get("signed_eta_difference_s") for item in probes)
            selected = (linear_position(POSITION_LOW_M, low_delta,
                                        POSITION_HIGH_M, high_delta, TARGET_DELTA_S)
                        if low_delta is not None and high_delta is not None else None)
            if selected is None:
                status = "unbracketed_or_missing_prefix"
            else:
                final, initial = probe(seed, speed, selected, 2,
                                       registration["config"], condition)
                probes.append(final)
                name = f"{seed}_prefix_2.json"
                (OUTPUT / name).write_text(json.dumps(initial, sort_keys=True,
                                                     separators=(",", ":")) + "\n", encoding="utf-8")
                artifacts[name] = sha256_file(OUTPUT / name)
                delta = final.get("signed_eta_difference_s")
                status = ("target_hit_on_approach" if delta is not None
                          and abs(delta - TARGET_DELTA_S) <= TOLERANCE_S
                          and str(final.get("actor_lane", "")).startswith("A1B1")
                          and str(final.get("ego_lane", "")).startswith("B0B1")
                          else "target_or_approach_failed")
            rows.append({"seed": seed, "ego_speed_setting_mps": speed,
                         "replicate": replicate, "selected_position_m": selected,
                         "status": status, "probes": probes})
            print(json.dumps({"completed": len(rows), "total": 8,
                              "status": status}), flush=True)
    hits = [row for row in rows if row["status"] == "target_hit_on_approach"]
    report = {"status": "PREFIX_ONLY_DEVELOPMENT_ENGINEERING",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "target_hits": len(hits),
              "hit_speeds": sorted({row["ego_speed_setting_mps"] for row in hits}),
              "solver_feasible": len(hits) >= 6 and len({row["ego_speed_setting_mps"] for row in hits}) >= 3,
              "no_policy_outcomes": True, "not_confirmatory": True}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("target_hits", "hit_speeds", "solver_feasible")}, indent=2))


if __name__ == "__main__":
    main()
