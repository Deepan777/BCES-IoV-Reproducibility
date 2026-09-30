#!/usr/bin/env python3
"""Fresh prefix-only near/benign crossing construction; no policy outcomes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.network.events import NetworkCondition
from bces.utils.hashing import sha256_file


OUTPUT = ROOT / "outputs/study_b/recovery_crossing_prefix_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
HELPER = ROOT / "scripts/86_probe_prefix_eta_solver_dev.py"
HELPER_SHA256 = "a903738895091ce24e21d3af92d57f3801b680583e164a61f54e9724b51f4f9a"
SPEEDS = (11.0, 12.0, 13.0, 14.0)
REPLICATES = (0, 1)
SEED_START = 8600500
REFERENCE_TIME_S = 4.0
POSITION_LOW_M = 0.0
POSITION_HIGH_M = 40.0
NEAR_SIGNED_ETA_S = 0.20
NEAR_TOLERANCE_S = 0.10
BENIGN_ABS_ETA_MIN_S = 1.0


def load_helper():
    if sha256_file(HELPER) != HELPER_SHA256:
        raise RuntimeError("frozen prefix helper changed")
    module_spec = importlib.util.spec_from_file_location("recovery_prefix_helper", HELPER)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load prefix helper")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def approach(row):
    return (str(row.get("actor_lane", "")).startswith("A1B1")
            and str(row.get("ego_lane", "")).startswith("B0B1"))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    helper = load_helper()
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    config = dict(registration["config"])
    config["reference_time_s"] = REFERENCE_TIME_S
    nominal = NetworkCondition(**registration["conditions"]["nominal"])
    seeds = [SEED_START + 2 * speed_index + replicate
             for speed_index in range(len(SPEEDS)) for replicate in REPLICATES]
    protocol = {
        "status": "FROZEN_PREFIX_ONLY_RECOVERY_CROSSING_DEVELOPMENT_V1",
        "registration_sha256": sha256_file(REGISTRATION),
        "runner_sha256": sha256_file(ROOT / "bces/simulation/development_map_loop.py"),
        "visibility_sha256": sha256_file(ROOT / "bces/simulation/development_visibility_map.py"),
        "helper_sha256": HELPER_SHA256,
        "script_sha256": sha256_file(Path(__file__)),
        "seeds": seeds, "speeds_mps": list(SPEEDS),
        "reference_time_s": REFERENCE_TIME_S,
        "actor_position_bracket_m": [POSITION_LOW_M, POSITION_HIGH_M],
        "near_target_signed_eta_s": NEAR_SIGNED_ETA_S,
        "near_tolerance_s": NEAR_TOLERANCE_S,
        "benign_min_absolute_eta_s": BENIGN_ABS_ETA_MIN_S,
        "solver": "two fixed endpoint prefixes, one bounded linear near interpolation, one final near prefix; benign is eligible endpoint with greatest absolute ETA gap, ties choose lower position; no driving outcomes",
        "eligibility": "all 8 seeds require near hit on approach and benign endpoint on approach with at least 1.0-s absolute ETA gap",
        "no_policy_outcomes": True, "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows, artifacts = [], {}
    for speed_index, speed in enumerate(SPEEDS):
        for replicate in REPLICATES:
            seed = SEED_START + 2 * speed_index + replicate
            probes = []
            for index, position in enumerate((POSITION_LOW_M, POSITION_HIGH_M)):
                row, initial = helper.probe(seed, speed, position, index, config, nominal)
                probes.append(row)
                name = f"{seed}_prefix_{index}.json"
                (OUTPUT / name).write_text(json.dumps(initial, sort_keys=True,
                                                     separators=(",", ":")) + "\n", encoding="utf-8")
                artifacts[name] = sha256_file(OUTPUT / name)
            low, high = probes
            low_delta, high_delta = (item.get("signed_eta_difference_s") for item in probes)
            selected = (helper.linear_position(POSITION_LOW_M, low_delta,
                                               POSITION_HIGH_M, high_delta,
                                               NEAR_SIGNED_ETA_S)
                        if low_delta is not None and high_delta is not None else None)
            near_status = "unbracketed_or_missing_prefix"
            if selected is not None:
                near, initial = helper.probe(seed, speed, selected, 2, config, nominal)
                probes.append(near)
                name = f"{seed}_prefix_2.json"
                (OUTPUT / name).write_text(json.dumps(initial, sort_keys=True,
                                                     separators=(",", ":")) + "\n", encoding="utf-8")
                artifacts[name] = sha256_file(OUTPUT / name)
                delta = near.get("signed_eta_difference_s")
                near_status = ("near_hit_on_approach" if delta is not None
                               and abs(delta - NEAR_SIGNED_ETA_S) <= NEAR_TOLERANCE_S
                               and approach(near) else "near_target_or_approach_failed")
            eligible_benign = [(index, row) for index, row in enumerate((low, high))
                               if row.get("signed_eta_difference_s") is not None
                               and abs(row["signed_eta_difference_s"]) >= BENIGN_ABS_ETA_MIN_S
                               and approach(row)]
            benign_index = (max(eligible_benign,
                                key=lambda pair: (abs(pair[1]["signed_eta_difference_s"]),
                                                  -pair[0]))[0]
                            if eligible_benign else None)
            rows.append({"seed": seed, "ego_speed_setting_mps": speed,
                         "replicate": replicate, "near_position_m": selected,
                         "near_status": near_status, "benign_prefix_index": benign_index,
                         "benign_position_m": (POSITION_LOW_M, POSITION_HIGH_M)[benign_index]
                         if benign_index is not None else None,
                         "benign_signed_eta_difference_s": probes[benign_index]["signed_eta_difference_s"]
                         if benign_index is not None else None,
                         "probes": probes})
            print(json.dumps({"completed": len(rows), "total": len(seeds),
                              "near_status": near_status,
                              "benign_eligible": benign_index is not None}), flush=True)
    eligible = [row for row in rows if row["near_status"] == "near_hit_on_approach"
                and row["benign_prefix_index"] is not None]
    report = {"status": "PREFIX_ONLY_RECOVERY_CROSSING_DEVELOPMENT_V1",
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "artifact_sha256": artifacts, "rows": rows,
              "fully_eligible_seeds": len(eligible),
              "fully_eligible_speed_settings": sorted({row["ego_speed_setting_mps"] for row in eligible}),
              "solver_feasible": len(eligible) == len(seeds),
              "no_policy_outcomes": True, "not_confirmatory": True}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("fully_eligible_seeds", "fully_eligible_speed_settings",
                       "solver_feasible")}, indent=2))


if __name__ == "__main__":
    main()
