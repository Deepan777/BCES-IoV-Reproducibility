#!/usr/bin/env python3
"""Development-only attribution of legal-route ambiguity in the map tube."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import route_for
from bces.simulation.scenario_builder import scenario_spec
from bces.utils.hashing import sha256_file

SOURCE = ROOT / "scripts/113_audit_map_reachable_action_feasibility_dev.py"
SOURCE_SHA256 = "13d5ac8db73739e8ce7b0d76e44f537cba1290ac151fbbb1c367f265d440c4e5"
BASELINE = ROOT / "outputs/study_b/map_reachable_action_feasibility_development_v1/report.json"
BASELINE_SHA256 = "b24bf20e6bc1b72f681aa3dc02e138b130f9bd7d19e19e42feac416ea458b987"
OUTPUT = ROOT / "outputs/study_b/crossing_route_information_development_v1"


def counts_by_seed(rows, field):
    out = {}
    for seed in sorted({row["seed"] for row in rows}):
        subset = [row for row in rows if row["seed"] == seed]
        available = sum(row[field] > 0 for row in subset)
        out[str(seed)] = {"states": len(subset), "available_states": available,
                          "fraction": available / len(subset)}
    return out


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha256_file(SOURCE) != SOURCE_SHA256 or sha256_file(BASELINE) != BASELINE_SHA256:
        raise RuntimeError("frozen map audit or its report changed")
    spec = importlib.util.spec_from_file_location("frozen_map_audit", SOURCE)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    if sha256_file(audit.MAP) != audit.MAP_SHA256:
        raise RuntimeError("registered map changed")
    if sha256_file(audit.PREFLIGHT) != audit.PREFLIGHT_SHA256:
        raise RuntimeError("lane preflight changed")
    if sha256_file(audit.REGISTRATION) != audit.REGISTRATION_SHA256:
        raise RuntimeError("v2 registration changed")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    preflight = json.loads(audit.PREFLIGHT.read_text(encoding="utf-8"))
    registration = json.loads(audit.REGISTRATION.read_text(encoding="utf-8"))
    branch_report = json.loads((audit.BASE / "report.json").read_text(encoding="utf-8"))
    for name, digest in branch_report["artifact_sha256"].items():
        if sha256_file(audit.BASE / name) != digest:
            raise RuntimeError("saved development branch changed: " + name)
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered source changed: " + name)
    scenario = scenario_spec("unprotected_crossing")
    if scenario.actor_route != ("A1B1", "B1C1"):
        raise RuntimeError("scenario-generator route differs from fixed attribution design")
    planner = KinematicPlanner(PlannerConfig.from_yaml(ROOT / registration["config"]["planner"]))
    map_root = ET.parse(audit.MAP).getroot()
    matched_by_seed = {row["seed"]: row["matching_lanes"][0] for row in preflight["rows"]}
    baseline_rows = {(row["seed"], row["condition"], row["timestamp_ms"]): row
                     for row in baseline["rows"]}
    rows = []
    for seed in sorted(matched_by_seed):
        for condition in sorted({row["condition"] for row in branch_report["rows"]}):
            filename = f"{seed}_near_{condition}_original_periodic_payload.json.gz"
            with gzip.open(audit.BASE / filename, "rt", encoding="utf-8") as handle:
                branch = json.load(handle)
            actor = audit.actor_at_reference(branch["initial_contract"]["snapshot"])
            paths = audit.paths_from_map(map_root, matched_by_seed[seed], actor["position"])
            straight = [item for item in paths if item["successor"]["to_edge"] == "B1C1"]
            right = [item for item in paths if item["successor"]["to_edge"] == "B1B0"]
            if len(paths) != 2 or len(straight) != 1 or len(right) != 1:
                raise RuntimeError("map continuation set differs from fixed attribution design")
            visibility = {item["timestamp_ms"]: item for item in branch["visibility_audit"]}
            for sample in branch["rows"]:
                timestamp = sample["timestamp_ms"]
                if (timestamp < 4200 or timestamp > 6000 or "receiver" not in sample
                        or visibility[timestamp]["target_locally_visible"]):
                    continue
                key = seed, condition, timestamp
                saved = baseline_rows[key]
                ego = ego_from_sample(sample["receiver"])
                route = route_for(ego, scenario.intended_behavior, planner)
                age = (timestamp - 4000) / 1000.0
                all_legal = audit.candidate_counts(planner, ego, route, paths, actor, age)
                straight_only = audit.candidate_counts(planner, ego, route, straight, actor, age)
                right_only = audit.candidate_counts(planner, ego, route, right, actor, age)
                if all_legal["map_tube_nonoverlap_count"] != saved["map_tube_nonoverlap_count"]:
                    raise RuntimeError("all-legal count failed frozen baseline check")
                rows.append({"seed": seed, "condition": condition, "timestamp_ms": timestamp,
                             "all_legal_count": all_legal["map_tube_nonoverlap_count"],
                             "generator_route_straight_only_count": straight_only["map_tube_nonoverlap_count"],
                             "counterfactual_right_only_count": right_only["map_tube_nonoverlap_count"]})
    if set(baseline_rows) != {(r["seed"], r["condition"], r["timestamp_ms"]) for r in rows}:
        raise RuntimeError("cohort changed")
    fields = ("all_legal_count", "generator_route_straight_only_count",
              "counterfactual_right_only_count")
    summary = {}
    for field in fields:
        by_seed = counts_by_seed(rows, field)
        summary[field] = {"available_states": sum(row[field] > 0 for row in rows),
                          "seeds_with_at_least_80pct_available":
                              sum(v["fraction"] >= 0.8 for v in by_seed.values()),
                          "by_seed": by_seed}
    summary["route_ambiguity_is_sole_screen_obstacle"] = (
        summary["generator_route_straight_only_count"]["available_states"] / len(rows) >= 0.9
        and summary["generator_route_straight_only_count"]["seeds_with_at_least_80pct_available"] >= 6
        and not baseline["summary"]["development_usefulness_gate"])
    report = {"status": "DEVELOPMENT_ONLY_ROUTE_INFORMATION_ATTRIBUTION_V1",
              "script_sha256": sha256_file(Path(__file__)),
              "map_audit_sha256": SOURCE_SHA256,
              "baseline_report_sha256": BASELINE_SHA256,
              "scenario_generator_route_used_label_only": list(scenario.actor_route),
              "evaluated_states": len(rows), "summary": summary, "rows": rows,
              "limitations": ["generator route is unavailable to current receiver and used only as an oracle attribution label",
                              "single-route geometries are not valid for a receiver without transmitted route intent",
                              "opened synthetic development states, not independent confirmation",
                              "no motion-bound validation, wire accounting, policy change, or BCES-vs-TTL result"]}
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evaluated_states": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
