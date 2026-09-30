#!/usr/bin/env python3
"""Exact bound-packet/receiver development replay for the fixed margin ablation.

Only the already-open 40 validation-partition SUMO development scenarios are
read. This is a sampled-slot diagnostic, not online risk or new confirmation.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.bound_policy import FrozenWirePolicy, bind_reference, reference_query
from bces.network.adaptive_margin_policy import select_reference_method
from bces.oracle.observational import BEHAVIOR_IDS
from bces.oracle.world import WorldObject
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundReceiver
from bces.protocol.receiver import DecisionState
from bces.simulation.controlled_contract import make_message, ego_from_sample
from bces.simulation.controlled_decisions import kinematic
from bces.utils.hashing import sha256_file

BASE = ROOT / "outputs/study_b/adaptive_margin_gate_development_v1"
SCORE_AUDIT = ROOT / "outputs/study_b/margin_slot_risk_development_v2"
OUTPUT = ROOT / "outputs/study_b/margin_exact_wire_development_v1"
REGISTRATION = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
CONFIG = ROOT / "configs/evaluation/study_b_controlled_v2.yaml"


def add(store: dict, method: str, scenario: int, family: str, accepted: bool, valid: bool) -> None:
    row = store[method, scenario, family]
    row["points"] += 1
    row["invalid_points"] += not valid
    row["accepted"] += accepted
    row["invalid_accepted"] += accepted and not valid


def totals(store: dict, methods: tuple[str, ...]) -> dict:
    return {method: {key: int(sum(row[key] for (name, _, _), row in store.items() if name == method))
                     for key in ("points", "invalid_points", "accepted", "invalid_accepted")}
            for method in methods}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    scored = json.loads((SCORE_AUDIT / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if len(base["selected"]) != 40 or scored["scenario_count"] != 40:
        raise RuntimeError("wrong development study")
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("registered v2 source changed: " + name)
    protocol = {
        "status": "FROZEN_EXACT_WIRE_DEVELOPMENT_AUDIT_ONLY",
        "base_protocol_sha256": sha256_file(BASE / "protocol.json"),
        "score_report_sha256": sha256_file(SCORE_AUDIT / "report.json"),
        "v2_registration_sha256": sha256_file(REGISTRATION),
        "config_sha256": sha256_file(CONFIG),
        "adaptive_source_sha256": sha256_file(ROOT / "bces/network/adaptive_margin_policy.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "selected_source_sha256": {row["development_source"]: row["development_source_sha256"] for row in base["selected"]},
        "methods": ["surface", "scalar_ttl", "margin_gate", "lead_only"],
        "not_confirmatory": True,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = {name: FrozenWirePolicy(ROOT / rule["path"], expected_sha256=rule["sha256"],
                                     policy_hash=registration["policy_hash"])
              for name, rule in registration["model_rules"].items()}
    methods = tuple(protocol["methods"])
    all_rows = defaultdict(lambda: defaultdict(int))
    low_rows = defaultdict(lambda: defaultdict(int))
    reference_count = point_count = low_reference_count = 0
    for case in base["selected"]:
        path = ROOT / case["development_source"]
        if sha256_file(path) != case["development_source_sha256"]:
            raise RuntimeError("development scene hash changed")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            scene = json.load(handle)
        seed = int(scene["scenario_id"])
        family = scene["family"]
        if seed != case["seed"] or family != case["family"] or scene["split"] != "validation":
            raise RuntimeError("development scene identity mismatch")
        lookup = {(p["behavior"], p["receiver_drift_variant"], p["cache_age_s"]): p for p in scene["points"]}
        if len(lookup) != len(scene["points"]):
            raise RuntimeError("ambiguous scene slot identity")
        used = set()
        for index, reference in enumerate(scene["references"]):
            reference_count += 1
            selected, evidence = select_reference_method(reference)
            lead_like = bool(evidence["lead_like"])
            low = lead_like and evidence["first_competitor_total_cost_gap"] < 0.20
            low_reference_count += low
            t = int(reference["reference_timestamp_ms"])
            frame = scene["traces"]["0"][str(t)]
            ego = ego_from_sample(frame["receiver"])
            objects = tuple(WorldObject(**obj) for obj in frame["objects"])
            message = make_message(tuple(obj for obj in objects if math.hypot(obj.x_m-ego.x_m, obj.y_m-ego.y_m)
                                         <= config["cooperative_range_m"]), t, message_id=seed)
            payload = message.encode()
            query = reference_query(reference, query_id=index+1,
                                    sender_hash=sender_id_hash(message.sender_id))
            behavior = tuple(BEHAVIOR_IDS)[int(query.behavior)]
            receivers = {}
            for name, model in models.items():
                bound = bind_reference(reference, query, payload, model_sha256=model.sha256,
                                       view=model.view, family=name,
                                       shrinkage=float(registration["model_rules"][name]["shrinkage"]))
                wire = bound.encode()
                receiver = BoundReceiver()
                receiver.register(wire)
                receiver.install(model.issue(wire, payload), payload)
                receivers[name] = receiver
            for variant in range(len(config["receiver_drift_variants"])):
                for age in config["ages_s"]:
                    point = lookup.get((behavior, variant, age))
                    if point is None:
                        continue
                    timestamp = t + round(age*1000)
                    sample = scene["traces"][str(variant)][str(timestamp)]["receiver"]
                    if sample is None or point["reference_id"] != reference["reference_id"]:
                        raise RuntimeError("slot/trace identity mismatch")
                    used.add(point["point_id"])
                    current = kinematic(ego_from_sample(sample), timestamp)
                    accepted = {name: receiver.evaluate(current, query.behavior, query.policy_hash,
                                                        message.sender_id).state == DecisionState.ACCEPT_REUSE
                                for name, receiver in receivers.items()}
                    accepted["margin_gate"] = accepted[selected]
                    accepted["lead_only"] = accepted["scalar_ttl" if lead_like else "surface"]
                    valid = bool(point["valid"])
                    for name in methods:
                        add(all_rows, name, seed, family, accepted[name], valid)
                        if low:
                            add(low_rows, name, seed, family, accepted[name], valid)
                    point_count += 1
        if len(used) != len(scene["points"]):
            raise RuntimeError("not all labelled scene slots replayed")
    overall = totals(all_rows, methods)
    low = totals(low_rows, methods)
    paired_scenarios = []
    for case in base["selected"]:
        seed, family = case["seed"], case["family"]
        margin = low_rows["margin_gate", seed, family]
        lead = low_rows["lead_only", seed, family]
        paired_scenarios.append({"seed": seed, "family": family,
                                 "low_margin_points": int(margin["points"]),
                                 "margin_accepted": int(margin["accepted"]),
                                 "margin_invalid": int(margin["invalid_accepted"]),
                                 "lead_only_accepted": int(lead["accepted"]),
                                 "lead_only_invalid": int(lead["invalid_accepted"])})
    result = {
        "status": "EXACT_WIRE_DEVELOPMENT_ONLY",
        "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "scenarios": len(base["selected"]), "references": reference_count,
        "low_margin_lead_references": low_reference_count, "points": point_count,
        "overall": overall, "low_margin_lead": low,
        "low_margin_by_scenario": paired_scenarios,
        "surrogate_summary_match": all(
            overall[name]["accepted"] == scored["policies"][name]["accepted"]
            and overall[name]["invalid_accepted"] == scored["policies"][name]["invalid_accepted"]
            for name in methods),
        "not_confirmatory": True,
    }
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenarios": result["scenarios"], "points": point_count,
                      "overall": overall, "low_margin_lead": low,
                      "surrogate_summary_match": result["surrogate_summary_match"]}, indent=2))


if __name__ == "__main__":
    main()
