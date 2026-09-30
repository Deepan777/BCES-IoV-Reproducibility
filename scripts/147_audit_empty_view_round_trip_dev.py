#!/usr/bin/env python3
"""Optimistic exact-frame timing screen for 0.2-s empty-view evidence."""

from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.bound_policy import reference_query
from bces.models.development_empty_wire_policy import DevelopmentEmptyWirePolicy
from bces.network.events import NetworkCondition
from bces.network.wire_channel import WireChannel, refresh_control
from bces.oracle.world import WorldObject
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundQuery
from bces.simulation.controlled_contract import ego_from_sample, make_message
from bces.utils.hashing import sha256_file

PLAN = ROOT / "docs/STUDY_B_EMPTY_VIEW_ROUND_TRIP_DEV_V1_PLAN.md"
SOURCE = ROOT / "outputs/study_b/diverse_empty_labels_development_v2"
MODELS = ROOT / "outputs/study_b/diverse_empty_transfer_models_dev_v1"
OUTPUT = ROOT / "outputs/study_b/empty_view_round_trip_dev_v1"
SOURCE_SHA256 = "1ddfc3480b01f29df6152feac4f815ed135d711ca9005e2e479e04aaf5126983"
MODELS_SHA256 = "4575c74428dfe115495a2b1f8c0a6058c1096c6c1eb22294487be0d55d2076f6"
REFERENCE_MS = 3000
STEP_MS = 200
MAXIMUM_EMPTY_AGE_MS = 200
SENDER_COMPUTATION_MS = 6.0
SECURITY_BYTES = 64


def _empty_reference(config, source_report):
    for name, digest in sorted(source_report["artifact_sha256"].items()):
        if "_absent_" not in name:
            continue
        path = SOURCE / name
        if sha256_file(path) != digest:
            raise RuntimeError("raw reference changed")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        frame = artifact["frames"][str(REFERENCE_MS)]
        ego = ego_from_sample(frame["receiver"])
        world = tuple(WorldObject(**item) for item in frame["objects"])
        cooperative = tuple(item for item in world
                            if math.hypot(item.x_m - ego.x_m, item.y_m - ego.y_m)
                            <= config["cooperative_range_m"])
        message = make_message(cooperative, REFERENCE_MS, message_id=artifact["seed"])
        if message.objects:
            continue
        reference = artifact["references"][0]
        if hashlib.sha256(message.encode()).hexdigest() != reference["frozen_input"]["payload_sha256"]:
            raise RuntimeError("source payload does not match saved reference")
        return name, reference, message
    raise RuntimeError("no exact empty development reference")


def _delivery(channel, kind):
    event = channel.next_delivery(1_000_000.0)
    if event is None or event.kind != kind:
        raise RuntimeError("optimistic network failed to deliver " + kind)
    return event


def _first_decision_tick(delivered_ms):
    ticks = max(1, math.ceil((delivered_ms - REFERENCE_MS) / STEP_MS - 1e-12))
    return REFERENCE_MS + ticks * STEP_MS


def _bound_path(condition, policy, reference, payload, sender_id):
    channel = WireChannel(condition, seed=20260925, security_bytes=SECURITY_BYTES)
    channel.send("refresh", refresh_control(sender_id_hash(sender_id), 0), REFERENCE_MS)
    refresh = _delivery(channel, "refresh")
    # Latest actually observed 0.2-s source frame at this receipt is 3.0 s.
    if not REFERENCE_MS <= refresh.delivered_ms < REFERENCE_MS + STEP_MS:
        raise RuntimeError("source sample identity requires another frame")
    channel.send("payload", payload, refresh.delivered_ms)
    message_event = _delivery(channel, "payload")
    generated_query_ms = math.ceil(message_event.delivered_ms)
    query = reference_query(reference, query_id=1, sender_hash=sender_id_hash(sender_id))
    batch = reference["frozen_input"]["batch"]
    bound = BoundQuery(
        query, hashlib.sha256(payload).hexdigest(), policy.sha256,
        "frozen_inputs", policy.family, 0.0, (),
        reference["frozen_input"]["context_available_at_ms"],
        batch["occlusion_proxy"], batch["estimated_delay_s"],
        batch["map_context_flags"], generated_query_ms, generated_query_ms,
    )
    query_wire = bound.encode()
    channel.send("query", query_wire, generated_query_ms)
    query_event = _delivery(channel, "query")
    response = policy.issue(query_wire, payload)
    response_generated_ms = query_event.delivered_ms + SENDER_COMPUTATION_MS
    channel.send("response", response, response_generated_ms)
    reply = _delivery(channel, "response")
    tick = _first_decision_tick(reply.delivered_ms)
    report = channel.report()
    if not report["byte_conservation_ok"] or not report["component_conservation_ok"]:
        raise RuntimeError("network byte conservation failed")
    return {
        "refresh_delivered_ms": refresh.delivered_ms,
        "payload_generated_ms": refresh.delivered_ms,
        "payload_delivered_ms": message_event.delivered_ms,
        "query_generated_ms": generated_query_ms,
        "query_delivered_ms": query_event.delivered_ms,
        "response_generated_ms": response_generated_ms,
        "response_delivered_ms": reply.delivered_ms,
        "first_decision_tick_ms": tick,
        "payload_age_at_first_decision_ms": tick - REFERENCE_MS,
        "within_200ms_empty_gate": tick - REFERENCE_MS <= MAXIMUM_EMPTY_AGE_MS,
        "application_bytes": {
            "refresh": len(refresh.body), "payload": len(payload),
            "query": len(query_wire), "response": len(response),
        },
        "generated_wire_bytes": report["generated_bytes"],
        "byte_conservation_ok": report["byte_conservation_ok"],
        "component_conservation_ok": report["component_conservation_ok"],
    }


def _direct_push(condition, payload):
    channel = WireChannel(condition, seed=20260925, security_bytes=SECURITY_BYTES)
    channel.send("payload", payload, REFERENCE_MS)
    event = _delivery(channel, "payload")
    tick = _first_decision_tick(event.delivered_ms)
    report = channel.report()
    return {
        "payload_delivered_ms": event.delivered_ms,
        "first_decision_tick_ms": tick,
        "payload_age_at_first_decision_ms": tick - REFERENCE_MS,
        "within_200ms_empty_gate": tick - REFERENCE_MS <= MAXIMUM_EMPTY_AGE_MS,
        "application_payload_bytes": len(payload),
        "generated_wire_bytes": report["generated_bytes"],
        "byte_conservation_ok": report["byte_conservation_ok"],
    }


def main():
    if OUTPUT.exists():
        raise RuntimeError("timing audit output exists; preserve first run")
    if (sha256_file(SOURCE / "report.json") != SOURCE_SHA256
            or sha256_file(MODELS / "report.json") != MODELS_SHA256):
        raise RuntimeError("source cohort/model freeze changed")
    source_report = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    model_report = json.loads((MODELS / "report.json").read_text(encoding="utf-8"))
    registration = json.loads((ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json")
                              .read_text(encoding="utf-8"))
    name, reference, message = _empty_reference(registration["config"], source_report)
    payload = message.encode()
    policies = {}
    for family in ("surface_teacher", "scalar_ttl_teacher"):
        path = MODELS / f"{family}.pt"
        digest = model_report["models"][family]["checkpoint_sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        policies[family] = DevelopmentEmptyWirePolicy(
            path, expected_sha256=digest, policy_hash=saved["policy_hash"])
    results = {}
    for name_condition in registration["condition_order"]:
        original = NetworkCondition(**registration["conditions"][name_condition])
        optimistic = replace(original, loss_probability=0.0, jitter_ms=0.0,
                             duplicate_probability=0.0, reorder_probability=0.0,
                             gilbert_good_to_bad=0.0, gilbert_bad_loss=0.0)
        methods = {family: _bound_path(optimistic, policy, reference, payload,
                                       message.sender_id) for family, policy in policies.items()}
        methods["periodic_direct_payload"] = _direct_push(optimistic, payload)
        results[name_condition] = methods
    OUTPUT.mkdir(parents=True)
    protocol = {
        "status": "OPTIMISTIC_EMPTY_VIEW_BOUND_RTT_DEV_V1",
        "plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
        "source_report_sha256": SOURCE_SHA256, "model_report_sha256": MODELS_SHA256,
        "registration_sha256": sha256_file(
            ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"),
        "reference_artifact": name,
        "maximum_empty_age_ms": MAXIMUM_EMPTY_AGE_MS,
        "step_ms": STEP_MS, "sender_computation_ms": SENDER_COMPUTATION_MS,
        "security_bytes_per_frame": SECURITY_BYTES,
        "loss_jitter_duplication_reordering_removed_for_best_case": True,
        "independent_confirmation": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    assessment = {
        "bound_surface_nominal_can_use_empty": results["nominal"]["surface_teacher"]["within_200ms_empty_gate"],
        "bound_ttl_nominal_can_use_empty": results["nominal"]["scalar_ttl_teacher"]["within_200ms_empty_gate"],
        "bound_surface_impaired_can_use_empty": results["impaired"]["surface_teacher"]["within_200ms_empty_gate"],
        "bound_ttl_impaired_can_use_empty": results["impaired"]["scalar_ttl_teacher"]["within_200ms_empty_gate"],
        "direct_nominal_can_use_empty": results["nominal"]["periodic_direct_payload"]["within_200ms_empty_gate"],
        "direct_impaired_can_use_empty": results["impaired"]["periodic_direct_payload"]["within_200ms_empty_gate"],
    }
    report = {
        "status": protocol["status"], "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
        "results": results, "assessment": assessment,
        "scope": "optimistic_modelled_timing_necessary_condition_only",
        "source_view_certified": False, "independent_confirmation": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results, "assessment": assessment,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
