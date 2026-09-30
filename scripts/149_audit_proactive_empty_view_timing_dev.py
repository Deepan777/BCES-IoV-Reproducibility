#!/usr/bin/env python3
"""Optimistic payload-first timing sensitivity for short empty-view evidence."""

from __future__ import annotations

from dataclasses import replace
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
from bces.network.wire_channel import WireChannel
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundQuery
from bces.utils.hashing import sha256_file
import importlib

previous = importlib.import_module("scripts.147_audit_empty_view_round_trip_dev")
PLAN = ROOT / "docs/STUDY_B_PROACTIVE_EMPTY_VIEW_TIMING_DEV_V1_PLAN.md"
OUTPUT = ROOT / "outputs/study_b/proactive_empty_view_timing_dev_v1"


def _path(condition, policy, reference, payload, sender_id):
    channel = WireChannel(condition, seed=20260925, security_bytes=previous.SECURITY_BYTES)
    channel.send("payload", payload, previous.REFERENCE_MS)
    message_event = previous._delivery(channel, "payload")
    generated_query_ms = math.ceil(message_event.delivered_ms)
    query = reference_query(reference, query_id=1, sender_hash=sender_id_hash(sender_id))
    batch = reference["frozen_input"]["batch"]
    bound = BoundQuery(query, hashlib.sha256(payload).hexdigest(), policy.sha256,
                       "frozen_inputs", policy.family, 0.0, (),
                       reference["frozen_input"]["context_available_at_ms"],
                       batch["occlusion_proxy"], batch["estimated_delay_s"],
                       batch["map_context_flags"], generated_query_ms, generated_query_ms)
    query_wire = bound.encode()
    channel.send("query", query_wire, generated_query_ms)
    query_event = previous._delivery(channel, "query")
    response = policy.issue(query_wire, payload)
    response_generated_ms = query_event.delivered_ms + previous.SENDER_COMPUTATION_MS
    channel.send("response", response, response_generated_ms)
    reply = previous._delivery(channel, "response")
    tick = previous._first_decision_tick(reply.delivered_ms)
    report = channel.report()
    if not report["byte_conservation_ok"] or not report["component_conservation_ok"]:
        raise RuntimeError("wire byte conservation failed")
    return {"payload_delivered_ms": message_event.delivered_ms,
            "query_generated_ms": generated_query_ms,
            "query_delivered_ms": query_event.delivered_ms,
            "response_generated_ms": response_generated_ms,
            "response_delivered_ms": reply.delivered_ms,
            "first_decision_tick_ms": tick,
            "payload_age_at_first_decision_ms": tick - previous.REFERENCE_MS,
            "within_200ms_empty_gate": tick - previous.REFERENCE_MS <= previous.MAXIMUM_EMPTY_AGE_MS,
            "application_bytes": {"payload": len(payload), "query": len(query_wire), "response": len(response)},
            "generated_wire_bytes": report["generated_bytes"],
            "byte_conservation_ok": report["byte_conservation_ok"],
            "component_conservation_ok": report["component_conservation_ok"]}


def main():
    if OUTPUT.exists():
        raise RuntimeError("sensitivity output already exists; preserve first run")
    if (sha256_file(previous.SOURCE / "report.json") != previous.SOURCE_SHA256
            or sha256_file(previous.MODELS / "report.json") != previous.MODELS_SHA256):
        raise RuntimeError("source cohort/model freeze changed")
    source_report = json.loads((previous.SOURCE / "report.json").read_text(encoding="utf-8"))
    model_report = json.loads((previous.MODELS / "report.json").read_text(encoding="utf-8"))
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    artifact_name, reference, message = previous._empty_reference(registration["config"], source_report)
    payload = message.encode()
    policies = {}
    for family in ("surface_teacher", "scalar_ttl_teacher"):
        path = previous.MODELS / f"{family}.pt"
        digest = model_report["models"][family]["checkpoint_sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        policies[family] = DevelopmentEmptyWirePolicy(path, expected_sha256=digest,
                                                       policy_hash=saved["policy_hash"])
    results = {}
    for condition_name in registration["condition_order"]:
        original = NetworkCondition(**registration["conditions"][condition_name])
        optimistic = replace(original, loss_probability=0.0, jitter_ms=0.0,
                             duplicate_probability=0.0, reorder_probability=0.0,
                             gilbert_good_to_bad=0.0, gilbert_bad_loss=0.0)
        methods = {family: _path(optimistic, policy, reference, payload, message.sender_id)
                   for family, policy in policies.items()}
        methods["periodic_direct_payload"] = previous._direct_push(optimistic, payload)
        results[condition_name] = methods
    OUTPUT.mkdir(parents=True)
    protocol = {"status": "PROACTIVE_EMPTY_VIEW_TIMING_DEV_V1",
                "plan_sha256": sha256_file(PLAN), "script_sha256": sha256_file(Path(__file__)),
                "source_report_sha256": previous.SOURCE_SHA256,
                "model_report_sha256": previous.MODELS_SHA256,
                "registration_sha256": sha256_file(registration_path),
                "reference_artifact": artifact_name,
                "reference_ms": previous.REFERENCE_MS, "step_ms": previous.STEP_MS,
                "maximum_empty_age_ms": previous.MAXIMUM_EMPTY_AGE_MS,
                "sender_computation_ms": previous.SENDER_COMPUTATION_MS,
                "security_bytes_per_frame": previous.SECURITY_BYTES,
                "optimistic_loss_and_jitter_removed": True,
                "independent_confirmation": False, "source_view_certified": False}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {"status": protocol["status"],
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "results": results,
              "scope": "optimistic_modelled_timing_necessary_condition_only",
              "full_run_communication_accounted": False,
              "independent_confirmation": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"results": results,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
