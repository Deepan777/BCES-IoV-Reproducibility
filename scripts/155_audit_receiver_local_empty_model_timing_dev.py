#!/usr/bin/env python3
"""Prospective development audit of locally preinstalled BCES/TTL inference."""

from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.models.bound_policy import reference_query
from bces.models.development_empty_wire_policy import (
    DevelopmentEmptyWirePolicy, raw_reference_vector, raw_wire_features,
)
from bces.network.events import NetworkCondition
from bces.network.wire_channel import WireChannel
from bces.oracle.world import WorldObject
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundQuery
from bces.simulation.controlled_contract import ego_from_sample, make_message
from bces.utils.hashing import sha256_file

previous = importlib.import_module("scripts.145_audit_empty_teacher_wire_parity_dev")
timing = importlib.import_module("scripts.147_audit_empty_view_round_trip_dev")
PLAN = ROOT / "docs/STUDY_B_RECEIVER_LOCAL_EMPTY_MODEL_TIMING_DEV_V1_PLAN.md"
OUTPUT = ROOT / "outputs/study_b/receiver_local_empty_model_timing_dev_v1"
PREVIOUS_SHA256 = "13e17be6c59c772165a6816694d044e675b4efe5adb966ac5bfc06dfb213d847"
REGISTRATION_SHA256 = "5da6fd09206d0bc70ef4b882bbedae5f55de5e97cb03e0dfe1642a3b3e57a1a6"
FAMILIES = ("surface_teacher", "scalar_ttl_teacher")
SEED_START = 260926000
SEED_COUNT = 4096
LOCAL_BUDGET_MS = 20.0


def _local_bound(reference, payload, policy, sender_id, received_ms):
    batch = reference["frozen_input"]["batch"]
    query = reference_query(reference, query_id=1, sender_hash=sender_id_hash(sender_id))
    return BoundQuery(
        query=query, payload_sha256=hashlib.sha256(payload).hexdigest(),
        model_sha256=policy.sha256, feature_view="frozen_inputs",
        family=policy.family, shrinkage=0.0, margins=(),
        context_available_ms=reference["frozen_input"]["context_available_at_ms"],
        occlusion_proxy=batch["occlusion_proxy"],
        estimated_delay_s=batch["estimated_delay_s"],
        map_context_flags=batch["map_context_flags"],
        generated_ms=received_ms, payload_received_ms=received_ms,
    )


def _one_delivery(condition, payload, seed):
    channel = WireChannel(condition, seed=seed, security_bytes=timing.SECURITY_BYTES)
    channel.send("payload", payload, timing.REFERENCE_MS)
    first = None
    while True:
        event = channel.next_delivery(1_000_000.0)
        if event is None:
            break
        if first is None:
            first = event.delivered_ms
    report = channel.report()
    if not report["byte_conservation_ok"] or not report["component_conservation_ok"]:
        raise RuntimeError("channel byte conservation failed")
    if report["generated_packets"] not in (1, 2):
        raise RuntimeError("single send made unexpected packet count")
    tick = timing._first_decision_tick(first + LOCAL_BUDGET_MS) if first is not None else None
    return {
        "delivered_ms": first,
        "first_usable_tick_ms": tick,
        "payload_age_ms": tick - timing.REFERENCE_MS if tick is not None else None,
        "timely": tick is not None and tick - timing.REFERENCE_MS <= timing.MAXIMUM_EMPTY_AGE_MS,
        "generated_wire_bytes": report["generated_bytes"],
        "packet_count": report["generated_packets"],
    }


def _percentile(times, probability):
    ordered = sorted(times)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def main():
    if OUTPUT.exists():
        raise RuntimeError("preserve first audit; output already exists")
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    if (sha256_file(previous.SOURCE / "report.json") != previous.SOURCE_SHA256
            or sha256_file(previous.MODELS / "report.json") != previous.MODELS_SHA256
            or sha256_file(previous.OUTPUT / "report.json") != PREVIOUS_SHA256
            or sha256_file(registration_path) != REGISTRATION_SHA256):
        raise RuntimeError("a frozen source/model/registration changed")
    source_report = json.loads((previous.SOURCE / "report.json").read_text(encoding="utf-8"))
    model_report = json.loads((previous.MODELS / "report.json").read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    policies = {}
    for family in FAMILIES:
        checkpoint = previous.MODELS / f"{family}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        policies[family] = DevelopmentEmptyWirePolicy(
            checkpoint,
            expected_sha256=model_report["models"][family]["checkpoint_sha256"],
            policy_hash=saved["policy_hash"],
        )
    config = registration["config"]
    counts = {family: {"feature_equal": 0, "prediction_equal": 0} for family in FAMILIES}
    timings = {family: [] for family in FAMILIES}
    negative = {}
    empty_payload = None
    checked = 0
    for name, digest in sorted(source_report["artifact_sha256"].items()):
        path = previous.SOURCE / name
        if sha256_file(path) != digest:
            raise RuntimeError("raw trace changed: " + name)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if len(artifact["references"]) != 1:
            raise RuntimeError("unexpected reference count")
        reference = artifact["references"][0]
        frame = artifact["frames"][str(timing.REFERENCE_MS)]
        ego = ego_from_sample(frame["receiver"])
        world = tuple(WorldObject(**item) for item in frame["objects"])
        observed = tuple(item for item in world if math.hypot(
            item.x_m - ego.x_m, item.y_m - ego.y_m) <= config["cooperative_range_m"])
        message = make_message(observed, timing.REFERENCE_MS, message_id=artifact["seed"])
        payload = message.encode()
        if hashlib.sha256(payload).hexdigest() != reference["frozen_input"]["payload_sha256"]:
            raise RuntimeError("reconstructed payload changed")
        if empty_payload is None and not message.objects:
            empty_payload = payload
        expected = raw_reference_vector(reference["frozen_input"]["batch"])
        for family in FAMILIES:
            policy = policies[family]
            original = _local_bound(reference, payload, policy, message.sender_id,
                                    timing.REFERENCE_MS)
            local = _local_bound(reference, payload, policy, message.sender_id,
                                 timing.REFERENCE_MS + 100)
            original_wire, local_wire = original.encode(), local.encode()
            features = raw_wire_features(BoundQuery.decode(local_wire), payload)
            if not np.array_equal(features, expected):
                raise RuntimeError("receiver-local features differ from trained reference")
            counts[family]["feature_equal"] += 1
            original_result = policy.predict(original_wire, payload)
            start = time.perf_counter_ns()
            local_result = policy.predict(local_wire, payload)
            timings[family].append((time.perf_counter_ns() - start) / 1e6)
            if (not np.array_equal(local_result[0], original_result[0])
                    or local_result[1] != original_result[1]):
                raise RuntimeError("receiver-local and exact-wire predictions differ")
            counts[family]["prediction_equal"] += 1
            if family not in negative:
                negative[family] = previous._negative_checks(local, payload, policy)
        checked += 1
    if checked != 576 or empty_payload is None:
        raise RuntimeError("incomplete reference sweep or no empty payload")
    optimistic = {}
    impaired = {}
    for condition_name in registration["condition_order"]:
        original = NetworkCondition(**registration["conditions"][condition_name])
        clean = replace(original, loss_probability=0.0, jitter_ms=0.0,
                        duplicate_probability=0.0, reorder_probability=0.0,
                        gilbert_good_to_bad=0.0, gilbert_bad_loss=0.0)
        optimistic[condition_name] = _one_delivery(clean, empty_payload, SEED_START)
        timely = delivered = generated_bytes = duplicates = 0
        for seed in range(SEED_START, SEED_START + SEED_COUNT):
            row = _one_delivery(original, empty_payload, seed)
            timely += int(row["timely"])
            delivered += int(row["delivered_ms"] is not None)
            generated_bytes += row["generated_wire_bytes"]
            duplicates += int(row["packet_count"] == 2)
        impaired[condition_name] = {
            "seed_count": SEED_COUNT,
            "delivered_count": delivered,
            "timely_200ms_count": timely,
            "timely_fraction": timely / SEED_COUNT,
            "mean_generated_wire_bytes": generated_bytes / SEED_COUNT,
            "duplicate_send_count": duplicates,
        }
    cpu = {family: {"p50_ms": statistics.median(times),
                    "p99_ms": _percentile(times, .99), "max_ms": max(times),
                    "checkpoint_bytes_not_charged_per_message":
                    (previous.MODELS / f"{family}.pt").stat().st_size}
           for family, times in timings.items()}
    negative_expected = {"payload_digest", "model_digest", "behavior"}
    gate = {
        "parity_576_both": all(values["feature_equal"] == 576 and
                               values["prediction_equal"] == 576
                               for values in counts.values()),
        "tamper_failed_closed": all(negative_expected.issubset(set(values))
                                    for values in negative.values()),
        "cpu_p99_at_most_20ms": all(row["p99_ms"] <= LOCAL_BUDGET_MS
                                  for row in cpu.values()),
        "optimistic_timely_both": all(row["timely"] for row in optimistic.values()),
    }
    OUTPUT.mkdir(parents=True)
    protocol = {
        "status": "RECEIVER_LOCAL_EMPTY_MODEL_TIMING_DEV_V1",
        "plan_sha256": sha256_file(PLAN),
        "script_sha256": sha256_file(Path(__file__)),
        "source_report_sha256": previous.SOURCE_SHA256,
        "models_report_sha256": previous.MODELS_SHA256,
        "previous_wire_audit_sha256": PREVIOUS_SHA256,
        "registration_sha256": REGISTRATION_SHA256,
        "seed_start": SEED_START, "seed_count_per_condition": SEED_COUNT,
        "local_inference_budget_ms": LOCAL_BUDGET_MS,
        "complete_source_view_certified": False,
        "receiver_model_preinstalled": True,
        "model_distribution_bytes_accounted": False,
        "independent_confirmation": False,
    }
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    report = {"status": protocol["status"],
              "protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "parity": counts, "negative_checks": negative,
              "cpu": cpu, "optimistic": optimistic,
              "full_impairment": impaired,
              "payload_application_bytes": len(empty_payload),
              "per_message_query_network_bytes": 0,
              "per_message_response_network_bytes": 0,
              "engineering_gate": gate,
              "engineering_gate_passed": all(gate.values()),
              "full_run_communication_accounted": False,
              "source_view_certified": False,
              "independent_confirmation": False}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "cpu": cpu, "optimistic": optimistic,
                      "full_impairment": impaired,
                      "report_sha256": sha256_file(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
