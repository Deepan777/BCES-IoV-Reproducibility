#!/usr/bin/env python3
"""Read-only independent replay of the receiver-local timing/accounting report."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.network.events import NetworkCondition
from bces.network.wire_channel import WireChannel
from bces.utils.hashing import sha256_file

OUTPUT = ROOT / "outputs/study_b/receiver_local_empty_model_timing_dev_v1"


def replay(condition, payload, seed, budget):
    channel = WireChannel(condition, seed=seed, security_bytes=64)
    channel.send("payload", payload, 3000)
    arrivals = []
    while True:
        event = channel.next_delivery(1_000_000.0)
        if event is None:
            break
        arrivals.append(event.delivered_ms)
    accounting = channel.report()
    assert accounting["byte_conservation_ok"] and accounting["component_conservation_ok"]
    if not arrivals:
        return False, False, accounting["generated_bytes"], accounting["generated_packets"]
    first_tick = 3000 + 200 * max(1, math.ceil((min(arrivals) + budget - 3000) / 200 - 1e-12))
    return True, first_tick <= 3200, accounting["generated_bytes"], accounting["generated_packets"]


def main():
    protocol_path, report_path = OUTPUT / "protocol.json", OUTPUT / "report.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["protocol_sha256"] == sha256_file(protocol_path)
    assert protocol["plan_sha256"] == sha256_file(
        ROOT / "docs/STUDY_B_RECEIVER_LOCAL_EMPTY_MODEL_TIMING_DEV_V1_PLAN.md")
    assert protocol["script_sha256"] == sha256_file(
        ROOT / "scripts/155_audit_receiver_local_empty_model_timing_dev.py")
    assert protocol["source_report_sha256"] == sha256_file(
        ROOT / "outputs/study_b/diverse_empty_labels_development_v2/report.json")
    assert protocol["models_report_sha256"] == sha256_file(
        ROOT / "outputs/study_b/diverse_empty_transfer_models_dev_v1/report.json")
    assert protocol["registration_sha256"] == sha256_file(
        ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json")
    assert protocol["previous_wire_audit_sha256"] == sha256_file(
        ROOT / "outputs/study_b/diverse_empty_teacher_wire_parity_dev_v1/report.json")
    assert protocol["seed_start"] == 260926000 and protocol["seed_count_per_condition"] == 4096
    assert protocol["local_inference_budget_ms"] == 20.0
    assert not protocol["complete_source_view_certified"]
    assert not report["full_run_communication_accounted"]
    assert not report["independent_confirmation"]
    for family in ("surface_teacher", "scalar_ttl_teacher"):
        assert report["parity"][family] == {"feature_equal": 576, "prediction_equal": 576}
        assert {"payload_digest", "model_digest", "behavior"}.issubset(
            report["negative_checks"][family])
        assert report["cpu"][family]["p99_ms"] <= 20
    assert all(report["engineering_gate"].values())
    # This is the exact empty reference selected by the registered earlier timing audit.
    import importlib
    prior = importlib.import_module("scripts.147_audit_empty_view_round_trip_dev")
    source_report = json.loads((ROOT / "outputs/study_b/diverse_empty_labels_development_v2/report.json")
                               .read_text(encoding="utf-8"))
    registration = json.loads((ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json")
                              .read_text(encoding="utf-8"))
    _, _, message = prior._empty_reference(registration["config"], source_report)
    payload = message.encode()
    assert len(payload) == report["payload_application_bytes"]
    assert report["per_message_query_network_bytes"] == 0
    assert report["per_message_response_network_bytes"] == 0
    for name in registration["condition_order"]:
        original = NetworkCondition(**registration["conditions"][name])
        clean = replace(original, loss_probability=0.0, jitter_ms=0.0,
                        duplicate_probability=0.0, reorder_probability=0.0,
                        gilbert_good_to_bad=0.0, gilbert_bad_loss=0.0)
        delivered, timely, wire, packets = replay(clean, payload, 260926000, 20.0)
        row = report["optimistic"][name]
        assert delivered and timely and row["timely"]
        assert wire == row["generated_wire_bytes"] == len(payload) + 10 + 64
        assert packets == row["packet_count"] == 1
        counts = {"delivered_count": 0, "timely_200ms_count": 0,
                  "generated_wire_bytes": 0, "duplicate_send_count": 0}
        for seed in range(260926000, 260930096):
            delivered, timely, wire, packets = replay(original, payload, seed, 20.0)
            counts["delivered_count"] += int(delivered)
            counts["timely_200ms_count"] += int(timely)
            counts["generated_wire_bytes"] += wire
            counts["duplicate_send_count"] += int(packets == 2)
        reported = report["full_impairment"][name]
        assert reported["seed_count"] == 4096
        for key in ("delivered_count", "timely_200ms_count", "duplicate_send_count"):
            assert counts[key] == reported[key], (name, key)
        assert counts["generated_wire_bytes"] / 4096 == reported["mean_generated_wire_bytes"]
        assert counts["timely_200ms_count"] / 4096 == reported["timely_fraction"]
    print(json.dumps({"verification": "PASS", "report_sha256": sha256_file(report_path),
                      "scope": "read-only development timing/accounting replay"}, indent=2))


if __name__ == "__main__":
    main()
