#!/usr/bin/env python3
"""Independent algebraic replay of the optimistic framed-message timing."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file

DIR = ROOT / "outputs/study_b/empty_view_round_trip_dev_v1"
FRAME_AND_SECURITY_BYTES = 10 + 64


def _arrival(generated, application_bytes, latency_ms, bandwidth_mbps):
    return generated + (application_bytes + FRAME_AND_SECURITY_BYTES) * 8 / (bandwidth_mbps * 1000) + latency_ms


def _tick(time_ms):
    return 3000 + max(1, math.ceil((time_ms - 3000) / 200 - 1e-12)) * 200


def main():
    protocol = json.loads((DIR / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((DIR / "report.json").read_text(encoding="utf-8"))
    registration_path = ROOT / "outputs/study_b/closed_loop_confirmation_v2/preregistration.json"
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if (sha256_file(DIR / "protocol.json") != report["protocol_sha256"]
            or sha256_file(ROOT / "docs/STUDY_B_EMPTY_VIEW_ROUND_TRIP_DEV_V1_PLAN.md") != protocol["plan_sha256"]
            or sha256_file(ROOT / "scripts/147_audit_empty_view_round_trip_dev.py") != protocol["script_sha256"]
            or sha256_file(registration_path) != protocol["registration_sha256"]):
        raise AssertionError("timing audit provenance changed")
    for name in registration["condition_order"]:
        condition = registration["conditions"][name]
        latency = condition["latency_ms"]
        bandwidth = condition["bandwidth_mbps"]
        for family in ("surface_teacher", "scalar_ttl_teacher"):
            row = report["results"][name][family]
            sizes = row["application_bytes"]
            refresh = _arrival(3000, sizes["refresh"], latency, bandwidth)
            payload = _arrival(refresh, sizes["payload"], latency, bandwidth)
            query_generated = math.ceil(payload)
            query = _arrival(query_generated, sizes["query"], latency, bandwidth)
            response_generated = query + 6.0
            response = _arrival(response_generated, sizes["response"], latency, bandwidth)
            expected = {
                "refresh_delivered_ms": refresh,
                "payload_generated_ms": refresh,
                "payload_delivered_ms": payload,
                "query_generated_ms": query_generated,
                "query_delivered_ms": query,
                "response_generated_ms": response_generated,
                "response_delivered_ms": response,
                "first_decision_tick_ms": _tick(response),
                "payload_age_at_first_decision_ms": _tick(response) - 3000,
                "generated_wire_bytes": sum(value + FRAME_AND_SECURITY_BYTES for value in sizes.values()),
            }
            for field, value in expected.items():
                if not math.isclose(row[field], value, abs_tol=1e-8, rel_tol=0):
                    raise AssertionError(name + ":" + family + ":" + field)
            if row["within_200ms_empty_gate"] != (expected["payload_age_at_first_decision_ms"] <= 200):
                raise AssertionError("empty-age gate miscomputed")
        direct = report["results"][name]["periodic_direct_payload"]
        arrival = _arrival(3000, direct["application_payload_bytes"], latency, bandwidth)
        if (not math.isclose(direct["payload_delivered_ms"], arrival, abs_tol=1e-8, rel_tol=0)
                or direct["first_decision_tick_ms"] != _tick(arrival)
                or direct["payload_age_at_first_decision_ms"] != _tick(arrival) - 3000
                or direct["generated_wire_bytes"] != direct["application_payload_bytes"] + FRAME_AND_SECURITY_BYTES):
            raise AssertionError("direct-push timing differs")
    if any(report["assessment"][key] for key in report["assessment"] if key.startswith("bound_")):
        raise AssertionError("bound exchange unexpectedly cleared short-age gate")
    if not all(report["assessment"][key] for key in report["assessment"] if key.startswith("direct_")):
        raise AssertionError("direct payload unexpectedly missed short-age gate")
    print(json.dumps({"status": "PASS", "conditions": registration["condition_order"],
                      "bound_age_ms": {name: report["results"][name]["surface_teacher"]["payload_age_at_first_decision_ms"]
                                       for name in registration["condition_order"]},
                      "report_sha256": sha256_file(DIR / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
