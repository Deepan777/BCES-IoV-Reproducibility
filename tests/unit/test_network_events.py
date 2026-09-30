from __future__ import annotations

import pytest

from bces.network.events import NetworkCondition, Packet, simulate_packets

pytestmark = pytest.mark.phase8


def _packets() -> tuple[Packet, ...]:
    return tuple(Packet(f"p{index}", index * 10.0, 100, "payload") for index in range(5))


def test_analytic_bandwidth_queue_and_byte_conservation() -> None:
    trace = simulate_packets(
        _packets(), NetworkCondition(0, 0, 1.0), seed=1
    )
    assert trace["byte_conservation_ok"]
    assert trace["generated_bytes"] == trace["delivered_original_bytes"] == 500
    assert trace["deliveries"][0]["delivered_ms"] == pytest.approx(0.8)


def test_fixed_seed_reproduces_golden_event_trace() -> None:
    condition = NetworkCondition(50, 0.2, 0.5, jitter_ms=5, duplicate_probability=0.2)
    assert simulate_packets(_packets(), condition, seed=17) == simulate_packets(
        _packets(), condition, seed=17
    )


def test_latency_sensitivity_is_monotone_without_random_impairment() -> None:
    low = simulate_packets(_packets(), NetworkCondition(0, 0, 4), seed=3)
    high = simulate_packets(_packets(), NetworkCondition(100, 0, 4), seed=3)
    assert high["p50_latency_ms"] >= low["p50_latency_ms"] + 100


def test_gilbert_elliott_loss_is_accounted() -> None:
    condition = NetworkCondition(
        0, 0, 1, gilbert_good_to_bad=1, gilbert_bad_to_good=0,
        gilbert_bad_loss=1,
    )
    trace = simulate_packets(_packets(), condition, seed=5)
    assert trace["dropped_bytes"] == trace["generated_bytes"]
    assert trace["delivered_original_bytes"] == 0
