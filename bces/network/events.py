"""Compact deterministic packet event simulator with auditable byte conservation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Packet:
    packet_id: str
    generated_ms: float
    size_bytes: int
    kind: str

    def __post_init__(self) -> None:
        if not self.packet_id or self.generated_ms < 0 or self.size_bytes <= 0:
            raise ValueError("invalid packet")


@dataclass(frozen=True)
class NetworkCondition:
    latency_ms: float
    loss_probability: float
    bandwidth_mbps: float
    jitter_ms: float = 0.0
    duplicate_probability: float = 0.0
    reorder_probability: float = 0.0
    clock_offset_ms: float = 0.0
    pose_error_m: float = 0.0
    yaw_error_deg: float = 0.0
    gilbert_good_to_bad: float = 0.0
    gilbert_bad_to_good: float = 1.0
    gilbert_bad_loss: float = 1.0

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or self.bandwidth_mbps <= 0 or self.jitter_ms < 0:
            raise ValueError("invalid latency, jitter, or bandwidth")
        probabilities = (
            self.loss_probability, self.duplicate_probability,
            self.reorder_probability, self.gilbert_good_to_bad,
            self.gilbert_bad_to_good, self.gilbert_bad_loss,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("probabilities must be in [0, 1]")


@dataclass(frozen=True)
class Delivery:
    packet_id: str
    kind: str
    generated_ms: float
    delivered_ms: float
    size_bytes: int
    duplicate: bool
    reordered: bool


def simulate_packets(
    packets: tuple[Packet, ...], condition: NetworkCondition, *, seed: int,
) -> dict:
    generator = random.Random(seed)
    queue_available_ms = 0.0
    bad_state = False
    deliveries: list[Delivery] = []
    dropped_ids: list[str] = []
    generated_by_kind: dict[str, int] = {}
    delivered_original_bytes = duplicate_bytes = 0
    for packet in sorted(packets, key=lambda item: (item.generated_ms, item.packet_id)):
        generated_by_kind[packet.kind] = generated_by_kind.get(packet.kind, 0) + packet.size_bytes
        if bad_state:
            bad_state = not (generator.random() < condition.gilbert_bad_to_good)
        else:
            bad_state = generator.random() < condition.gilbert_good_to_bad
        lost = generator.random() < condition.loss_probability
        if bad_state:
            lost = lost or generator.random() < condition.gilbert_bad_loss
        if lost:
            dropped_ids.append(packet.packet_id)
            continue
        serialization_ms = packet.size_bytes * 8.0 / (condition.bandwidth_mbps * 1000.0)
        transmission_start = max(packet.generated_ms, queue_available_ms)
        queue_available_ms = transmission_start + serialization_ms
        jitter = generator.uniform(-condition.jitter_ms, condition.jitter_ms)
        reordered = generator.random() < condition.reorder_probability
        reorder_delay = generator.uniform(0.0, max(condition.latency_ms, 1.0)) if reordered else 0.0
        delivered_ms = max(
            queue_available_ms,
            queue_available_ms + condition.latency_ms + jitter + reorder_delay,
        )
        deliveries.append(
            Delivery(
                packet.packet_id, packet.kind, packet.generated_ms,
                delivered_ms, packet.size_bytes, False, reordered,
            )
        )
        delivered_original_bytes += packet.size_bytes
        if generator.random() < condition.duplicate_probability:
            duplicate_delay = generator.uniform(0.01, max(condition.latency_ms, 1.0))
            deliveries.append(
                Delivery(
                    packet.packet_id, packet.kind, packet.generated_ms,
                    delivered_ms + duplicate_delay, packet.size_bytes, True, reordered,
                )
            )
            duplicate_bytes += packet.size_bytes
    deliveries.sort(key=lambda item: (item.delivered_ms, item.packet_id, item.duplicate))
    latencies = sorted(item.delivered_ms - item.generated_ms for item in deliveries if not item.duplicate)

    def percentile(probability: float) -> float | None:
        if not latencies:
            return None
        index = round(probability * (len(latencies) - 1))
        return latencies[index]

    generated_bytes = sum(packet.size_bytes for packet in packets)
    dropped_bytes = generated_bytes - delivered_original_bytes
    return {
        "seed": seed, "condition": asdict(condition),
        "deliveries": [asdict(item) for item in deliveries],
        "dropped_packet_ids": dropped_ids,
        "generated_bytes": generated_bytes,
        "generated_bytes_by_kind": dict(sorted(generated_by_kind.items())),
        "delivered_original_bytes": delivered_original_bytes,
        "duplicate_bytes": duplicate_bytes, "dropped_bytes": dropped_bytes,
        "byte_conservation_ok": generated_bytes == delivered_original_bytes + dropped_bytes,
        "p50_latency_ms": percentile(0.50), "p95_latency_ms": percentile(0.95),
        "delivered_packets": len(deliveries),
        "duplicate_packets": sum(item.duplicate for item in deliveries),
        "reordered_packets": sum(item.reordered for item in deliveries),
        "clock_offset_ms": condition.clock_offset_ms,
        "pose_error_m": condition.pose_error_m, "yaw_error_deg": condition.yaw_error_deg,
    }
