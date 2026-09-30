"""Registered network-grid construction and complete application accounting."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable

from .events import Packet


def grid_conditions(grid: dict[str, list[float]]) -> tuple[dict[str, float], ...]:
    keys = (
        "latency_ms", "random_loss", "bandwidth_mbps",
        "pose_error_m", "yaw_error_deg", "clock_offset_ms",
    )
    if set(grid) != set(keys):
        raise ValueError("network grid axes changed")
    return tuple(
        dict(zip(keys, map(float, values)))
        for values in itertools.product(*(grid[key] for key in keys))
    )


def application_workload(
    method: str,
    accept_reuse: Iterable[bool],
    object_counts: Iterable[int],
    sizes: dict[str, int],
    *,
    scenario_index: int,
    scenario_period_ms: int,
    decision_step_ms: int,
) -> tuple[tuple[Packet, ...], dict[str, int], tuple[tuple[str, float], ...]]:
    """Build transport packets, component bytes, and decision deadlines."""
    accepted = tuple(bool(value) for value in accept_reuse)
    counts = tuple(int(value) for value in object_counts)
    if len(accepted) != len(counts) or not accepted:
        raise ValueError("decision and object-count sequences must align")
    if method not in {"surface", "learned_scalar_ttl", "always_fresh"}:
        raise ValueError("unknown method")
    start = float(scenario_index * scenario_period_ms)
    packets: list[Packet] = []
    components: Counter[str] = Counter()
    requirements: list[tuple[str, float]] = []

    def add(packet_id: str, generated: float, kind: str, parts: dict[str, int]) -> None:
        total = sum(parts.values())
        packets.append(Packet(packet_id, generated, total, kind))
        components.update(parts)

    extension = "bces_extension" if method == "surface" else "scalar_extension"
    prefix = f"s{scenario_index}:{method}"
    if method != "always_fresh":
        add(f"{prefix}:initial_query", start, "query", {"query": sizes["query"], "security_envelope": sizes["security_envelope"]})
        add(
            f"{prefix}:initial_response", start, "payload",
            {"object_header": sizes["object_header"], "object_records": counts[0] * sizes["object_record"], extension: sizes[extension], "security_envelope": sizes["security_envelope"]},
        )
    for index, (reuse, object_count) in enumerate(zip(accepted, counts), 1):
        generated = start + index * decision_step_ms
        deadline = generated + decision_step_ms
        if method == "always_fresh":
            add(f"{prefix}:query:{index}", generated, "query", {"query": sizes["query"], "security_envelope": sizes["security_envelope"]})
            response_id = f"{prefix}:response:{index}"
            add(response_id, generated, "payload", {"object_header": sizes["object_header"], "object_records": object_count * sizes["object_record"], "security_envelope": sizes["security_envelope"]})
            requirements.append((response_id, deadline))
        elif reuse:
            requirements.append((f"{prefix}:initial_response", deadline))
        else:
            add(f"{prefix}:refresh:{index}", generated, "refresh", {"refresh_request": sizes["refresh_request"], "security_envelope": sizes["security_envelope"]})
            response_id = f"{prefix}:response:{index}"
            add(response_id, generated, "payload", {"object_header": sizes["object_header"], "object_records": object_count * sizes["object_record"], extension: sizes[extension], "security_envelope": sizes["security_envelope"]})
            requirements.append((response_id, deadline))
    components.setdefault("retry_control", 0)
    return tuple(packets), dict(sorted(components.items())), tuple(requirements)


def delivered_decisions(trace: dict, requirements: Iterable[tuple[str, float]]) -> int:
    first_delivery: dict[str, float] = {}
    for delivery in trace["deliveries"]:
        if not delivery["duplicate"]:
            first_delivery.setdefault(delivery["packet_id"], float(delivery["delivered_ms"]))
    return sum(first_delivery.get(packet_id, float("inf")) <= deadline for packet_id, deadline in requirements)

