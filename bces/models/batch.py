"""Inference-only, padded model-batch representation for object messages."""

from __future__ import annotations

from dataclasses import dataclass

from bces.data.objects import ObjectMessage, ObjectSource, PerceivedObject
from bces.protocol.query import ReceiverQuery

MAX_MESSAGE_OBJECTS = 32
OBJECT_FEATURE_COUNT = 10

_SOURCE_IDS = {
    ObjectSource.VEHICLE: 0.0,
    ObjectSource.INFRASTRUCTURE: 1.0,
    ObjectSource.COOPERATIVE: 2.0,
}


@dataclass(frozen=True)
class ModelBatchRow:
    object_features: tuple[tuple[float, ...], ...]
    object_mask: tuple[bool, ...]
    reference_state: tuple[float, ...]
    drift_scales: tuple[float, ...]
    reference_path: tuple[tuple[float, ...], ...]
    identifiers: tuple[int, ...]
    object_count: int
    occlusion_proxy: float
    estimated_delay_s: float
    map_context_flags: int

    def __post_init__(self) -> None:
        if len(self.object_features) != MAX_MESSAGE_OBJECTS:
            raise ValueError("model batch must contain 32 padded object rows")
        if any(len(row) != OBJECT_FEATURE_COUNT for row in self.object_features):
            raise ValueError("unexpected object feature width")
        if len(self.object_mask) != MAX_MESSAGE_OBJECTS:
            raise ValueError("object mask must contain 32 entries")
        if sum(self.object_mask) != self.object_count:
            raise ValueError("object mask and count disagree")
        if len(self.reference_path) != 12:
            raise ValueError("model batch must contain 12 reference-path points")


def _relevance_key(item: PerceivedObject, query: ReceiverQuery) -> tuple[float, str]:
    dx = item.x_m - query.reference_state.x_m
    dy = item.y_m - query.reference_state.y_m
    return (dx * dx + dy * dy, item.track_id)


def build_model_batch_row(
    message: ObjectMessage,
    query: ReceiverQuery,
    *,
    occlusion_proxy: float,
    estimated_delay_s: float,
    map_context_flags: int = 0,
) -> ModelBatchRow:
    if not 0.0 <= occlusion_proxy <= 1.0:
        raise ValueError("occlusion_proxy must be in [0, 1]")
    if estimated_delay_s < 0.0:
        raise ValueError("estimated_delay_s must be non-negative")
    if not isinstance(map_context_flags, int) or not 0 <= map_context_flags <= 255:
        raise ValueError("map_context_flags must be uint8")
    ordered = sorted(message.objects, key=lambda item: _relevance_key(item, query))[
        :MAX_MESSAGE_OBJECTS
    ]
    rows: list[tuple[float, ...]] = []
    for item in ordered:
        rows.append(
            (
                item.x_m - query.reference_state.x_m,
                item.y_m - query.reference_state.y_m,
                item.vx_mps,
                item.vy_mps,
                item.length_m,
                item.width_m,
                float(item.class_id),
                item.confidence,
                _SOURCE_IDS[item.source],
                1.0,
            )
        )
    rows.extend([(0.0,) * OBJECT_FEATURE_COUNT] * (MAX_MESSAGE_OBJECTS - len(rows)))
    mask = (True,) * len(ordered) + (False,) * (MAX_MESSAGE_OBJECTS - len(ordered))
    state = query.reference_state
    reference_state = (
        state.x_m,
        state.y_m,
        state.speed_mps,
        state.heading_rad,
        state.acceleration_mps2,
        state.curvature_inv_m,
        float(state.timestamp_ms),
    )
    reference_path = tuple(
        (
            point.x_m,
            point.y_m,
            point.speed_mps,
            point.heading_rad,
            point.curvature_inv_m,
            point.time_offset_s,
        )
        for point in query.reference_path
    )
    return ModelBatchRow(
        object_features=tuple(rows),
        object_mask=mask,
        reference_state=reference_state,
        drift_scales=query.drift_scales.as_tuple(),
        reference_path=reference_path,
        identifiers=(
            int(query.behavior),
            query.risk_class,
            query.policy_hash,
            query.drift_schema_id,
            query.normal_codebook_id,
            query.calibration_id,
        ),
        object_count=len(ordered),
        occlusion_proxy=float(occlusion_proxy),
        estimated_delay_s=float(estimated_delay_s),
        map_context_flags=map_context_flags,
    )
