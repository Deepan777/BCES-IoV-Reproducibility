"""Unregistered development-only setup for the short-age sensitivity."""

from __future__ import annotations

from contextlib import contextmanager
import json

from bces.simulation import development_map_loop
from bces.simulation.development_negative_evidence_planner import (
    DevelopmentNegativeEvidencePlanner, VerifiedEmptyObservation,
)
from bces.simulation.development_visibility_map import StaticMapObstruction
from bces.utils.hashing import sha256_file

BLOCKERS = (StaticMapObstruction("southwest_corner", 93.0, 89.5, 0.0, 8.0, 6.0),)
SELECTED_PROTOCOL = "outputs/study_b/recovery_crossing_driving_development_v1/protocol.json"
SELECTED_PROTOCOL_SHA256 = "b85ec159b2ac8323eee2dfd5deeea924302231c6a85ed74346484ba6fc6d58f3"


def selected_development_seeds(root):
    path = root / SELECTED_PROTOCOL
    if sha256_file(path) != SELECTED_PROTOCOL_SHA256:
        raise RuntimeError("original development selection changed")
    selected = json.loads(path.read_text(encoding="utf-8"))["selected"]
    if len(selected) != 8 or len({item["seed"] for item in selected}) != 8:
        raise RuntimeError("expected eight unique development seed clusters")
    return selected


@contextmanager
def branch_binding(actor_presence):
    original_planner = development_map_loop.ControlledDecisionPlanner
    original_populate = development_map_loop._populate
    original_payload_objects = development_map_loop.payload_objects

    def populate(connection, spec):
        original_populate(connection, spec)
        if actor_presence == "absent":
            connection.vehicle.remove(spec.actor_id)

    def payload_objects(message):
        physical = original_payload_objects(message)
        if not physical:
            return (VerifiedEmptyObservation(),)
        return physical

    if actor_presence not in ("present", "absent"):
        raise ValueError("unknown actor-presence stratum")
    try:
        development_map_loop.ControlledDecisionPlanner = DevelopmentNegativeEvidencePlanner
        development_map_loop._populate = populate
        development_map_loop.payload_objects = payload_objects
        yield
    finally:
        development_map_loop.ControlledDecisionPlanner = original_planner
        development_map_loop._populate = original_populate
        development_map_loop.payload_objects = original_payload_objects
