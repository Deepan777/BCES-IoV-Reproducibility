"""Development-only, payload-bound choice of surface or learned scalar TTL.

The rule uses only the newly received cooperative payload, reference receiver
state, and observable planner decision margin. It does not inspect a scenario
family label, future trajectory, or an oracle validity outcome. This is not a
certified regret bound and is not part of any registered v1/v2 policy.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from bces.models.bound_policy import FrozenWirePolicy, bind_reference
from bces.protocol.bound_exchange import BoundQuery, decode_response, encode_response
from bces.utils.hashing import canonical_json_hash, sha256_file

FRONT_MIN_M = 5.0
FRONT_MAX_M = 100.0
LATERAL_MAX_M = 2.25
LATERAL_VELOCITY_MAX_MPS = 1.5
FIRST_COMPETITOR_TOTAL_GAP_MIN = 0.20


def select_reference_method(reference: dict) -> tuple[str, dict]:
    """Return a method from fields available at the reference query time."""
    batch = reference["frozen_input"]["batch"]
    heading = float(batch["reference_state"][3])
    cosine, sine = math.cos(heading), math.sin(heading)
    lead_like = False
    for obj, observed in zip(batch["object_features"], batch["object_mask"]):
        if not observed or int(obj[6]) != 0:
            continue
        dx, dy, vx, vy = map(float, obj[:4])
        along = cosine * dx + sine * dy
        lateral = abs(-sine * dx + cosine * dy)
        lateral_velocity = abs(-sine * vx + cosine * vy)
        forward_velocity = cosine * vx + sine * vy
        if (FRONT_MIN_M <= along <= FRONT_MAX_M and lateral <= LATERAL_MAX_M
                and lateral_velocity <= LATERAL_VELOCITY_MAX_MPS
                and forward_velocity >= 0.0):
            lead_like = True
            break
    gap = float(reference["observable_margin_features"][12])
    if not math.isfinite(gap):
        raise ValueError("nonfinite first-competitor total-cost gap")
    selected = "scalar_ttl" if lead_like and gap >= FIRST_COMPETITOR_TOTAL_GAP_MIN else "surface"
    return selected, {"lead_like": lead_like, "first_competitor_total_cost_gap": gap}


class AdaptiveMarginWirePolicy:
    """One hybrid identity binds either frozen model to each exact query/payload.

    The registered FrozenWirePolicy instances remain immutable. A new query
    revokes the previous extension in the existing BoundReceiver. Re-encoding
    the selected model's extension against the *hybrid* query binds it to the
    actual model-selection and payload context.
    """

    def __init__(self, *, surface: FrozenWirePolicy, scalar_ttl: FrozenWirePolicy,
                 shrinkages: dict[str, float]):
        if (surface.family != "surface" or scalar_ttl.family != "scalar_ttl"
                or surface.view != scalar_ttl.view
                or surface.policy_hash != scalar_ttl.policy_hash):
            raise ValueError("incompatible frozen policy pair")
        if set(shrinkages) != {"surface", "scalar_ttl"}:
            raise ValueError("both frozen shrinkages required")
        self.models = {"surface": surface, "scalar_ttl": scalar_ttl}
        self.shrinkages = {name: float(value) for name, value in shrinkages.items()}
        if any(not math.isfinite(value) or value < 0 for value in self.shrinkages.values()):
            raise ValueError("invalid shrinkage")
        self.view = surface.view
        self.policy_hash = surface.policy_hash
        self.family = "surface"  # used only before the first payload arrives
        self.sha256 = canonical_json_hash({
            "kind": "development_adaptive_margin_wire_v1",
            "surface_sha256": surface.sha256,
            "scalar_ttl_sha256": scalar_ttl.sha256,
            "shrinkages": self.shrinkages,
            "gate_source_sha256": sha256_file(Path(__file__)),
        })
        self.selections: list[dict] = []

    def bind(self, reference, query, payload, **_ignored):
        selected, evidence = select_reference_method(reference)
        self.family = selected
        bound = bind_reference(reference, query, payload, model_sha256=self.sha256,
                               view=self.view, family=selected,
                               shrinkage=self.shrinkages[selected])
        self.selections.append({"reference_id": reference["reference_id"],
                                "family": selected, "query_id": query.query_id,
                                "payload_sha256": bound.payload_sha256,
                                "evidence": evidence})
        return bound

    def issue(self, query_wire: bytes, payload: bytes) -> bytes:
        bound = BoundQuery.decode(query_wire)
        if (bound.model_sha256 != self.sha256 or bound.feature_view != self.view
                or bound.query.policy_hash != self.policy_hash
                or bound.shrinkage != self.shrinkages[bound.family]):
            raise ValueError("hybrid query policy/model binding mismatch")
        selected = self.models[bound.family]
        selected_query = replace(bound, model_sha256=selected.sha256).encode()
        selected_response = selected.issue(selected_query, payload)
        extension = decode_response(selected_response, selected_query)
        return encode_response(query_wire, extension)
