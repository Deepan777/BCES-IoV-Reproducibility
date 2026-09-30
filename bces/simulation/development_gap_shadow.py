"""Off-path query-gap shadow intervention for development experiments only.

This wrapper retains a previously response-validated payload solely as
uncertain obstacle evidence while a new BCES query is pending. It does not
declare the old expiry surface valid. The combined fallback behavior has
no new wire-policy identity yet and must not be described as deployed or
packet-bound BCES. Motion bounds are assumptions, not empirically certified.
"""

from __future__ import annotations

from contextvars import ContextVar

from bces.geometry.reachable_shadow import ReachabilityAssumptions, reachable_shadow
from bces.network.bound_session import BoundSession
from bces.protocol.bound_exchange import decode_objects
from bces.protocol.receiver import DecisionState
from bces.simulation.controlled_contract import payload_objects
from bces.simulation.controlled_decisions import ControlledDecisionPlanner


SHADOW_ASSUMPTIONS = ReachabilityAssumptions(
    position_error_m=0.5,
    velocity_error_mps=1.0,
    acceleration_bound_mps2=4.0,
)
PLANNING_HORIZON_S = 3.0
ALLOWED_GAP_REASONS = frozenset({
    "awaiting_bound_response",
    "inside_refresh_guard_band",
    "surface_violated",
})
_pending_shadow: ContextVar[tuple] = ContextVar("development_pending_shadow", default=())


class GapShadowSession(BoundSession):
    """Preserve only a previously installed, bound-validated payload."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_verified_payload: bytes | None = None

    def decision(self, current_state, behavior):
        _pending_shadow.set(())
        result = super().decision(current_state, behavior)
        if self.receiver.payload is not None:
            self.last_verified_payload = self.receiver.payload
        if (result.state == DecisionState.ACCEPT_REUSE
                or result.reason not in ALLOWED_GAP_REASONS
                or self.last_verified_payload is None):
            return result
        message = decode_objects(self.last_verified_payload)
        if message.sender_id != self.sender_id:
            return result
        age_s = (current_state.timestamp_ms - message.timestamp_ms) / 1000.0
        if age_s < 0:
            return result
        shadows = tuple(reachable_shadow(obj, age_s=age_s,
                                         horizon_s=PLANNING_HORIZON_S,
                                         assumptions=SHADOW_ASSUMPTIONS)
                        for obj in payload_objects(message))
        _pending_shadow.set(shadows)
        return result


class GapShadowPlanner(ControlledDecisionPlanner):
    """Use shadows only on local-only gap decisions; locals override by ID."""

    shadow_plans = 0

    def plan(self, local_objects, cooperative_objects, ego, route):
        shadows = _pending_shadow.get()
        _pending_shadow.set(())
        if shadows and not cooperative_objects:
            type(self).shadow_plans += 1
            return super().plan(local_objects, shadows, ego, route)
        return super().plan(local_objects, cooperative_objects, ego, route)


def clear_gap_shadow():
    _pending_shadow.set(())
