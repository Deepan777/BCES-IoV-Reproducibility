"""Development-only guard-band request coalescing for a BCES session.

The confirmed BoundSession is left untouched. This candidate may defer a new
request only while the receiver has already chosen LOCAL_ONLY in the guard
band. It never changes an acceptance decision or a packet binding.
"""

from __future__ import annotations

from bces.network.bound_session import BoundSession
from bces.protocol.receiver import DecisionState


class GuardCoalescingSession(BoundSession):
    def __init__(self, *, guard_cooldown_ms: int, **kwargs):
        if type(guard_cooldown_ms) is not int or guard_cooldown_ms < 0:
            raise ValueError("guard_cooldown_ms must be a nonnegative integer")
        super().__init__(**kwargs)
        self.guard_cooldown_ms = guard_cooldown_ms
        self.last_guard_request_ms: int | None = None

    def decision(self, current_state, behavior):
        now = current_state.timestamp_ms
        self.advance(now)
        decision = self.receiver.evaluate(current_state, behavior, self.policy.policy_hash, self.sender_id)
        if decision.request_refresh:
            guard = decision.state is DecisionState.REFRESH and decision.reason == "inside_refresh_guard_band"
            cooling = (guard and self.last_guard_request_ms is not None
                       and now - self.last_guard_request_ms < self.guard_cooldown_ms)
            if cooling and self.pending is None:
                self.counts["guard_refresh_coalesced"] += 1
            else:
                was_pending = self.pending is not None
                self.request_refresh(now)
                if guard and not was_pending:
                    self.last_guard_request_ms = now
        self.counts["decision:" + decision.state.value] += 1
        self.counts["reason:" + decision.reason] += 1
        return decision

    def report(self):
        result = super().report()
        result["development_guard_cooldown_ms"] = self.guard_cooldown_ms
        result["scope"] = "development-only policy candidate; not registered confirmation"
        return result
