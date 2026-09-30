from collections import Counter
from types import SimpleNamespace

import pytest

from bces.network.guard_coalescing import GuardCoalescingSession
from bces.protocol.receiver import DecisionState, EvidenceMode, ReceiverDecision


def fake_session(decision, cooldown=600):
    app = object.__new__(GuardCoalescingSession)
    app.guard_cooldown_ms = cooldown
    app.last_guard_request_ms = None
    app.pending = None
    app.policy = SimpleNamespace(policy_hash=7)
    app.sender_id = "sender"
    app.receiver = SimpleNamespace(evaluate=lambda *args: decision)
    app.counts = Counter()
    app.advance = lambda now: None
    requests = []
    app.request_refresh = requests.append
    return app, requests


def test_guard_coalescing_keeps_local_only_decision_and_defers_repeated_request():
    decision = ReceiverDecision(DecisionState.REFRESH, EvidenceMode.LOCAL_ONLY, True,
                                "inside_refresh_guard_band")
    app, requests = fake_session(decision)
    for tick in (1000, 1200, 1400, 1600):
        assert app.decision(SimpleNamespace(timestamp_ms=tick), 0) == decision
    assert requests == [1000, 1600]
    assert app.counts["guard_refresh_coalesced"] == 2
    assert app.counts["decision:REFRESH"] == 4


def test_surface_violation_is_never_deferred_by_guard_cooldown():
    decision = ReceiverDecision(DecisionState.INVALID, EvidenceMode.LOCAL_ONLY, True,
                                "surface_violated")
    app, requests = fake_session(decision)
    app.last_guard_request_ms = 1000
    assert app.decision(SimpleNamespace(timestamp_ms=1200), 0) == decision
    assert requests == [1200]


def test_bad_cooldown_is_rejected():
    with pytest.raises(ValueError):
        GuardCoalescingSession(guard_cooldown_ms=-1)
