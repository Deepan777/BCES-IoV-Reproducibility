"""Development-only interpretation of a fresh empty infrastructure view.

The marker is in-memory receiver metadata derived from a decoded empty
message; it is not a physical object or a new registered wire format.
Its age and coverage premises must be bound explicitly in any real policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from bces.simulation.controlled_decisions import ControlledDecisionPlanner
from bces.simulation.development_early_yield_planner import DevelopmentEarlyYieldPlanner
from bces.utils.hashing import canonical_json_hash, sha256_file


@dataclass(frozen=True)
class VerifiedEmptyObservation:
    age_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.age_s) or self.age_s < 0:
            raise ValueError("empty-observation age must be finite and nonnegative")

    def propagated(self, seconds: float) -> VerifiedEmptyObservation:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("elapsed time must be finite and nonnegative")
        return VerifiedEmptyObservation(self.age_s + seconds)


class DevelopmentNegativeEvidencePlanner(DevelopmentEarlyYieldPlanner):
    maximum_empty_age_s = 0.4
    last_instance: DevelopmentNegativeEvidencePlanner | None = None

    def __init__(self, config=None):
        super().__init__(config)
        self.fresh_empty_uses = 0
        self.policy_hash = int(canonical_json_hash({
            "base_policy_hash": self.policy_hash,
            "negative_evidence_source_sha256": sha256_file(Path(__file__)),
            "maximum_empty_age_s": self.maximum_empty_age_s,
        })[:8], 16)
        type(self).last_instance = self

    def plan(self, local_objects, cooperative_objects, ego, route):
        cooperative = tuple(cooperative_objects or ())
        markers = tuple(item for item in cooperative if isinstance(item, VerifiedEmptyObservation))
        if len(markers) > 1:
            raise ValueError("at most one verified empty observation is allowed")
        physical = tuple(item for item in cooperative
                         if not isinstance(item, VerifiedEmptyObservation))
        fresh_empty = bool(markers and markers[0].age_s <= self.maximum_empty_age_s)
        if fresh_empty and not physical and not local_objects and self._within_approach(ego, route):
            self.fresh_empty_uses += 1
            self.yield_latched = False
            return ControlledDecisionPlanner.plan(self, local_objects, physical, ego, route)
        return super().plan(local_objects, physical, ego, route)
