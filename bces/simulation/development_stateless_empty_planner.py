"""Development-only stateless crossing fallback with a short empty-view gate.

The caller must establish the source-view coverage and absence premises before
constructing VerifiedEmptyObservation. This class does not verify a sensor,
authenticate a packet, or certify a decision-regret region.
"""

from __future__ import annotations

import math
from pathlib import Path

from bces.oracle.world import CostVector, GroundTruthWorld, PlannedTrajectory
from bces.simulation.controlled_decisions import ControlledDecisionPlanner
from bces.simulation.development_negative_evidence_planner import VerifiedEmptyObservation
from bces.simulation.development_stop_hold_candidate import stop_hold_points
from bces.utils.hashing import canonical_json_hash, sha256_file


class DevelopmentStatelessEmptyPlanner(ControlledDecisionPlanner):
    crossing_x_m = 104.8
    crossing_y_m = 95.2
    assumed_actor_length_m = 5.0
    assumed_actor_width_m = 1.8
    lane_center_deviation_m = 2.5
    position_error_m = 0.5
    braking_mps2 = -4.0
    maximum_empty_age_s = 0.2
    last_instance: DevelopmentStatelessEmptyPlanner | None = None

    def __init__(self, config=None):
        super().__init__(config)
        self.fresh_empty_uses = 0  # audit only; never read by plan decisions
        self.fallback_uses = 0
        self.policy_hash = int(canonical_json_hash({
            "base_policy_hash": self.policy_hash,
            "source_sha256": sha256_file(Path(__file__)),
            "stop_hold_source_sha256": sha256_file(
                Path(__file__).with_name("development_stop_hold_candidate.py")),
            "maximum_empty_age_s": self.maximum_empty_age_s,
        })[:8], 16)
        type(self).last_instance = self

    def _approaching_crossing(self, ego, route):
        return (route.intended_behavior == "keep"
                and abs(ego.x_m - self.crossing_x_m) <= 2.0
                and abs(math.sin(ego.heading_rad) - 1.0) <= 0.05
                and ego.y_m < self.crossing_y_m)

    def _stop_threshold_m(self, ego):
        body_margin = (0.5 * math.hypot(ego.length_m, ego.width_m)
                       + 0.5 * math.hypot(self.assumed_actor_length_m,
                                          self.assumed_actor_width_m)
                       + self.lane_center_deviation_m + self.position_error_m
                       + self.config.collision_margin_m)
        braking_distance = ego.speed_mps**2 / (2 * -self.braking_mps2)
        reaction_distance = ego.speed_mps * self.config.step_s
        return body_margin + braking_distance + reaction_distance

    def plan(self, local_objects, cooperative_objects, ego, route):
        cooperative = tuple(cooperative_objects or ())
        markers = tuple(item for item in cooperative
                        if isinstance(item, VerifiedEmptyObservation))
        if len(markers) > 1:
            raise ValueError("at most one empty-view marker is allowed")
        physical = tuple(item for item in cooperative
                         if not isinstance(item, VerifiedEmptyObservation))
        if markers and physical:
            raise ValueError("empty-view marker contradicts physical cooperative objects")
        if not self._approaching_crossing(ego, route) or physical or local_objects:
            return super().plan(local_objects, physical, ego, route)
        if markers and markers[0].age_s <= self.maximum_empty_age_s:
            self.fresh_empty_uses += 1
            return super().plan(local_objects, physical, ego, route)
        if self.crossing_y_m - ego.y_m > self._stop_threshold_m(ego):
            return super().plan(local_objects, physical, ego, route)
        self.fallback_uses += 1
        points = stop_hold_points(
            ego, route, braking_mps2=self.braking_mps2,
            horizon_s=self.config.horizon_s, step_s=self.config.step_s)
        placeholder = PlannedTrajectory(
            maneuver="keep", acceleration_mps2=self.braking_mps2,
            points=points, perceived_cost=CostVector(0, 0, 0, 0, 0, 0, 0, None))
        cost = self.evaluate(placeholder, GroundTruthWorld(0, ()), ego, route)
        return PlannedTrajectory("keep", self.braking_mps2, points, cost)
