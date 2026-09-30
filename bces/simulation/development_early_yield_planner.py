"""Development-only map-aware yield planner for an occluded crossing.

This is not a registered BCES policy. The fixed crossing map is locally
available; hidden actor truth is never inspected. It tests whether a causal
fallback can create a meaningful information-versus-progress trade-off.
"""

from __future__ import annotations

import math
from pathlib import Path

from bces.oracle.world import CostVector, GroundTruthWorld, PlannedTrajectory
from bces.simulation.controlled_decisions import ControlledDecisionPlanner
from bces.simulation.development_stop_hold_candidate import stop_hold_points
from bces.utils.hashing import canonical_json_hash, sha256_file


class DevelopmentEarlyYieldPlanner(ControlledDecisionPlanner):
    crossing_x_m = 104.8
    crossing_y_m = 95.2
    assumed_actor_length_m = 5.0
    assumed_actor_width_m = 1.8
    lane_center_deviation_m = 2.5
    position_error_m = 0.5
    braking_mps2 = -4.0

    def __init__(self, config=None):
        super().__init__(config)
        self.yield_latched = False
        self.actor_passed_observed = False
        self.policy_hash = int(canonical_json_hash({
            "base_policy_hash": self.policy_hash,
            "early_yield_source_sha256": sha256_file(Path(__file__)),
            "stop_hold_source_sha256": sha256_file(
                Path(__file__).with_name("development_stop_hold_candidate.py")),
            "map_crossing_xy_m": [self.crossing_x_m, self.crossing_y_m],
        })[:8], 16)

    def _clearance_m(self, ego) -> float:
        return (0.5 * math.hypot(ego.length_m, ego.width_m)
                + 0.5 * math.hypot(self.assumed_actor_length_m,
                                   self.assumed_actor_width_m)
                + self.lane_center_deviation_m + self.position_error_m
                + self.config.collision_margin_m)

    def _within_approach(self, ego, route) -> bool:
        return (route.intended_behavior == "keep"
                and abs(ego.x_m - self.crossing_x_m) <= 2.0
                and abs(math.sin(ego.heading_rad) - 1.0) <= 0.05
                and ego.y_m < self.crossing_y_m)

    def plan(self, local_objects, cooperative_objects, ego, route):
        if not self._within_approach(ego, route):
            return super().plan(local_objects, cooperative_objects, ego, route)
        observed = tuple(local_objects) + tuple(cooperative_objects or ())
        for actor in observed:
            if (actor.x_m > self.crossing_x_m + actor.length_m / 2
                    + ego.width_m / 2 + self.config.collision_margin_m
                    and actor.vx_mps >= 0):
                self.actor_passed_observed = True
        if self.actor_passed_observed:
            self.yield_latched = False
            return super().plan(local_objects, cooperative_objects, ego, route)
        if observed:
            return super().plan(local_objects, cooperative_objects, ego, route)
        distance = self.crossing_y_m - ego.y_m
        braking_distance = ego.speed_mps**2 / (2 * -self.braking_mps2)
        reaction_distance = ego.speed_mps * self.config.step_s
        if distance <= self._clearance_m(ego) + braking_distance + reaction_distance:
            self.yield_latched = True
        if not self.yield_latched:
            return super().plan(local_objects, cooperative_objects, ego, route)
        points = stop_hold_points(
            ego, route, braking_mps2=self.braking_mps2,
            horizon_s=self.config.horizon_s, step_s=self.config.step_s)
        placeholder = PlannedTrajectory(
            maneuver="keep", acceleration_mps2=self.braking_mps2,
            points=points, perceived_cost=CostVector(0, 0, 0, 0, 0, 0, 0, None))
        cost = self.evaluate(placeholder, GroundTruthWorld(0, ()), ego, route)
        return PlannedTrajectory("keep", self.braking_mps2, points, cost)
