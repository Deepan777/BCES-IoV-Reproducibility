"""Development-only cruise-recovery planner; never substitutes for v1/v2 policy.

The reference speed is the first receiver speed observed at the branch point.
It is causal and thereafter fixed. The registered candidate library and
collision/TTC-first ranking are inherited unchanged; only progress scoring
differs. Any BCES policy trained under another planner hash is incompatible.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from bces.oracle.world import GroundTruthWorld
from bces.simulation.controlled_decisions import ControlledDecisionPlanner
from bces.utils.hashing import canonical_json_hash, sha256_file


class DevelopmentCruiseRecoveryPlanner(ControlledDecisionPlanner):
    progress_weight = 4.0

    def __init__(self, config=None):
        super().__init__(config)
        self.reference_speed_mps: float | None = None
        self.policy_hash = int(canonical_json_hash({
            "base_policy_hash": self.policy_hash,
            "recovery_source_sha256": sha256_file(Path(__file__)),
            "causal_target": "first_branch_receiver_speed",
            "progress_weight": self.progress_weight,
        })[:8], 16)

    def plan(self, local_objects, cooperative_objects, ego, route):
        if self.reference_speed_mps is None:
            self.reference_speed_mps = min(route.speed_limit_mps, ego.speed_mps)
        return super().plan(local_objects, cooperative_objects, ego, route)

    def evaluate(self, trajectory, world: GroundTruthWorld, ego, route):
        base = super().evaluate(trajectory, world, ego, route)
        target_speed = self.reference_speed_mps
        if target_speed is None:
            # Evaluation before the first plan is deterministic and does not
            # mutate the policy state. The branch runner calls plan first.
            target_speed = min(route.speed_limit_mps, ego.speed_mps)
        desired = max(1.0, min(route.speed_limit_mps, target_speed)) * self.config.horizon_s
        final = trajectory.points[-1]
        distance = math.hypot(final.x_m - ego.x_m, final.y_m - ego.y_m)
        progress = max(0.0, desired - distance) / max(desired, 1e-6)
        total = (base.total
                 - self.config.weight_map["progress"] * base.progress
                 + self.progress_weight * progress)
        return replace(base, progress=progress, total=total)
