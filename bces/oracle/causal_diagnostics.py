"""Past-only kinematics and observable planner diagnostics for Study B."""
from __future__ import annotations

import math
from bces.geometry.drift import KinematicState, wrap_angle_rad
from bces.oracle.observational import _ego
from bces.oracle.world import CostVector, GroundTruthWorld, PlannedTrajectory


def causal_kinematic_states(rows, history_frames=5):
    if history_frames < 1:
        raise ValueError("positive history required")
    times = sorted(rows)
    states = {}
    for index, timestamp in enumerate(times):
        previous_time = times[max(0, index-history_frames)]
        previous, current = _ego(rows[previous_time]), _ego(rows[timestamp])
        seconds = max((timestamp-previous_time)/1000, 1e-6)
        acceleration = max(-8, min(8, (current.speed_mps-previous.speed_mps)/seconds))
        yaw_rate = wrap_angle_rad(current.heading_rad-previous.heading_rad)/seconds
        curvature = max(-.3, min(.3, yaw_rate/max(current.speed_mps, 2.0)))
        states[timestamp] = KinematicState(current.x_m, current.y_m, current.speed_mps,
            current.heading_rad, acceleration, curvature, timestamp)
    return states


def rank_key(candidate):
    cost = candidate.perceived_cost
    return (cost.collision, cost.ttc, cost.total, candidate.maneuver, candidate.acceleration_mps2)


def observable_planner_features(planner, local, cooperative, ego, route):
    """No fresh or truth state. Candidate costs follow the frozen planner exactly."""
    objects = {item.track_id: item for item in cooperative}
    objects.update({item.track_id: item for item in local})
    world = GroundTruthWorld(0, tuple(objects[key] for key in sorted(objects)))
    candidates = []
    for maneuver in planner.config.lateral_maneuvers:
        for acceleration in planner.config.longitudinal_accelerations_mps2:
            points = planner._candidate(ego, route, acceleration, maneuver)
            if points is None:
                continue
            trajectory = PlannedTrajectory(maneuver, acceleration, points, CostVector(0,0,0,0,0,0,0,None))
            cost = planner.evaluate(trajectory, world, ego, route)
            candidates.append(PlannedTrajectory(maneuver, acceleration, points, cost))
    if not candidates:
        raise RuntimeError("no feasible trajectory candidate")
    candidates.sort(key=rank_key)
    selected = candidates[0]
    best = selected.perceived_cost
    feature = [float(len(candidates)), float(len(local)), float(len(cooperative)),
               selected.acceleration_mps2, best.collision, best.ttc, best.total,
               best.comfort, best.progress]
    # Rank-aware differences may be negative in a lower-priority component.
    for index in range(1, 5):
        if index < len(candidates):
            other = candidates[index].perceived_cost
            feature.extend([1.0, other.collision-best.collision, other.ttc-best.ttc,
                            other.total-best.total, candidates[index].acceleration_mps2])
        else:
            feature.extend([0.0]*5)
    if not all(math.isfinite(x) for x in feature):
        raise ValueError("nonfinite observable planner feature")
    return feature
