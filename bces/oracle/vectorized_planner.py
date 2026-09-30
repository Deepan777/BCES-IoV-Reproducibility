"""Numerically checked array implementation of the frozen kinematic cost rule.

This adapter has a distinct policy binding. It preserves the candidate library
and lexicographic ordering; it never reads future ground truth during planning.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from bces.oracle.planner import KinematicPlanner
from bces.oracle.world import CostVector
from bces.utils.hashing import canonical_json_hash, sha256_file


class VectorizedDiagnosticPlanner(KinematicPlanner):
    def __init__(self, config=None):
        super().__init__(config)
        self.policy_hash = int(canonical_json_hash({'base_policy':self.policy_hash,
            'adapter_sha256':sha256_file(Path(__file__))})[:8],16)

    def evaluate(self, trajectory, world, ego, route):
        points = trajectory.points
        minimum_ttc, collision = math.inf, 0.0
        if world.objects:
            p = np.array([[q.time_s,q.x_m,q.y_m,q.heading_rad,q.speed_mps] for q in points])
            o = np.array([[q.x_m,q.y_m,q.vx_mps,q.vy_mps,q.heading_rad,q.length_m,q.width_m] for q in world.objects])
            dx = o[None,:,0]+p[:,None,0]*o[None,:,2]-p[:,None,1]
            dy = o[None,:,1]+p[:,None,0]*o[None,:,3]-p[:,None,2]
            ac, ass = np.cos(p[:,None,3]), np.sin(p[:,None,3])
            bc, bs = np.cos(o[None,:,4]), np.sin(o[None,:,4])
            margin = self.config.collision_margin_m
            al, aw = ego.length_m/2+margin, ego.width_m/2+margin
            bl, bw = o[None,:,5]/2+margin, o[None,:,6]/2+margin
            overlap = np.ones(dx.shape,dtype=bool)
            for ux,uy in ((ac,ass),(-ass,ac),(bc,bs),(-bs,bc)):
                distance = abs(dx*ux+dy*uy)
                ar = al*abs(ac*ux+ass*uy)+aw*abs(-ass*ux+ac*uy)
                br = bl*abs(bc*ux+bs*uy)+bw*abs(-bs*ux+bc*uy)
                overlap &= distance <= ar+br
            collision = float(overlap.any())
            along, lateral = ac*dx+ass*dy, abs(-ass*dx+ac*dy)
            closing = p[:,None,4]-(ac*o[None,:,2]+ass*o[None,:,3])
            relevant = (along>0)&(closing>1e-6)&(lateral <= (ego.width_m+o[None,:,6])/2+margin)
            if relevant.any():
                minimum_ttc = float(np.min(along[relevant]/closing[relevant]))
        ttc = 0.0 if math.isinf(minimum_ttc) else math.exp(-max(0.0,minimum_ttc)/2)
        final = points[-1]
        route_cost = abs(final.lateral_offset_m)/route.lane_width_m if trajectory.maneuver=='keep' else 0.0
        lane = max(0.,abs(final.lateral_offset_m)/(1.5*route.lane_width_m)-1)
        comfort = abs(trajectory.acceleration_mps2-ego.acceleration_mps2)/4+(0.25 if trajectory.maneuver!='keep' else 0.)
        desired = min(route.speed_limit_mps,max(ego.speed_mps,1.))*self.config.horizon_s
        progress = max(0.,desired-math.hypot(final.x_m-ego.x_m,final.y_m-ego.y_m))/max(desired,1e-6)
        components = dict(collision=collision,ttc=ttc,route=route_cost,lane=lane,comfort=comfort,progress=progress)
        total = sum(self.config.weight_map[key]*value for key,value in components.items())
        return CostVector(**components,total=total,minimum_ttc_s=None if math.isinf(minimum_ttc) else minimum_ttc)
