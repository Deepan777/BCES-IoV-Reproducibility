"""Payload-faithful decision pairs on one recorded simulated world trajectory.

These are discrete-time open-loop counterfactual labels, not closed-loop
collisions. Future world frames enter only the label evaluator, never planning.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from bces.geometry.drift import KinematicState,compute_drift
from bces.models.reference_input import freeze_reference_input
from bces.oracle.causal_diagnostics import observable_planner_features
from bces.oracle.observational import BEHAVIOR_IDS
from bces.oracle.planner import trajectory_deviation
from bces.oracle.validity import ValidityObservation,evaluate_validity
from bces.oracle.vectorized_planner import VectorizedDiagnosticPlanner
from bces.oracle.world import GroundTruthWorld,Route
from bces.protocol.bindings import digest32,sender_id_hash
from bces.protocol.query import Behavior,PathPoint,ReceiverQuery
from bces.simulation.controlled_contract import make_message,payload_objects
from bces.utils.hashing import canonical_json_hash,sha256_file


class ControlledDecisionPlanner(VectorizedDiagnosticPlanner):
    def __init__(self,config=None):
        super().__init__(config)
        self.policy_hash=int(canonical_json_hash({'base_policy':self.policy_hash,
            'controlled_source':sha256_file(Path(__file__)),
            'sensing_source':sha256_file(Path(__file__).with_name('controlled_contract.py'))})[:8],16)


def kinematic(ego,timestamp):
    return KinematicState(ego.x_m,ego.y_m,ego.speed_mps,ego.heading_rad,
                          ego.acceleration_mps2,ego.curvature_inv_m,timestamp)


def route_for(ego,behavior,planner):
    return Route(ego.x_m,ego.y_m,ego.heading_rad,planner.config.lane_width_m,
                 planner.config.speed_limit_mps,behavior=='left',behavior=='right',
                 behavior,ego.curvature_inv_m)


def reference_contract(*,scenario,split,behavior,timestamp,ego,local,message,planner,scales):
    if message.timestamp_ms!=timestamp:
        raise ValueError('reference payload and query must refer to the same sampled time')
    cooperative=payload_objects(message)
    route=route_for(ego,behavior,planner)
    plan=planner.plan(local,cooperative,ego,route)
    identity=f'controlled:{scenario}:{behavior}:{timestamp}'
    query=ReceiverQuery(query_id=digest32((identity+':query').encode()),
        expected_sender_id_hash=sender_id_hash(message.sender_id),
        reference_state=kinematic(ego,timestamp),
        reference_path=tuple(PathPoint(p.x_m,p.y_m,p.speed_mps,p.heading_rad,
                                       route.curvature_inv_m,p.time_s) for p in plan.points[:12]),
        behavior=Behavior(BEHAVIOR_IDS[behavior]),risk_class=1,policy_hash=planner.policy_hash,
        calibration_id=0,drift_scales=scales)
    frozen=freeze_reference_input(message,query,generated_at_ms=timestamp,
        query_available_at_ms=timestamp,context_available_at_ms=timestamp,
        query_provenance='frozen_reference_policy',
        occlusion_proxy=max(0.,min(1.,(len(cooperative)-len(local))/max(len(cooperative),1))),
        estimated_delay_s=0.,map_context_flags=1)
    margins=observable_planner_features(planner,local,cooperative,ego,route)
    reference={'reference_id':identity,'scenario_id':str(scenario),'split':split,
        'behavior':behavior,'behavior_id':BEHAVIOR_IDS[behavior],'reference_timestamp_ms':timestamp,
        'policy_hash':planner.policy_hash,'frozen_input':frozen.to_dict(),
        'input_contract':'controlled_payload_only_causal_receiver_v1',
        'label_source':'controlled_simulation_open_loop','receiver_role_verified':True,
        'observable_margin_features':margins,'additional_query_float32_bytes':4*len(margins),
        'margin_sidecar_sha256':canonical_json_hash(margins),
        'margin_sidecar_wire_binding_integrated':False,
        'payload_bytes_actual_json':len(message.encode())}
    return reference,query,cooperative


def evaluate_recorded_future(planner,trajectory,future_world,ego,route,timestamp):
    """Use the same label-only, time-varying world for both planned paths."""
    base=planner.evaluate(trajectory,GroundTruthWorld(timestamp,()),ego,route)
    collision=0.; minimum_ttc=math.inf
    for p in trajectory.points:
        time=timestamp+round(p.time_s*1000)
        if time not in future_world:
            raise KeyError(f'missing label-world frame at {time}')
        # evaluate() propagates by p.time_s. Shift the actual future frame back
        # algebraically so its positions at this one evaluation step are exact.
        shifted=tuple(replace(o,x_m=o.x_m-o.vx_mps*p.time_s,
                              y_m=o.y_m-o.vy_mps*p.time_s) for o in future_world[time])
        cost=planner.evaluate(replace(trajectory,points=(p,)),GroundTruthWorld(time,shifted),ego,route)
        collision=max(collision,cost.collision)
        if cost.minimum_ttc_s is not None:
            minimum_ttc=min(minimum_ttc,cost.minimum_ttc_s)
    ttc=0. if math.isinf(minimum_ttc) else math.exp(-minimum_ttc/2.)
    return replace(base,collision=collision,ttc=ttc,
                   total=base.total+planner.config.weight_map['collision']*collision+planner.config.weight_map['ttc']*ttc,
                   minimum_ttc_s=None if math.isinf(minimum_ttc) else minimum_ttc)


def decision_pair(*,reference,query,cached_reference,ego,local,fresh_message,
                  future_world,planner,thresholds):
    timestamp=fresh_message.timestamp_ms
    age=(timestamp-query.reference_state.timestamp_ms)/1000
    if age<0:
        raise ValueError('decision cannot precede its reference message')
    cached=tuple(o.propagated(age) for o in cached_reference)
    fresh=payload_objects(fresh_message)
    route=route_for(ego,reference['behavior'],planner)
    cached_plan=planner.plan(local,cached,ego,route)
    fresh_plan=planner.plan(local,fresh,ego,route)
    cached_cost=evaluate_recorded_future(planner,cached_plan,future_world,ego,route,timestamp)
    fresh_cost=evaluate_recorded_future(planner,fresh_plan,future_world,ego,route,timestamp)
    deviation=trajectory_deviation(cached_plan,fresh_plan)
    observation=ValidityObservation(cached_cost.total,fresh_cost.total,cached_cost.risk,deviation)
    return {'point_id':f"{reference['reference_id']}:{timestamp}",
        'reference_id':reference['reference_id'],'scenario_id':reference['scenario_id'],
        'split':reference['split'],'behavior':reference['behavior'],'current_timestamp_ms':timestamp,
        'cache_age_s':age,'normalized_drift':compute_drift(query.reference_state,kinematic(ego,timestamp),query.drift_scales).normalized,
        'valid':evaluate_validity(observation,thresholds),'cost_regret':observation.cost_regret,
        'cached_risk':cached_cost.risk,'cached_cost':cached_cost.to_dict(),'fresh_cost':fresh_cost.to_dict(),
        'trajectory_deviation_m':deviation,'policy_hash':planner.policy_hash,
        'cached_acceleration_mps2':cached_plan.acceleration_mps2,'fresh_acceleration_mps2':fresh_plan.acceleration_mps2,
        'observable_current_features':observable_planner_features(planner,local,cached,ego,route),
        'observable_features_latest_ms':timestamp,'label_source':'controlled_simulation_open_loop',
        'future_world_used_only_for_labels':True,'same_label_world':True,
        'violations':{'regret':observation.cost_regret>thresholds.max_cost_regret+1e-10,
            'absolute_risk':cached_cost.risk>thresholds.max_cached_risk+1e-10,
            'deviation':deviation>thresholds.max_trajectory_deviation_m+1e-10}}
