"""Corrected closed-loop engineering branches, not confirmatory safety results.

Independent complete prefixes initialize every branch. SUMO evolves the other
agents in response to the driven ego. The frozen kinematic planner is unchanged;
its compatibility with SUMO actuation is measured, not assumed.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from bces.geometry.drift import DriftScales, wrap_angle_rad
from bces.models.bound_policy import bind_reference, reference_query
from bces.network.bound_session import BoundSession, ReferenceUnavailable
from bces.network.wire_channel import WireChannel
from bces.oracle.planner import PlannerConfig, _rect_overlap
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import decode_objects
from bces.protocol.receiver import DecisionState
from bces.simulation.branching import _populate
from bces.simulation.controlled_contract import (start_controlled,receiver_state,centered_world,
    ego_from_sample,local_observations,make_message,payload_objects,checkpoint_snapshot)
from bces.simulation.controlled_decisions import ControlledDecisionPlanner,reference_contract,kinematic,route_for
from bces.oracle.observational import BEHAVIOR_IDS
from bces.simulation.controlled_slots import development_generator

ROOT=Path(__file__).resolve().parents[2]
EGO_LANE_CHANGE_MODE=512


def configure_ego_lane_actuation(connection):
    """Prevent uncommanded SUMO lane changes, retaining TraCI safety gaps.

    SUMO mode 512 disables autonomous strategic/cooperative/speed/rightward
    changes while respecting speed/brake gaps for explicit changeLane requests.
    It must be applied before the branch-common prefix, not after branching.
    """
    connection.vehicle.setLaneChangeMode('ego',EGO_LANE_CHANGE_MODE)
    observed=connection.vehicle.getLaneChangeMode('ego')
    if observed!=EGO_LANE_CHANGE_MODE:
        raise RuntimeError(f'ego lane-change mode not applied: {observed}')


def drive_candidate(connection, plan, step_s):
    """Direct speed/lane adapter under audit; never pretend illegal moves execute."""
    connection.vehicle.setSpeed('ego',plan.points[0].speed_mps)
    result={'requested_speed_mps':plan.points[0].speed_mps,'maneuver':plan.maneuver,
            'lane_request_applied':False,'illegal_lane_request':False}
    if plan.maneuver in ('left','right'):
        edge=connection.vehicle.getRoadID('ego')
        lane=connection.vehicle.getLaneIndex('ego')
        target=lane+(1 if plan.maneuver=='left' else -1)
        if edge.startswith(':') or not 0<=target<connection.edge.getLaneNumber(edge):
            result['illegal_lane_request']=True
        else:
            connection.vehicle.changeLane('ego',target,step_s)
            result['lane_request_applied']=True
    return result


def instantaneous_events(ego,objects):
    overlaps=[]; ttc=None
    for o in objects:
        if _rect_overlap(ego.x_m,ego.y_m,ego.heading_rad,ego.length_m,ego.width_m,
                         o.x_m,o.y_m,o.heading_rad,o.length_m,o.width_m,0.):
            overlaps.append(o.track_id)
        dx,dy=o.x_m-ego.x_m,o.y_m-ego.y_m
        along=math.cos(ego.heading_rad)*dx+math.sin(ego.heading_rad)*dy
        lateral=abs(-math.sin(ego.heading_rad)*dx+math.cos(ego.heading_rad)*dy)
        closing=ego.speed_mps-(math.cos(ego.heading_rad)*o.vx_mps+math.sin(ego.heading_rad)*o.vy_mps)
        if along>0 and closing>1e-6 and lateral<=(ego.width_m+o.width_m)/2:
            value=max(0.,along-(ego.length_m+o.length_m)/2)/closing
            ttc=value if ttc is None else min(ttc,value)
    return {'geometric_overlap_ids':overlaps,'longitudinal_corridor_ttc_s':ttc}


def run_branch(*,seed,method,config,condition,network_seed,model=None,rule=None,horizon_s=6.):
    if method not in ('surface','scalar_ttl','periodic_payload','local_only'):
        raise ValueError('unregistered pilot method')
    if method in ('surface','scalar_ttl') and (model is None or rule is None):
        raise ValueError('frozen policy required')
    generator=development_generator()
    spec,parameters=generator.sample_scenario(seed,config)
    planner=ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT/config['planner']))
    scales=DriftScales(*config['drift_scales'])
    connection=start_controlled(network=ROOT/config['network'],label=f'controlled_pilot_{method}_{seed}_{network_seed}',
                                seed=seed,step_s=config['step_s'])
    history={}; previous=None; event_applied=False
    channel=WireChannel(condition,seed=network_seed,security_bytes=64)
    hidden=(('person:' if spec.pedestrian else 'vehicle:')+spec.actor_id,) if spec.hidden_from_local else ()
    counts=Counter(); rows=[]; input_audit=[]
    def observe():
        nonlocal previous
        if 'ego' not in connection.vehicle.getIDList(): return None
        sample=receiver_state(connection,previous); previous=sample
        objects=centered_world(connection)
        local=() if not sample['curvature_available'] else local_observations(objects,ego_from_sample(sample),
                    range_m=config['local_range_m'],hidden_ids=hidden)
        history[sample['timestamp_ms']]={'receiver':sample,'objects':objects,'local':local}
        return sample
    def exogenous():
        nonlocal event_applied
        now=connection.simulation.getTime()
        if spec.uncontrolled_signal:
            for signal in connection.trafficlight.getIDList():
                value=connection.trafficlight.getRedYellowGreenState(signal)
                connection.trafficlight.setRedYellowGreenState(signal,'G'*len(value))
        if (spec.braking_event and not event_applied and
                now+1e-9>=config['reference_time_s']+parameters['braking_event_delay_s'] and
                spec.actor_id in connection.vehicle.getIDList()):
            connection.vehicle.slowDown(spec.actor_id,0.,1.);event_applied=True
    def provide_payload(available_ms):
        timestamp=max(t for t in history if t<=available_ms)
        frame=history[timestamp]; ego=ego_from_sample(frame['receiver'])
        cooperative=tuple(o for o in frame['objects'] if math.hypot(o.x_m-ego.x_m,o.y_m-ego.y_m)<=config['cooperative_range_m'])
        input_audit.append({'role':'sender','used_timestamp_ms':timestamp,'available_ms':available_ms})
        return make_message(cooperative,timestamp,message_id=seed).encode()
    def build_query(payload,query_id,arrival):
        message=decode_objects(payload)
        frame=history.get(message.timestamp_ms)
        if frame is None: raise ReferenceUnavailable('no recorded receiver history for payload timestamp')
        if message.timestamp_ms>arrival: raise AssertionError('future receiver state')
        try:
            reference,_,_=reference_contract(scenario=str(seed),split='closed_loop_pilot',
                behavior=spec.intended_behavior,timestamp=message.timestamp_ms,
                ego=ego_from_sample(frame['receiver']),local=frame['local'],message=message,planner=planner,scales=scales)
        except RuntimeError as exc:
            if str(exc)!='no feasible trajectory candidate': raise
            raise ReferenceUnavailable(str(exc)) from exc
        query=reference_query(reference,query_id=query_id,sender_hash=sender_id_hash(message.sender_id))
        bound=bind_reference(reference,query,payload,model_sha256=model.sha256,
            view=model.view,family=method,shrinkage=rule['shrinkage'])
        input_audit.append({'role':'receiver','used_timestamp_ms':message.timestamp_ms,'available_ms':arrival})
        return replace(bound,generated_ms=arrival,payload_received_ms=arrival,context_available_ms=arrival)
    app=None; latest_payload=None; outcome='horizon_complete'; initial=None
    try:
        _populate(connection,spec)
        configure_ego_lane_actuation(connection)
        # Exact same controls/history as the preserved development prefix.
        for _ in range(round(config['reference_time_s']/config['step_s'])):
            if 'ego' in connection.vehicle.getIDList():
                connection.vehicle.setSpeed('ego',parameters['ego_initial_speed_mps'])
            exogenous();connection.simulationStep();observe()
        if previous is None or previous['timestamp_ms']!=round(config['reference_time_s']*1000):
            raise RuntimeError('receiver unavailable at pilot branch point')
        initial={'snapshot':checkpoint_snapshot(connection),'receiver_estimator':previous,
                 'local_observations':[asdict(o) for o in history[previous['timestamp_ms']]['local']],
                 'parameters':parameters,'policy_hash':planner.policy_hash,'braking_event_applied':event_applied}
        start_distance=connection.vehicle.getDistance('ego');last_distance=start_distance
        if model is not None:
            if planner.policy_hash!=model.policy_hash: raise ValueError('pilot planner/checkpoint mismatch')
            app=BoundSession(channel=channel,policy=model,build_query=build_query,payload_provider=provide_payload,
                sender_id='controlled:rsu',computation_delay_ms=6.,timeout_ms=400.,max_retries=1)
        for _ in range(round(horizon_s/config['step_s'])):
            sample=previous; now=sample['timestamp_ms'];frame=history[now];ego=ego_from_sample(sample)
            cooperative=();reuse=False
            if app is not None:
                decision=app.decision(kinematic(ego,now),BEHAVIOR_IDS[spec.intended_behavior])
                reuse=decision.state==DecisionState.ACCEPT_REUSE
                if reuse: latest_payload=app.receiver.payload
            elif method=='periodic_payload':
                channel.send('payload',provide_payload(now),now)
                while (event:=channel.next_delivery(now)) is not None:
                    message=decode_objects(event.body)
                    if latest_payload is None or message.timestamp_ms>decode_objects(latest_payload).timestamp_ms:
                        latest_payload=event.body
                    else: channel.discard(event)
                reuse=latest_payload is not None
            if reuse:
                message=decode_objects(latest_payload)
                age=(now-message.timestamp_ms)/1000
                if age<0: raise AssertionError('future payload used in planning')
                cooperative=tuple(o.propagated(age) for o in payload_objects(message))
            try:
                plan=planner.plan(frame['local'],cooperative,ego,route_for(ego,spec.intended_behavior,planner))
            except RuntimeError as exc:
                if str(exc)!='no feasible trajectory candidate': raise
                counts['planner_infeasible']+=1;outcome='planner_infeasible';break
            command=drive_candidate(connection,plan,config['step_s'])
            counts['illegal_lane_requests']+=command['illegal_lane_request']
            counts['reuse_decisions']+=reuse;counts['local_only_decisions']+=not reuse
            exogenous();connection.simulationStep()
            collisions=list(connection.simulation.getCollidingVehiclesIDList())
            actual=observe()
            if actual is None:
                outcome='route_completed' if 'ego' in connection.simulation.getArrivedIDList() else 'receiver_disappeared'
                rows.append({'timestamp_ms':round(connection.simulation.getTime()*1000),'outcome':outcome,
                             'sumo_colliding_vehicle_ids':collisions,'command':command})
                break
            current=ego_from_sample(actual); expected=plan.points[0]
            tracking={'position_error_m':math.hypot(actual['x_m']-expected.x_m,actual['y_m']-expected.y_m),
                      'speed_error_mps':abs(actual['speed_mps']-expected.speed_mps),
                      'heading_error_rad':abs(wrap_angle_rad(actual['heading_rad']-expected.heading_rad))}
            violation=tracking['position_error_m']>.5 or tracking['speed_error_mps']>.5 or tracking['heading_error_rad']>.1
            counts['tracking_violations']+=violation
            events=instantaneous_events(current,history[actual['timestamp_ms']]['objects'])
            counts['geometric_overlap_ticks']+=bool(events['geometric_overlap_ids'])
            counts['sumo_ego_collision_ticks']+='ego' in collisions
            last_distance=connection.vehicle.getDistance('ego')
            rows.append({'timestamp_ms':actual['timestamp_ms'],'receiver':actual,'reused_payload':reuse,
                'command':command,'tracking':tracking,'events':events,'sumo_colliding_vehicle_ids':collisions,
                'jerk_mps3':(actual['acceleration_mps2']-sample['acceleration_mps2'])/config['step_s'],
                'route_distance_m':last_distance-start_distance})
        end_ms=round(connection.simulation.getTime()*1000)
        if app is not None: app.advance(end_ms)
        else:
            while channel.next_delivery(end_ms) is not None: pass
        traffic=app.report() if app is not None else channel.report()
        duration=(end_ms-round(config['reference_time_s']*1000))/1000
        traffic['transmitted_bytes_per_second']=traffic['generated_bytes']/duration if duration else None
        return {'scenario_id':str(seed),'method':method,'family':spec.family,'outcome':outcome,
            'initial_contract':initial,'counts':dict(counts),'rows':rows,'traffic':traffic,
            'input_availability_audit':input_audit,'duration_s':duration,'route_distance_lower_bound_m':last_distance-start_distance,
            'actuation_contract_passed':not counts['illegal_lane_requests'] and not counts['tracking_violations']
                and not counts['planner_infeasible'] and outcome!='receiver_disappeared',
            'planner_map_compatibility_observed':not counts['illegal_lane_requests'] and not counts['tracking_violations'],
            'tracking_review_thresholds':{'position_m':.5,'speed_mps':.5,'heading_rad':.1},
            'scientific_success':False,'manuscript_allowed':False,
            'limitations':['engineering pilot using designed scenarios, not powered safety confirmation',
                'lane/speed adapter is under audit; failed actuation prevents driving-benefit claims',
                'turn/lane-change semantics must match the map route; incompatibility requires policy revision and retraining',
                'geometric contacts sampled every 0.2 seconds; longitudinal TTC does not cover every crossing interaction',
                'same network seed/distribution, not identical packet losses for unequal traffic',
                '6 ms simulated sender delay is a declared model, not a real-time hardware guarantee']}
    finally:
        connection.close()
