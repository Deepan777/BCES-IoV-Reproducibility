"""Auditable state and sensing contract for new controlled SUMO experiments.

Historical adapters are intentionally unchanged. Payload reconstruction has no
access to ground-truth heading; simulator coordinates use bounding-box centers.
"""
from __future__ import annotations

import math
from dataclasses import replace

from bces.data.objects import ObjectMessage, ObjectSource, PerceivedObject
from bces.geometry.drift import wrap_angle_rad
from bces.oracle.world import EgoState, WorldObject
from bces.simulation.sumo_adapter import heading_rad, load_traci, sumo_binary


def center_from_front(position, heading, length):
    return (position[0]-.5*length*math.cos(heading),
            position[1]-.5*length*math.sin(heading))


def receiver_state(connection, previous=None):
    """Measured simulated pose/speed, past-step acceleration, causal curvature.

Curvature is the explicitly regularized estimator yaw_rate / max(speed, 2).
The first sample has no curvature estimate and cannot form a reference query.
"""
    domain=connection.vehicle
    heading=heading_rad(domain.getAngle('ego'))
    length,width=domain.getLength('ego'),domain.getWidth('ego')
    x,y=center_from_front(domain.getPosition('ego'),heading,length)
    timestamp=round(connection.simulation.getTime()*1000)
    curvature=None
    if previous is not None:
        dt=(timestamp-previous['timestamp_ms'])/1000
        if dt<=0:
            raise ValueError('receiver estimator needs strictly increasing time')
        curvature=wrap_angle_rad(heading-previous['heading_rad'])/dt/max(domain.getSpeed('ego'),2.)
    return {'timestamp_ms':timestamp,'x_m':x,'y_m':y,'heading_rad':heading,
            'speed_mps':domain.getSpeed('ego'),'acceleration_mps2':domain.getAcceleration('ego'),
            'curvature_inv_m':curvature,'length_m':length,'width_m':width,
            'curvature_available':curvature is not None,'latest_input_timestamp_ms':timestamp}


def ego_from_sample(sample):
    if not sample['curvature_available'] or sample['curvature_inv_m'] is None:
        raise ValueError('reference state requires past/current curvature inputs')
    return EgoState(**{key:sample[key] for key in (
        'x_m','y_m','speed_mps','heading_rad','acceleration_mps2','curvature_inv_m','length_m','width_m')})


def centered_world(connection):
    objects=[]
    for key in sorted(connection.vehicle.getIDList()):
        if key=='ego':
            continue
        d=connection.vehicle
        h=heading_rad(d.getAngle(key)); length=d.getLength(key)
        x,y=center_from_front(d.getPosition(key),h,length)
        speed=d.getSpeed(key)
        objects.append(WorldObject('vehicle:'+key,x,y,speed*math.cos(h),speed*math.sin(h),
                                   h,length,d.getWidth(key),source='simulated_truth'))
    for key in sorted(connection.person.getIDList()):
        d=connection.person; x,y=d.getPosition(key); h=heading_rad(d.getAngle(key)); speed=d.getSpeed(key)
        objects.append(WorldObject('person:'+key,x,y,speed*math.cos(h),speed*math.sin(h),
                                   h,d.getLength(key),d.getWidth(key),class_id=1,source='simulated_truth'))
    return tuple(objects)


def make_message(objects, timestamp_ms, message_id=0, sender='controlled:rsu'):
    return ObjectMessage(message_id,sender,timestamp_ms,tuple(
        PerceivedObject(o.track_id,o.class_id,o.x_m,o.y_m,o.vx_mps,o.vy_mps,
                        o.length_m,o.width_m,o.confidence,ObjectSource.INFRASTRUCTURE) for o in objects))


def payload_objects(message):
    """Reconstruct exclusively from transmitted fields.

Moving-heading alignment is a declared no-side-slip sensing assumption. With
speed <= 0.1 m/s, orientation is unavailable: an enclosing square contains the
rectangle at every orientation. This is not a bound on unknown object motion.
"""
    result=[]
    for o in message.objects:
        moving=math.hypot(o.vx_mps,o.vy_mps)>.1
        heading=math.atan2(o.vy_mps,o.vx_mps) if moving else 0.
        length,width=(o.length_m,o.width_m) if moving else (math.hypot(o.length_m,o.width_m),)*2
        result.append(WorldObject(o.track_id,o.x_m,o.y_m,o.vx_mps,o.vy_mps,heading,
                                  length,width,o.class_id,o.confidence,'cooperative_payload'))
    return tuple(result)


def local_observations(objects, ego, *, range_m=30., hidden_ids=()):
    """Controlled ideal local sensing with declared radius and occlusion mask."""
    return tuple(replace(o,source='local_sensor') for o in objects if o.track_id not in hidden_ids
                 and math.hypot(o.x_m-ego.x_m,o.y_m-ego.y_m)<=range_m)


def checkpoint_snapshot(connection):
    """Observable checkpoint signature; continuation replay tests hidden state."""
    vehicles={}
    for key in sorted(connection.vehicle.getIDList()):
        d=connection.vehicle
        vehicles[key]={'position':list(d.getPosition(key)),'speed':d.getSpeed(key),
            'acceleration':d.getAcceleration(key),'heading':d.getAngle(key),
            'length':d.getLength(key),'width':d.getWidth(key),'lane':d.getLaneID(key),
            'lane_position':d.getLanePosition(key),'route':list(d.getRoute(key)),
            'route_index':d.getRouteIndex(key),'speed_mode':d.getSpeedMode(key),
            'lane_change_mode':d.getLaneChangeMode(key),'speed_factor':d.getSpeedFactor(key)}
    persons={}
    for key in sorted(connection.person.getIDList()):
        d=connection.person
        persons[key]={'position':list(d.getPosition(key)),'speed':d.getSpeed(key),
                      'heading':d.getAngle(key),'road':d.getRoadID(key)}
    signals={}
    for key in sorted(connection.trafficlight.getIDList()):
        d=connection.trafficlight
        signals[key]={'state':d.getRedYellowGreenState(key),'phase':d.getPhase(key),
                      'program':d.getProgram(key),'next_switch':d.getNextSwitch(key)}
    return {'time_s':connection.simulation.getTime(),'vehicles':vehicles,'persons':persons,'signals':signals}


def restore_control_modes(connection,snapshot):
    """Reapply recorded TraCI control settings omitted by the checkpoint.

This is not restoration of arbitrary pending commands, controller memory or
pedestrian model state. Those must be validated separately or prefix-replayed.
"""
    if set(connection.vehicle.getIDList()) != set(snapshot['vehicles']):
        raise ValueError('cannot restore controls across different vehicle sets')
    for key,values in snapshot['vehicles'].items():
        connection.vehicle.setSpeedMode(key,values['speed_mode'])
        connection.vehicle.setLaneChangeMode(key,values['lane_change_mode'])
        connection.vehicle.setSpeedFactor(key,values['speed_factor'])


def compare_snapshots(first, second, tolerance=1e-6):
    differences=[]; maximum=0.
    def compare(a,b,path):
        nonlocal maximum
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):
                differences.append(path+':keys')
            for key in sorted(set(a)&set(b)):
                compare(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            if len(a)!=len(b):
                differences.append(path+':length')
            for index,(x,y) in enumerate(zip(a,b)):
                compare(x,y,path+'/'+str(index))
        elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
            error=abs(a-b)
            if not math.isfinite(error):
                differences.append(path+':nonfinite')
            else:
                maximum=max(maximum,error)
                if error>tolerance:
                    differences.append(path)
        elif a!=b:
            differences.append(path)
    compare(first,second,'root')
    return {'equal_within_tolerance':not differences,'maximum_numeric_error':maximum,
            'difference_paths':differences,'tolerance':tolerance}


def start_controlled(*,network, label, seed, step_s, state=None):
    command=[str(sumo_binary()),'-n',str(network),'--step-length',str(step_s),'--seed',str(seed),
        '--no-step-log','true','--no-warnings','true','--time-to-teleport','-1',
        '--collision.action','warn','--save-state.rng','true','--save-state.transportables','true',
        '--save-state.precision','12','--thread-rngs','1']
    if state is not None:
        command.extend(['--load-state',str(state)])
    traci=load_traci()
    traci.start(command,label=label)
    return traci.getConnection(label)
