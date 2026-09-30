import math
from dataclasses import replace
from types import SimpleNamespace
import pytest

from bces.oracle.world import WorldObject
from bces.simulation.controlled_contract import (center_from_front, receiver_state, ego_from_sample,
    payload_objects,make_message,local_observations,compare_snapshots,restore_control_modes)


def test_front_bumper_is_converted_to_box_center():
    assert center_from_front((10,0),0,4)==(8,0)
    assert center_from_front((0,10),math.pi/2,4)==pytest.approx((0,8))


def test_receiver_curvature_is_past_only_and_first_state_is_unavailable():
    vehicle=SimpleNamespace(getAngle=lambda _:80.,getLength=lambda _:4.8,getWidth=lambda _:1.9,
        getPosition=lambda _:(10,20),getSpeed=lambda _:5.,getAcceleration=lambda _:1.)
    connection=SimpleNamespace(vehicle=vehicle,simulation=SimpleNamespace(getTime=lambda:1.))
    first=receiver_state(connection)
    with pytest.raises(ValueError): ego_from_sample(first)
    state=receiver_state(connection,{'timestamp_ms':800,'heading_rad':0.})
    assert state['curvature_inv_m']==pytest.approx(math.radians(10)/.2/5)
    assert ego_from_sample(state).curvature_inv_m>0
    with pytest.raises(ValueError): receiver_state(connection,state)


def test_payload_reconstruction_cannot_access_hidden_truth_heading():
    obj=WorldObject('actor',10,0,3,0,0,4.8,1.9)
    rotated=replace(obj,heading_rad=1.2)
    a,b=make_message((obj,),100),make_message((rotated,),100)
    assert a.encode()==b.encode()
    assert payload_objects(a)==payload_objects(b)
    assert payload_objects(a)[0].heading_rad==0


def test_stopped_payload_box_encloses_every_unknown_orientation():
    obj=WorldObject('actor',0,0,0,0,1.2,4.8,1.9)
    estimate=payload_objects(make_message((obj,),100))[0]
    for angle in [i*math.pi/100 for i in range(200)]:
        for x in (-obj.length_m/2,obj.length_m/2):
            for y in (-obj.width_m/2,obj.width_m/2):
                assert abs(x*math.cos(angle)-y*math.sin(angle))<=estimate.length_m/2+1e-12
                assert abs(x*math.sin(angle)+y*math.cos(angle))<=estimate.width_m/2+1e-12


def test_local_visibility_is_explicit_and_self_is_not_inserted():
    from bces.oracle.world import EgoState
    objects=tuple(WorldObject(str(x),x,0,1,0,0,4,2) for x in (5,20,50))
    observed=local_observations(objects,EgoState(0,0,3,0),hidden_ids=('5',))
    assert [o.track_id for o in observed]==['20']


def test_checkpoint_comparison_includes_ego_discrete_state_and_numeric_tolerance():
    first={'ego':{'speed':2.,'lane':'a'},'actors':[1.,2.]}
    assert compare_snapshots(first,{'ego':{'speed':2.+1e-8,'lane':'a'},'actors':[1.,2.]})['equal_within_tolerance']
    assert not compare_snapshots(first,{'ego':{'speed':2.,'lane':'b'},'actors':[1.,2.]})['equal_within_tolerance']
    assert not compare_snapshots(first,{'ego':{'speed':3.,'lane':'a'},'actors':[1.,2.]})['equal_within_tolerance']
    assert not compare_snapshots(first,{'actors':[1.,2.]})['equal_within_tolerance']


def test_control_modes_are_restored_from_recorded_values_not_assumed_defaults():
    calls=[]
    domain=SimpleNamespace(getIDList=lambda:['ego'],
        setSpeedMode=lambda k,v:calls.append(('speed',k,v)),
        setLaneChangeMode=lambda k,v:calls.append(('lane',k,v)),
        setSpeedFactor=lambda k,v:calls.append(('factor',k,v)))
    connection=SimpleNamespace(vehicle=domain)
    snapshot={'vehicles':{'ego':{'speed_mode':0,'lane_change_mode':1621,'speed_factor':.9}}}
    restore_control_modes(connection,snapshot)
    assert calls==[('speed','ego',0),('lane','ego',1621),('factor','ego',.9)]
    with pytest.raises(ValueError): restore_control_modes(connection,{'vehicles':{}})
