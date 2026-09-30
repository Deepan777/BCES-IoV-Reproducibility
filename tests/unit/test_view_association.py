from copy import deepcopy
from dataclasses import replace
import pytest

from bces.oracle.view_association import (AssociationConfig, associate_replay_views,
    audit_receiver_roles, complete_kinematics)


def row(key,x=0,y=0,tag='OTHERS'):
    return dict(id=str(key),x=str(x),y=str(y),theta='0',v_x='3',v_y='0',
                length='4.5',width='1.8',type='VEHICLE',tag=tag)


def snapshots(rows):
    return {t:tuple(deepcopy(rows)) for t in (0,100,200,300)}


def test_self_and_duplicate_use_different_source_ids_without_mutating_input():
    ego=snapshots([row(1,tag='TARGET_AGENT'),row(9,20)])
    local=snapshots([row(300,1),row(301,20)])
    remote=snapshots([row(200,1.1),row(201,20.2),row(202,40)])
    original=deepcopy(remote)
    l,r,a=associate_replay_views(ego,local,remote,'1')
    assert [x['id'] for x in l[200]] == ['1','vehicle:VEHICLE:301']
    assert [x['id'] for x in r[200]] == ['1','vehicle:VEHICLE:301','infrastructure:VEHICLE:202']
    assert [x['id'] for x in r[0]] == ['infrastructure:VEHICLE:200','infrastructure:VEHICLE:201','infrastructure:VEHICLE:202']
    assert remote==original
    assert a['counts']['cross_source_matches']==2


def test_ambiguous_nearby_vehicles_are_retained_not_deleted():
    ego=snapshots([row(1)])
    local=snapshots([row(3,.1),row(4,.2)])
    remote=snapshots([row(2,.15)])
    l,r,a=associate_replay_views(ego,local,remote,'1')
    assert [x['id'] for x in l[200]] == ['vehicle:VEHICLE:3','vehicle:VEHICLE:4']
    assert a['counts']['vehicle:self_matches']==0


def test_adjacent_lane_actor_and_missing_kinematics_not_self():
    bad=row(4,.1); bad['theta']=''
    ego=snapshots([row(1)])
    local=snapshots([row(3,.1,1.8),bad])
    l,_,_=associate_replay_views(ego,local,{},'1')
    assert all(x['id']!='1' for x in l[200])
    assert not complete_kinematics(bad)


def test_future_and_withheld_nonfocal_objects_cannot_change_past_matches():
    ego=snapshots([row(1),row(9,20)])
    local=snapshots([row(3,.1),row(5,20)])
    remote=snapshots([row(2,.2),row(6,20.1)])
    before=associate_replay_views(ego,local,remote,'1')
    ego[100]=(row(1),row(9,2000)); remote[300]=(row(2,1000),)
    after=associate_replay_views(ego,local,remote,'1')
    assert before[0][200]==after[0][200]
    assert before[1][200]==after[1][200]


def test_equal_numeric_ids_do_not_imply_cross_source_identity():
    ego=snapshots([row(1)])
    l,r,_=associate_replay_views(ego,snapshots([row(5,20)]),snapshots([row(5,40)]),'1')
    assert l[200][0]['id']=='vehicle:VEHICLE:5'
    assert r[200][0]['id']=='infrastructure:VEHICLE:5'


def test_missing_history_and_duplicate_ids_fail_closed():
    ego=snapshots([row(1)])
    local={0:(row(3,.1),),200:(row(3,.1),),300:(row(3,.1),)}
    l,_,_=associate_replay_views(ego,local,{},'1')
    assert l[300][0]['id']=='vehicle:VEHICLE:3'
    with pytest.raises(ValueError):
        associate_replay_views(ego,snapshots([row(3),row(3)]),{},'1')
    with pytest.raises(ValueError):
        replace(AssociationConfig(),history_frames=1)


def test_role_audit_does_not_promote_forecasting_target_or_impute_av():
    av=row(8,tag='AV'); av['v_x']=''; av['theta']=''
    audit=audit_receiver_roles(snapshots([row(1,tag='TARGET_AGENT'),av]),'1')
    assert not audit['receiver_role_verified']
    assert not audit['source_is_complete_ground_truth']
    assert audit['counts']['av_complete_kinematic_rows']==0


def test_same_source_numeric_id_reused_across_classes_preserves_both_actors():
    car=row(200009,20)
    pedestrian={**row(200009,50),'type':'PEDESTRIAN','length':'.6','width':'.5'}
    _,remote,_=associate_replay_views(snapshots([row(1)]),{},snapshots([car,pedestrian]),'1')
    assert len(remote[200])==2
    assert {r['id'] for r in remote[200]} == {
        'infrastructure:VEHICLE:200009','infrastructure:PEDESTRIAN:200009'}
