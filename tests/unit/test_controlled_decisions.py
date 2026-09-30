from dataclasses import replace
import pytest

from bces.geometry.drift import DriftScales
from bces.oracle.validity import ValidityThresholds
from bces.oracle.world import EgoState,WorldObject,GroundTruthWorld
from bces.simulation.controlled_contract import make_message
from bces.simulation.controlled_decisions import (ControlledDecisionPlanner,reference_contract,
    decision_pair,evaluate_recorded_future,route_for)


def test_recorded_constant_velocity_world_equals_existing_evaluator():
    planner=ControlledDecisionPlanner(); ego=EgoState(0,0,10,0)
    obj=WorldObject('actor',25,0,4,0,0,4,2)
    route=route_for(ego,'keep',planner); plan=planner.plan((),(obj,),ego,route)
    future={1000+round(p.time_s*1000):(obj.propagated(p.time_s),) for p in plan.points}
    expected=planner.evaluate(plan,GroundTruthWorld(1000,(obj,)),ego,route)
    actual=evaluate_recorded_future(planner,plan,future,ego,route,1000)
    assert actual.total==pytest.approx(expected.total)
    assert actual.risk==pytest.approx(expected.risk)
    with pytest.raises(KeyError): evaluate_recorded_future(planner,plan,{},ego,route,1000)


def test_future_label_world_cannot_change_reference_or_available_current_features():
    planner=ControlledDecisionPlanner(); ego=EgoState(0,0,10,0)
    message=make_message((),1000)
    kwargs=dict(scenario='test',split='train',behavior='keep',timestamp=1000,ego=ego,local=(),
                message=message,planner=planner,scales=DriftScales(40,5,5,.7,3,.1,4))
    ref,query,cached=reference_contract(**kwargs)
    route=route_for(ego,'keep',planner); plan=planner.plan((),(),ego,route)
    clear={1000+round(p.time_s*1000):() for p in plan.points}
    blocked={1000+round(p.time_s*1000):(WorldObject('actor',p.x_m,p.y_m,0,0,0,4,2),) for p in plan.points}
    common=dict(reference=ref,query=query,cached_reference=cached,ego=ego,local=(),fresh_message=message,
                planner=planner,thresholds=ValidityThresholds(5,.4,2))
    a=decision_pair(**common,future_world=clear); b=decision_pair(**common,future_world=blocked)
    assert a['valid'] and not b['valid']
    assert a['observable_current_features']==b['observable_current_features']
    assert reference_contract(**kwargs)[0]==ref
    assert a['cost_regret']==0 and a['trajectory_deviation_m']==0
    assert b['cost_regret']==0 and b['trajectory_deviation_m']==0


def test_reference_query_rejects_mismatched_payload_time():
    with pytest.raises(ValueError):
        reference_contract(scenario='x',split='train',behavior='keep',timestamp=1000,
            ego=EgoState(0,0,5,0),local=(),message=make_message((),1200),
            planner=ControlledDecisionPlanner(),scales=DriftScales(40,5,5,.7,3,.1,4))
