from types import SimpleNamespace

from bces.oracle.world import EgoState, WorldObject
from bces.simulation.controlled_closed_loop import (
    EGO_LANE_CHANGE_MODE, configure_ego_lane_actuation, drive_candidate, instantaneous_events,
)


class Vehicle:
    def __init__(self,edge='E0',lane=0,lanes=2):
        self.edge,self.lane,self.lanes=edge,lane,lanes
        self.speed_commands=[];self.lane_commands=[]
    def setSpeed(self,key,value): self.speed_commands.append((key,value))
    def getRoadID(self,key): return self.edge
    def getLaneIndex(self,key): return self.lane
    def changeLane(self,key,target,duration): self.lane_commands.append((key,target,duration))
    def setLaneChangeMode(self,key,value): self.lane_mode=(key,value)
    def getLaneChangeMode(self,key): return self.lane_mode[1]


def connection(edge='E0',lane=0,lanes=2):
    vehicle=Vehicle(edge,lane,lanes)
    return SimpleNamespace(vehicle=vehicle,edge=SimpleNamespace(getLaneNumber=lambda key:lanes))


def plan(maneuver='keep',speed=8.):
    return SimpleNamespace(maneuver=maneuver,points=(SimpleNamespace(speed_mps=speed),))


def test_drive_adapter_applies_speed_and_legal_lane_request():
    c=connection(lane=0,lanes=2)
    result=drive_candidate(c,plan('left'),.2)
    assert c.vehicle.speed_commands==[('ego',8.)]
    assert c.vehicle.lane_commands==[('ego',1,.2)]
    assert result['lane_request_applied'] and not result['illegal_lane_request']


def test_drive_adapter_records_but_does_not_send_illegal_request():
    c=connection(edge=':junction',lane=0,lanes=1)
    result=drive_candidate(c,plan('right'),.2)
    assert result['illegal_lane_request'] and not result['lane_request_applied']
    assert not c.vehicle.lane_commands


def test_ego_actuation_disables_autonomous_lane_changes_but_keeps_safety_checks():
    c=connection()
    configure_ego_lane_actuation(c)
    assert EGO_LANE_CHANGE_MODE==512
    assert c.vehicle.lane_mode==('ego',512)


def test_instantaneous_events_use_center_geometry_and_surface_gap_ttc():
    ego=EgoState(0.,0.,10.,0.,0.,0.,4.,2.)
    overlap=WorldObject('vehicle:overlap',1.,0.,0.,0.,0.,4.,2.)
    lead=WorldObject('vehicle:lead',14.,0.,5.,0.,0.,4.,2.)
    lateral=WorldObject('vehicle:lateral',10.,5.,0.,0.,0.,4.,2.)
    result=instantaneous_events(ego,(overlap,lead,lateral))
    assert result['geometric_overlap_ids']==['vehicle:overlap']
    # overlap has zero surface gap and therefore the minimum TTC is zero.
    assert result['longitudinal_corridor_ttc_s']==0.
    result=instantaneous_events(ego,(lead,lateral))
    assert result['geometric_overlap_ids']==[]
    assert result['longitudinal_corridor_ttc_s']==2.
