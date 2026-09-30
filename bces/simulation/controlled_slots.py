"""One preregistered decision slot per independent simulated scenario.

Uniform slots refer to the full 5-behavior x 3-continuation x 21-age domain.
Reference-infeasible or receiver-unavailable slots abstain. An unavailable
oracle label preserves the receiver check and counts adversely if accepted.
No slot is ever replaced by another slot.
"""
from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from bces.geometry.drift import DriftScales
from bces.oracle.observational import BEHAVIOR_IDS
from bces.oracle.planner import PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.simulation.controlled_contract import ego_from_sample, local_observations, make_message
from bces.simulation.controlled_decisions import ControlledDecisionPlanner, reference_contract, decision_pair

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def development_generator():
    spec = importlib.util.spec_from_file_location('controlled_reference_generator',ROOT/'scripts/16_generate_controlled_development.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_slot(slot, config):
    ages, variants, behaviors = config['ages_s'], config['receiver_drift_variants'], tuple(BEHAVIOR_IDS)
    maximum = len(ages)*len(variants)*len(behaviors)
    if type(slot) is not int or not 0 <= slot < maximum:
        raise ValueError('slot outside fixed domain')
    behavior, rest = divmod(slot,len(ages)*len(variants))
    variant, age = divmod(rest,len(ages))
    return behaviors[behavior],variant,ages[age]


def generate_slot(seed, slot, config, thresholds_dict, split):
    behavior,variant,age = decode_slot(slot,config)
    generator = development_generator()
    spec,parameters = generator.sample_scenario(seed,config)
    parameters['ego_post_reference_acceleration_mps2'] = config['receiver_drift_variants'][variant]
    planner = ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT/config['planner']))
    thresholds = ValidityThresholds(**thresholds_dict)
    # A shorter prefix+future duration changes no preceding controls or RNG use.
    trace_config = {**config,'ages_s':[age]}
    trace = generator.collect_trace(seed,spec,parameters,trace_config,planner.config.horizon_s)
    reference_time = round(config['reference_time_s']*1000)
    current_time = reference_time+round(age*1000)
    hidden = (('person:' if spec.pedestrian else 'vehicle:')+spec.actor_id,) if spec.hidden_from_local else ()
    result = {'scenario_id':str(seed),'slot':slot,'split':split,'family':spec.family,
        'behavior':behavior,'variant':variant,'age_s':age,'sampled_parameters':parameters,
        'trace':{str(t):{'receiver':f['receiver'],'objects':[asdict(o) for o in f['objects']]} for t,f in trace.items()}}
    def observed(timestamp):
        frame = trace[timestamp]
        if frame['receiver'] is None:
            raise ValueError('receiver_unavailable')
        ego = ego_from_sample(frame['receiver'])
        local = local_observations(frame['objects'],ego,range_m=config['local_range_m'],hidden_ids=hidden)
        cooperative = tuple(o for o in frame['objects'] if math.hypot(o.x_m-ego.x_m,o.y_m-ego.y_m)<=config['cooperative_range_m'])
        return ego,local,make_message(cooperative,timestamp,message_id=seed)
    try:
        ego,local,message = observed(reference_time)
        reference,query,cached = reference_contract(scenario=str(seed),split=split,behavior=behavior,
            timestamp=reference_time,ego=ego,local=local,message=message,planner=planner,
            scales=DriftScales(*config['drift_scales']))
    except RuntimeError as exc:
        if str(exc) != 'no feasible trajectory candidate': raise
        return {**result,'status':'abstain','reason':'planner_infeasible'}
    except ValueError as exc:
        if str(exc) != 'receiver_unavailable': raise
        return {**result,'status':'abstain','reason':'receiver_unavailable'}
    try:
        current_ego,current_local,fresh = observed(current_time)
    except ValueError as exc:
        if str(exc) != 'receiver_unavailable': raise
        return {**result,'status':'abstain','reason':'receiver_unavailable'}
    available = {**result,'reference':reference,'payload_hex':message.encode().hex(),
                 'current_receiver':trace[current_time]['receiver']}
    try:
        point = decision_pair(reference=reference,query=query,cached_reference=cached,ego=current_ego,
            local=current_local,fresh_message=fresh,future_world={t:f['objects'] for t,f in trace.items()},
            planner=planner,thresholds=thresholds)
    except RuntimeError as exc:
        if str(exc) != 'no feasible trajectory candidate': raise
        # Oracle failure is not observable when issuing the reference token.
        # Preserve the actual receiver check; accepted unknown labels count
        # against the primary risk bound, never as artificial abstentions.
        return {**available,'status':'oracle_unavailable','reason':'planner_infeasible'}
    return {**available,'status':'evaluated','point':point}
