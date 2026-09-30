"""Independent-scenario random-slot audit for future registered confirmation.

Not a new statistical method, and not wired into the current development gate.
Assumes IID scenarios, a fixed candidate rule/family, and an independent uniform
slot plan fixed before labels. One audited decision per scenario avoids treating
correlated continuations as independent observations.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import beta


def random_slot_plan(scenario_ids,maximum_points,seed):
    ids=list(scenario_ids)
    if type(maximum_points) is not int or maximum_points<1 or len(ids)!=len(set(ids)) or not ids:
        raise ValueError('unique nonempty scenario IDs and positive slot count required')
    rng=np.random.default_rng(seed)
    return {key:int(rng.integers(maximum_points)) for key in sorted(ids)}


def exact_binomial_upper(unsafe,accepted,*,alpha=.05,comparisons=1):
    if (type(unsafe) is not int or type(accepted) is not int or not 0<=unsafe<=accepted
            or not 0<alpha<1 or type(comparisons) is not int or comparisons<1):
        raise ValueError('invalid binomial risk-bound inputs')
    if accepted==0 or unsafe==accepted:
        return 1.
    return float(beta.ppf(1-alpha/comparisons,unsafe+1,accepted-unsafe))


def zero_error_accepted_requirement(target,*,alpha=.05,comparisons=1):
    if not 0<target<1 or not 0<alpha<1 or type(comparisons) is not int or comparisons<1:
        raise ValueError('invalid sample-size inputs')
    return math.ceil(math.log(alpha/comparisons)/math.log1p(-target))


def audit_slot_risk(valid,accepted,scenarios,slots,plan,*,maximum_points,alpha=.05,comparisons=1):
    """Bound E[unsafe accepted count]/E[accepted count] over IID scenarios.

Each scenario has M predeclared slots, including unavailable/infeasible ones.
Uniformly sampling a slot makes P(U=1)=E[unsafe_count]/M and
P(A=1)=E[accepted_count]/M. Therefore P(U=1 | A=1) is the target ratio.
Given the accepted audit count, the unsafe count is binomial. Clopper-Pearson
and Bonferroni cover a fixed finite candidate family. Unavailable slots always
abstain. This guarantee does not apply to a plan selected using outcomes.
"""
    if not plan or type(maximum_points) is not int or maximum_points<1 or any(type(x) is not int or not 0<=x<maximum_points for x in plan.values()):
        raise ValueError('a valid preregistered slot plan is required')
    if not (len(valid)==len(accepted)==len(scenarios)==len(slots)):
        raise ValueError('unaligned decision arrays')
    lookup={}
    for y,a,scenario,slot in zip(valid,accepted,scenarios,slots):
        if scenario not in plan or type(slot) is not int or not 0<=slot<maximum_points:
            raise ValueError('decision outside registered scenario/slot domain')
        key=(scenario,slot)
        if key in lookup:
            raise ValueError('duplicate decision slot within a scenario')
        lookup[key]=(bool(y),bool(a))
    n=k=available=0
    for scenario,slot in plan.items():
        entry=lookup.get((scenario,slot))
        if entry is None:
            continue
        available+=1
        y,a=entry
        n+=int(a); k+=int(a and not y)
    return {'independent_scenarios':len(plan),'available_audit_slots':available,
            'accepted_audit_slots':n,'unsafe_accepted_audit_slots':k,
            'uar_upper':exact_binomial_upper(k,n,alpha=alpha,comparisons=comparisons),
            'confidence_familywise':1-alpha,'candidate_comparisons':comparisons,
            'estimand':'ratio_of_expected_unsafe_and_accepted_counts',
            'requires_iid_scenarios_and_label_independent_uniform_slot_plan':True,
            'physical_safety_certificate':False}
