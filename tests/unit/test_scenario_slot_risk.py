import math
import pytest
import numpy as np
from scipy.stats import binom
from bces.evaluation.scenario_slot_risk import (random_slot_plan,exact_binomial_upper,
    zero_error_accepted_requirement,audit_slot_risk)


def test_zero_error_bound_is_positive_and_uses_multiplicity():
    assert exact_binomial_upper(0,80)==pytest.approx(1-.05**(1/80))
    assert exact_binomial_upper(0,80,comparisons=36)>.05
    assert exact_binomial_upper(0,0)==1
    assert exact_binomial_upper(4,4)==1
    assert zero_error_accepted_requirement(.05,comparisons=36)==129
    assert exact_binomial_upper(0,129,comparisons=36)<=.05
    assert exact_binomial_upper(0,128,comparisons=36)>.05


def test_auditing_many_correlated_rows_does_not_inflate_independent_count():
    result=audit_slot_risk([True]*6,[True]*6,['a']*3+['b']*3,[0,1,2]*2,
                          {'a':1,'b':2},maximum_points=3)
    assert result['accepted_audit_slots']==2
    assert result['uar_upper']==pytest.approx(exact_binomial_upper(0,2))


def test_unavailable_slots_abstain_instead_of_resampling_favorable_available_rows():
    result=audit_slot_risk([True],[True],['a'],[0],{'a':1,'b':2},maximum_points=3)
    assert result['independent_scenarios']==2
    assert result['accepted_audit_slots']==0
    assert result['uar_upper']==1


def test_plan_is_reproducible_and_does_not_take_labels():
    assert random_slot_plan(['b','a'],315,17)==random_slot_plan(['a','b'],315,17)
    with pytest.raises(ValueError): random_slot_plan(['a','a'],315,17)
    with pytest.raises(ValueError): random_slot_plan([],315,17)


def test_invalid_and_duplicate_slots_are_rejected():
    with pytest.raises(ValueError):
        audit_slot_risk([True]*2,[True]*2,['a']*2,[1,1],{'a':1},maximum_points=3)
    with pytest.raises(ValueError):
        audit_slot_risk([True],[True],['b'],[1],{'a':1},maximum_points=3)
    with pytest.raises(ValueError): exact_binomial_upper(3,2)
    with pytest.raises(ValueError): exact_binomial_upper(0,10,comparisons=0)


def test_exact_noncoverage_probability_is_bounded_without_monte_carlo():
    alpha=.05/36
    for n in (1,2,10,80,129):
        bounds=np.array([exact_binomial_upper(k,n,comparisons=36) for k in range(n+1)])
        for probability in np.linspace(0,1,101):
            failure_probability=binom.pmf(np.arange(n+1),n,probability)[bounds<probability].sum()
            assert failure_probability<=alpha+1e-12
