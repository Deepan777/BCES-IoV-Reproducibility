import pytest
from scipy.stats import binom
from bces.evaluation.confirmation_power import exact_risk_audit_power, exact_paired_superiority_power
from bces.evaluation.scenario_slot_risk import exact_binomial_upper


def test_exact_power_matches_direct_small_binomial_enumeration():
    n,c,p,target,alpha=9,.7,.1,.5,.2
    direct=0.
    for a in range(1,n+1):
        for k in range(a+1):
            if exact_binomial_upper(k,a,alpha=alpha)<=target:
                direct+=binom.pmf(a,n,c)*binom.pmf(k,a,p)
    assert exact_risk_audit_power(n,c,p,target=target,alpha=alpha,comparisons=1,minimum_accepted=1)==pytest.approx(direct)


def test_no_accepted_slots_have_no_power_and_large_plan_has_declared_power():
    assert exact_risk_audit_power(100,0,.01)==0
    assert exact_risk_audit_power(8000,.75,.04)>.95


def test_paired_power_is_controlled_under_null_and_increases_for_declared_effect():
    assert exact_paired_superiority_power(100,.02,.02)<=.025
    assert exact_paired_superiority_power(1200,.008,.032)>.9
    with pytest.raises(ValueError): exact_risk_audit_power(0,.8,.02)
