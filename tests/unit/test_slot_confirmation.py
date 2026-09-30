from types import SimpleNamespace
import pytest
from scipy.stats import binomtest

import bces.evaluation.slot_confirmation as audit
from bces.protocol.bound_exchange import encode_response
from tests.unit.test_bound_exchange import fixture


def test_slot_selection_reproducible_order_independent_and_separate():
    seeds = range(600000,600100)
    forward = {s:audit.random_slot(s,150926,1) for s in seeds}
    reverse = {s:audit.random_slot(s,150926,1) for s in reversed(seeds)}
    assert forward == reverse
    assert all(0 <= value < 315 for value in forward.values())
    assert forward != {s:audit.random_slot(s,150926,2) for s in seeds}
    with pytest.raises(ValueError): audit.random_slot(-1,150926,1)


@pytest.mark.parametrize('k,n',[(0,0),(0,100),(4,100),(100,100)])
def test_exact_upper_matches_independent_scipy_binomtest(k,n):
    expected = binomtest(k,n,alternative='less').proportion_ci(.975,method='exact').high if n else 1.
    assert audit.exact_upper(k,n,.025) == pytest.approx(expected,abs=1e-12)


def test_unavailable_oracle_never_suppresses_observable_acceptance(monkeypatch):
    bound,payload,extension = fixture()
    monkeypatch.setattr(audit,'bind_reference',lambda *a,**k:bound)
    monkeypatch.setattr(audit,'reference_query',lambda *a,**k:bound.query)
    monkeypatch.setattr(audit,'ego_from_sample',lambda x:x)
    monkeypatch.setattr(audit,'kinematic',lambda *a:bound.query.reference_state)
    model = SimpleNamespace(sha256=bound.model_sha256,view=bound.feature_view,
        issue=lambda wire,payload:encode_response(wire,extension))
    common = dict(scenario_id='1',slot=0,family='fixture',payload_hex=payload.hex(),
                  reference={},current_receiver={},age_s=0.)
    rule = {'surface':{'shrinkage':0.}}
    known = audit.evaluate_record({**common,'status':'evaluated','point':{'valid':True}}, {'surface':model},rule)
    unknown = audit.evaluate_record({**common,'status':'oracle_unavailable'}, {'surface':model},rule)
    assert known['methods']['surface']['accepted']
    assert not known['methods']['surface']['adverse']
    assert unknown['methods']['surface']['accepted']
    assert unknown['methods']['surface']['adverse']
    assert unknown['methods']['surface']['unknown_accepted']


def test_fixed_n_rejects_duplicates_and_reports_conservative_unknown_counts():
    def value(accepted,adverse,unknown=False):
        return dict(accepted=accepted,adverse=adverse,unknown_accepted=unknown)
    rows = [dict(scenario_id=str(i), methods={'surface':value(True,i==0,i==0),
                'scalar_ttl':value(True,i<10)}) for i in range(500)]
    result = audit.summarize(rows,expected_n=500)
    assert result['methods']['surface']['unknown_accepted']==1
    assert result['methods']['surface']['observed_invalid_accepted']==0
    assert result['paired_adverse_superiority']
    assert result['coverage_equivalence']
    assert not result['scientific_success'] and not result['manuscript_allowed']
    with pytest.raises(ValueError): audit.summarize(rows,expected_n=499)
    with pytest.raises(ValueError): audit.summarize(rows[:-1]+[rows[0]],expected_n=500)


def test_paired_difference_interval_symmetry_and_no_discordance():
    interval = audit.paired_difference_interval(10,25,1000)
    opposite = audit.paired_difference_interval(25,10,1000)
    assert interval == pytest.approx([-opposite[1],-opposite[0]])
    lower,upper = audit.paired_difference_interval(0,0,8000)
    assert lower < 0 < upper
    with pytest.raises(ValueError): audit.paired_difference_interval(3,3,5)
