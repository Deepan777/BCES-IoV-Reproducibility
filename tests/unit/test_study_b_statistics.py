import numpy as np
import pytest
from bces.evaluation.study_b import scenario_ratio_upper
from bces.evaluation.study_b import reference_features
from copy import deepcopy


def test_no_errors_in_small_cluster_sample_never_gives_zero_bound():
    assert scenario_ratio_upper([True]*5,[True]*5,list('abcde'),maximum_points=1)>0
    assert scenario_ratio_upper([True]*5,[False]*5,list('abcde'),maximum_points=1)==1


def test_more_independent_scenarios_tightens_bound():
    small = scenario_ratio_upper([True]*100,[True]*100,np.arange(100),maximum_points=1)
    large = scenario_ratio_upper([True]*10000,[True]*10000,np.arange(10000),maximum_points=1)
    assert 0<large<small<1


def test_repeated_ticks_do_not_add_independent_scenarios():
    one = scenario_ratio_upper([True]*10,[True]*10,np.arange(10),maximum_points=1)
    repeated = scenario_ratio_upper([True]*100,[True]*100,np.repeat(np.arange(10),10),maximum_points=10)
    assert one == pytest.approx(repeated)
    with pytest.raises(ValueError):
        scenario_ratio_upper([True]*100,[True]*100,np.repeat(np.arange(10),10),maximum_points=1)


def test_reference_features_ignore_outcomes_identifiers_and_absolute_time():
    batch={'reference_state':[10,20,5,0,0,0,1000],
           'object_features':[[0]*10 for _ in range(32)],'object_mask':[False]*32,
           'reference_path':[[10+i,20,5,0,0,i*.2] for i in range(12)],
           'drift_scales':[1]*7,'identifiers':[0,1,22,1,1,0],
           'object_count':0,'occlusion_proxy':0,'estimated_delay_s':0,'map_context_flags':0}
    reference={'frozen_input':{'batch':batch},'scenario_id':'first','valid':False}
    changed=deepcopy(reference)
    changed['scenario_id']='another'
    changed['valid']=True
    changed['fresh_cost']=999
    changed['frozen_input']['batch']['reference_state'][6]=999999
    changed['frozen_input']['batch']['identifiers'][2]=999
    assert np.array_equal(reference_features(reference),reference_features(changed))
def test_origin_validity_uses_all_futures_and_is_order_independent():
    from bces.evaluation.study_b import origin_validity
    points=[{'reference_id':'a','cache_age_s':0,'valid':True},
            {'reference_id':'a','cache_age_s':0,'valid':False},
            {'reference_id':'b','cache_age_s':0,'valid':True}]
    assert origin_validity(['a','b','c'],points).tolist()==[0,1,0]
    assert origin_validity(['a','b','c'],list(reversed(points))).tolist()==[0,1,0]
