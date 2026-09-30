import importlib.util
from pathlib import Path
import pytest

from bces.evaluation.slot_confirmation import random_slot
from bces.utils.hashing import canonical_json_hash, sha256_file

ROOT=Path(__file__).resolve().parents[2]


def test_registered_assignments_and_source_tampering_fail_closed(tmp_path):
    spec=importlib.util.spec_from_file_location('slot_runner',ROOT/'scripts/23_run_slot_confirmation.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source=tmp_path/'source.py'
    source.write_text('registered content\n')
    plan=[(seed,random_slot(seed,150926,1)) for seed in range(600000,600005)]
    registration={'source_sha256':{'source.py':sha256_file(source)},'prerequisite_sha256':{},
        'slot_randomization_seed':150926,'plan':{'calibration':{'scenario_seed_start':600000,
            'scenarios':5,'phase_id':1,'assignment_sha256':canonical_json_hash(plan)}}}
    assert module.verify_registration(registration,'calibration',tmp_path)==plan
    registration['slot_randomization_seed']=150927
    with pytest.raises(RuntimeError,match='assignment'): module.verify_registration(registration,'calibration',tmp_path)
    registration['slot_randomization_seed']=150926
    source.write_text('changed content\n')
    with pytest.raises(RuntimeError,match='input changed'): module.verify_registration(registration,'calibration',tmp_path)
