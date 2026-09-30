import pytest
import yaml
from pathlib import Path
from bces.simulation.controlled_slots import decode_slot


def test_fixed_slot_layout_covers_all_choices_once():
    config=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/evaluation/study_b_controlled_v2.yaml').read_text())
    decoded=[decode_slot(i,config) for i in range(315)]
    assert len(set(decoded))==315
    assert decoded[0]==('keep',0,0.)
    assert decoded[62]==('keep',2,4.)
    assert decoded[-1]==('right',2,4.)
    with pytest.raises(ValueError): decode_slot(315,config)
    with pytest.raises(ValueError): decode_slot(-1,config)
