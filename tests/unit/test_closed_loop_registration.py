import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), ROOT/'scripts'/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_selection_is_balanced_outcome_blind_and_without_replacement():
    module = _script('26_register_closed_loop_confirmation.py')
    families = module.FAMILIES
    selected = module.select_seeds(lambda seed: families[seed % 4],
                                   first=100, scan_limit=30, n_per_family=3)
    assert len(selected) == 12
    assert len({row['seed'] for row in selected}) == 12
    assert [row['family'] for row in selected[:4]] == list(families)
    assert all(sum(row['family'] == family for row in selected) == 3 for family in families)
    with pytest.raises(RuntimeError, match='did not fill'):
        module.select_seeds(lambda seed: families[0], first=100, scan_limit=30, n_per_family=3)


def test_preserved_branch_has_identity_and_body_digest(tmp_path):
    module = _script('27_run_closed_loop_confirmation.py')
    path = tmp_path / module.artifact_name(900001, 'nominal', 'surface')
    body = {'scenario_id': '900001', 'family': module.FAMILIES[0],
            'method': 'surface', 'counts': {'reuse_decisions': 2}}
    arguments = {'digest': 'frozen', 'seed': 900001, 'family': module.FAMILIES[0],
                 'condition': 'nominal', 'method': 'surface'}
    module.save_branch(path, **arguments, branch=body)
    assert module.read_branch(path, **arguments) == body
    with pytest.raises(FileExistsError, match='cannot replace'):
        module.save_branch(path, **arguments, branch=body)
    with pytest.raises(RuntimeError, match='identity'):
        module.read_branch(path, **(arguments | {'seed': 900002}))
