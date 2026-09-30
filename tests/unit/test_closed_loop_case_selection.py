import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/24_audit_controlled_closed_loop.py'
SPEC = importlib.util.spec_from_file_location('closed_loop_pilot_case_selection', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_cases_takes_fixed_first_n_per_explicit_family():
    available = {'b': [{'seed': 4}, {'seed': 5}, {'seed': 6}],
                 'a': [{'seed': 1}, {'seed': 2}, {'seed': 3}]}
    assert [row['seed'] for row in MODULE.select_cases(available, ['b', 'a'], 2, 2)] == [4, 5, 1, 2]
    assert [row['seed'] for row in MODULE.select_cases(available, None, 2, 1)] == [1, 4]


def test_select_cases_rejects_missing_duplicate_and_short_blocks():
    available = {'a': [{'seed': 1}], 'b': [{'seed': 2}]}
    with pytest.raises(ValueError, match='duplicates'):
        MODULE.select_cases(available, ['a', 'a'], 2, 1)
    with pytest.raises(ValueError, match='unknown'):
        MODULE.select_cases(available, ['c'], 1, 1)
    with pytest.raises(RuntimeError, match='insufficient'):
        MODULE.select_cases(available, ['a'], 1, 2)
    with pytest.raises(ValueError, match='positive'):
        MODULE.select_cases(available, ['a'], 1, 0)
