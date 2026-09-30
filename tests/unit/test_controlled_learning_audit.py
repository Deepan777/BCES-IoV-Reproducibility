import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('controlled_learning_audit',
    Path(__file__).resolve().parents[2] / 'scripts/19_audit_controlled_learning.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_rank_audit_never_fills_coverage_with_rejected_points():
    assert audit.rank_subset([1, -.1, -.2, -.3], .5) is None
    assert audit.rank_subset([1, .5, -.1, -.2], .5).tolist() == [True, True, False, False]
    with pytest.raises(ValueError):
        audit.rank_subset([np.nan, 1], .5)


def test_paired_identical_rules_have_exact_zero_difference():
    y = np.array([True, False, True, True])
    accepted = np.ones(4, bool)
    result = audit.paired_difference(y, accepted, accepted, ['a', 'a', 'b', 'b'])
    assert result['surface_minus_ttl_uar'] == 0
    assert result['paired_scenario_bootstrap_interval_descriptive'] == [0, 0]
    assert result['scenario_count'] == 2


def test_paired_difference_direction_means_lower_unsafe_rate():
    result = audit.paired_difference(np.array([True, False] * 2),
        np.array([True, False] * 2), np.array([False, True] * 2), ['a', 'a', 'b', 'b'])
    assert result['surface_minus_ttl_uar'] == -1
    assert result['paired_scenario_bootstrap_interval_descriptive'] == [-1, -1]
