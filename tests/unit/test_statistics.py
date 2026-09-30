from __future__ import annotations

import pytest

from bces.evaluation.statistics import (
    clustered_uar_difference_ci,
    holm_adjust,
    paired_bootstrap_mean_ci,
    paired_permutation_test,
)


def test_paired_bootstrap_is_deterministic_and_contains_constant_effect() -> None:
    first = paired_bootstrap_mean_ci([2, 2, 2], confidence=.95, replicates=100, seed=7)
    second = paired_bootstrap_mean_ci([2, 2, 2], confidence=.95, replicates=100, seed=7)
    assert first == second
    assert first["lower"] == first["upper"] == 2


def test_zero_paired_difference_has_unit_permutation_p_value() -> None:
    assert paired_permutation_test([0, 0, 0], replicates=100, seed=1)["p_value"] == 1


def test_clustered_uar_difference_uses_cluster_resampling() -> None:
    result = clustered_uar_difference_ci(
        [True, True, False, False], [True, False, True, False],
        [False, True, False, True], ["a", "a", "b", "b"],
        confidence=.95, replicates=200, seed=2,
    )
    assert result["cluster_count"] == 2
    assert result["estimate_first_minus_second"] == pytest.approx(0.5)


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": .01, "b": .04, "c": .03})
    assert adjusted == {"a": pytest.approx(.03), "b": pytest.approx(.06), "c": pytest.approx(.06)}

