"""Algebraic regret bound for a changing finite feasible-action set.

The caller, not this function, must justify *simultaneous* lower/upper cost
intervals at the evaluated state. In particular, data-fit intervals do not
become uniform certificates merely by passing this function's checks.
"""

from __future__ import annotations

import math
from typing import Hashable, Mapping


def interval_regret_upper(
    chosen: Hashable,
    feasible: set[Hashable] | frozenset[Hashable],
    cost_intervals: Mapping[Hashable, tuple[float, float]],
) -> float | None:
    """Return U_chosen - min_j L_j, or abstain when the data are incomplete.

    The set may differ from the reference action set. ``None`` means that no
    bound is asserted, including when the cached action is newly infeasible.
    """
    if not feasible or chosen not in feasible:
        return None
    if any(action not in cost_intervals for action in feasible):
        return None
    for action in feasible:
        lower, upper = cost_intervals[action]
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            return None
    return max(0.0, cost_intervals[chosen][1] - min(cost_intervals[action][0] for action in feasible))
