from bces.geometry.interval_regret import interval_regret_upper


def test_newly_feasible_competitor_is_included_in_bound():
    # Action 'new' did not have to exist at the reference state.
    assert interval_regret_upper('cached', {'cached', 'new'},
                                 {'cached': (7.0, 9.0), 'new': (3.0, 5.0)}) == 6.0


def test_missing_new_action_bound_or_infeasible_cached_action_abstains():
    assert interval_regret_upper('cached', {'cached', 'new'}, {'cached': (7.0, 9.0)}) is None
    assert interval_regret_upper('cached', {'new'}, {'new': (3.0, 5.0)}) is None


def test_invalid_intervals_and_empty_set_abstain():
    assert interval_regret_upper('a', set(), {'a': (1.0, 2.0)}) is None
    assert interval_regret_upper('a', {'a'}, {'a': (2.0, 1.0)}) is None
    assert interval_regret_upper('a', {'a'}, {'a': (0.0, float('inf'))}) is None


def test_bound_is_algebraically_sound_for_discontinuous_point_costs():
    intervals = {'cached': (2.0, 11.0), 'other': (0.0, 8.0)}
    upper = interval_regret_upper('cached', {'cached', 'other'}, intervals)
    # No continuity or Lipschitz assumption enters this pointwise inequality.
    for cached_cost in (2.0, 11.0):
        for other_cost in (0.0, 8.0):
            regret = cached_cost - min(cached_cost, other_cost)
            assert regret <= upper
