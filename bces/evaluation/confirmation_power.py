"""Prospective exact power under explicitly supplied IID slot probabilities."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def lower_rejection_count(trials, null_probability, alpha):
    """Largest k with P_null(X <= k) <= alpha; -1 means never reject."""
    trials = np.asarray(trials)
    k = binom.ppf(alpha, trials, null_probability).astype(int)
    return k - (binom.cdf(k, trials, null_probability) > alpha)


def exact_risk_audit_power(scenarios, coverage, unsafe_probability, *, target=.05,
                           alpha=.05, comparisons=2, minimum_accepted=100):
    """Integrate binomial accepted count and conditional unsafe count.

    This is power at declared alternative probabilities, not a guarantee that
    unknown fresh data have those probabilities or will pass certification.
    """
    if (type(scenarios) is not int or scenarios < 1 or not 0 <= coverage <= 1
            or not 0 <= unsafe_probability <= 1 or not 0 < target < 1
            or not 0 < alpha < 1 or type(comparisons) is not int or comparisons < 1
            or type(minimum_accepted) is not int or minimum_accepted < 1):
        raise ValueError('invalid prospective power assumptions')
    accepted = np.arange(scenarios+1)
    cutoff = lower_rejection_count(accepted, target, alpha/comparisons)
    return float(np.sum(binom.pmf(accepted,scenarios,coverage)
        * binom.cdf(cutoff,accepted,unsafe_probability) * (accepted >= minimum_accepted)))


def exact_paired_superiority_power(scenarios, surface_only_unsafe, ttl_only_unsafe, *, alpha=.025):
    """Exact one-sided conditional paired-binary test, averaged over discordance."""
    discordance = surface_only_unsafe+ttl_only_unsafe
    if (type(scenarios) is not int or scenarios < 1 or not 0 < alpha < 1
            or surface_only_unsafe < 0 or ttl_only_unsafe < 0 or not 0 < discordance <= 1):
        raise ValueError('invalid paired power assumptions')
    d = np.arange(scenarios+1)
    cutoff = lower_rejection_count(d,.5,alpha)
    return float(np.sum(binom.pmf(d,scenarios,discordance)
        * binom.cdf(cutoff,d,surface_only_unsafe/discordance)))
