"""Rebuildable clustered and paired statistics for BCES-IoV."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def paired_bootstrap_mean_ci(
    differences: Sequence[float], *, confidence: float, replicates: int, seed: int
) -> dict:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired differences must be nonempty and one-dimensional")
    generator = np.random.default_rng(seed)
    sampled = values[generator.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(sampled, alpha, method="linear")),
        "upper": float(np.quantile(sampled, 1.0 - alpha, method="linear")),
        "confidence": confidence, "replicates": replicates, "n_pairs": len(values),
    }


def paired_permutation_test(differences: Sequence[float], *, replicates: int, seed: int) -> dict:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired differences must be nonempty and one-dimensional")
    observed = abs(float(values.mean()))
    if np.all(values == 0):
        return {"mean_difference": 0.0, "p_value": 1.0, "replicates": replicates, "exact_zero": True}
    generator = np.random.default_rng(seed)
    signs = generator.choice((-1.0, 1.0), size=(replicates, len(values)))
    permuted = np.abs((signs * values).mean(axis=1))
    return {
        "mean_difference": float(values.mean()),
        "p_value": float((1 + np.count_nonzero(permuted >= observed)) / (replicates + 1)),
        "replicates": replicates, "exact_zero": False,
    }


def standardized_paired_effect(differences: Sequence[float]) -> float | None:
    values = np.asarray(differences, dtype=np.float64)
    standard = values.std(ddof=1)
    if standard == 0:
        return 0.0 if values.mean() == 0 else None
    return float(values.mean() / standard)


def clustered_uar_difference_ci(
    accepted_first: Sequence[bool], accepted_second: Sequence[bool],
    invalid: Sequence[bool], clusters: Sequence[str], *,
    confidence: float, replicates: int, seed: int,
) -> dict:
    first = np.asarray(accepted_first, dtype=bool)
    second = np.asarray(accepted_second, dtype=bool)
    unsafe = np.asarray(invalid, dtype=bool)
    if first.shape != second.shape or first.shape != unsafe.shape or len(clusters) != len(first):
        raise ValueError("clustered decision arrays must align")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[str(cluster)].append(index)
    names = sorted(grouped)
    counts = np.asarray([
        [first[grouped[name]].sum(), (first[grouped[name]] & unsafe[grouped[name]]).sum(),
         second[grouped[name]].sum(), (second[grouped[name]] & unsafe[grouped[name]]).sum()]
        for name in names
    ], dtype=np.int64)

    def difference(values: np.ndarray) -> float:
        first_rate = values[1] / values[0] if values[0] else 1.0
        second_rate = values[3] / values[2] if values[2] else 1.0
        return float(first_rate - second_rate)

    observed = difference(counts.sum(axis=0))
    generator = np.random.default_rng(seed)
    sampled_indices = generator.integers(0, len(names), size=(replicates, len(names)))
    sampled_counts = counts[sampled_indices].sum(axis=1)
    samples = np.asarray([difference(value) for value in sampled_counts])
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate_first_minus_second": observed,
        "lower": float(np.quantile(samples, alpha, method="linear")),
        "upper": float(np.quantile(samples, 1.0 - alpha, method="linear")),
        "confidence": confidence, "replicates": replicates, "cluster_count": len(names),
        "first_accepted": int(first.sum()), "second_accepted": int(second.sum()),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(value)))
        adjusted[name] = running
    return dict(sorted(adjusted.items()))

