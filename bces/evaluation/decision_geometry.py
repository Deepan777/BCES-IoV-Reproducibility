"""Finite-candidate regret bounds and empirical fixed-normal geometry diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


def regret_upper_bound(costs, errors, selected: int) -> float:
    """Conditional bound; errors must bound ALL candidate costs in one world."""
    costs, errors = np.asarray(costs, float), np.asarray(errors, float)
    if costs.ndim != 1 or costs.shape != errors.shape or not len(costs):
        raise ValueError("matching nonempty cost/error vectors required")
    if not np.isfinite(costs).all() or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("finite costs and nonnegative errors required")
    if not 0 <= selected < len(costs):
        raise ValueError("selected action out of range")
    competitors = np.arange(len(costs)) != selected
    return float(max(0.0, np.max((costs[selected] - costs + errors[selected] + errors)[competitors], initial=0.0)))


def conservative_box(costs, base_errors, sensitivities, selected, tolerance, caps):
    """Maximize sum of normalized widths under supplied uniform error envelopes.

    This does not certify the supplied envelopes, and is not a risk/deviation bound.
    A missing/invalid envelope must not be replaced by empirical finite differences.
    """
    costs, base = np.asarray(costs, float), np.asarray(base_errors, float)
    slopes, caps = np.asarray(sensitivities, float), np.asarray(caps, float)
    regret_upper_bound(costs, base, selected)
    if slopes.shape != (len(costs), len(caps)) or not np.isfinite(slopes).all() or (slopes < 0).any():
        raise ValueError("nonnegative finite sensitivity matrix required")
    if not np.isfinite(caps).all() or (caps <= 0).any() or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("positive caps and nonnegative tolerance required")
    competitors = np.arange(len(costs)) != selected
    rhs = tolerance - costs[selected] + costs[competitors] - base[selected] - base[competitors]
    if (rhs < 0).any():
        return {"abstain": True, "widths": None, "reason": "reference_regret_bound_exceeds_tolerance"}
    if not competitors.any():
        return {"abstain": False, "widths": caps.tolist(), "conditional_on_envelopes": True}
    result = linprog(-1 / caps, A_ub=slopes[selected] + slopes[competitors], b_ub=rhs,
                     bounds=list(zip(np.zeros_like(caps), caps)), method="highs")
    if not result.success:
        return {"abstain": True, "widths": None, "reason": result.message}
    return {"abstain": False, "widths": result.x.tolist(), "conditional_on_envelopes": True}


def ray_radius(offsets, normals, direction):
    offsets, normals, direction = np.asarray(offsets), np.asarray(normals), np.asarray(direction)
    projections = normals @ direction
    positive = projections > 1e-12
    return float(np.min(offsets[positive] / projections[positive])) if positive.any() else float("inf")


def fit_zero_error_ttl(ages, valid):
    ages, valid = np.asarray(ages), np.asarray(valid, bool)
    best = np.zeros(len(ages), bool)
    threshold = None
    for value in np.unique(ages):
        accept = ages <= value
        if not (accept & ~valid).any() and accept.sum() > best.sum():
            best, threshold = accept, float(value)
    return {"accepted": best.tolist(), "threshold": threshold, "abstain": threshold is None}


def fit_zero_error_surface(drift, valid, normals, *, cap=65535/32768, time_limit=3.0):
    """MILP maximum sample coverage at zero sample error; never a predictive test.

    Uses 256 levels at a fixed, representable Q15 scale. The result is optimal
    within this fixed-scale family only when the solver reports optimality.
    """
    drift, valid, normals = np.asarray(drift, float), np.asarray(valid, bool), np.asarray(normals, float)
    if drift.ndim != 2 or len(drift) != len(valid) or normals.shape[1] != drift.shape[1]:
        raise ValueError("incompatible point/label/normal dimensions")
    if not np.isfinite(drift).all() or not np.isfinite(normals).all() or cap <= 0:
        raise ValueError("finite points and positive cap required")
    projected = drift @ normals.T
    # ceil determines the smallest accepting integer plane for each point.
    required = np.ceil(projected / (cap / 255) - 1e-10)
    good = np.flatnonzero(valid & (required <= 255).all(axis=1))
    bad = np.flatnonzero(~valid & (required <= 255).all(axis=1))
    if not len(good) or any((required[index] <= 0).all() for index in bad):
        return {"accepted": [False] * len(valid), "offsets": None, "abstain": True,
                "status": "origin_infeasible_or_no_representable_positive", "optimal": True}
    width = len(normals)
    witnesses = [(int(i), j) for i in bad for j in range(width) if required[i, j] > 0]
    count = width + len(good) + len(witnesses)
    objective = np.zeros(count)
    objective[width:width+len(good)] = -1
    lower, upper = np.zeros(count), np.ones(count)
    upper[:width] = 255
    rows, lows, highs = [], [], []
    for k, index in enumerate(good):
        for j in range(width):
            if required[index, j] <= 0:
                continue
            row = np.zeros(count)
            row[j], row[width+k] = 1, -required[index, j]
            rows.append(row); lows.append(0); highs.append(np.inf)
    for k, (index, j) in enumerate(witnesses):
        row = np.zeros(count)
        row[j] = 1
        row[width+len(good)+k] = 256 - required[index, j]
        rows.append(row); lows.append(-np.inf); highs.append(255)
    for index in bad:
        row = np.zeros(count)
        for k, (other, _) in enumerate(witnesses):
            if index == other:
                row[width+len(good)+k] = 1
        rows.append(row); lows.append(1); highs.append(np.inf)
    constraints = LinearConstraint(np.array(rows), lows, highs) if rows else None
    result = milp(objective, integrality=np.ones(count), bounds=Bounds(lower, upper),
                  constraints=constraints, options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if result.x is None:
        return {"accepted": [False]*len(valid), "offsets": None, "abstain": True,
                "status": result.message, "optimal": False}
    offsets = np.rint(result.x[:width]) * (cap / 255)
    accept = (projected <= offsets + 1e-10).all(axis=1)
    if (accept & ~valid).any():
        raise AssertionError("MILP witness failed independent membership check")
    return {"accepted": accept.tolist(), "offsets": offsets.tolist(), "abstain": False,
            "status": result.message, "optimal": result.status == 0,
            "mip_gap": float(result.mip_gap), "accepted_valid": int(accept.sum())}
