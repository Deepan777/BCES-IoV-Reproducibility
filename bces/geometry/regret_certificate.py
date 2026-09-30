"""Conditional decision-regret certificate for a fixed finite action set.

This is deterministic mathematics, not evidence that an empirical envelope is
valid. A caller must independently justify every supplied uniform error bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np

from bces.geometry.codebooks import get_normal_codebook


def _vector(name: str, values: Sequence[float], *, nonnegative: bool = False) -> np.ndarray:
    result=np.asarray(values,dtype=float)
    if result.ndim!=1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a nonempty finite vector')
    if nonnegative and np.any(result<0):
        raise ValueError(f'{name} must be nonnegative')
    return result


def conditional_regret_bound(estimates, errors, selected: int) -> float:
    """Upper bound on J_selected - min_i J_i from simultaneous intervals.

    The selected action need not minimize the scalar estimates. This explicitly
    retains any nominal non-optimality induced by a lexicographic planner.
    """
    costs=_vector('estimates',estimates)
    radii=_vector('errors',errors,nonnegative=True)
    if costs.shape!=radii.shape or type(selected) is not int or not 0<=selected<len(costs):
        raise ValueError('aligned actions and a valid selected index required')
    comparisons=[costs[selected]-costs[j]+radii[selected]+radii[j]
                 for j in range(len(costs)) if j!=selected]
    return max(0.,max(comparisons,default=0.))


def linear_errors(intercepts, sensitivities, drift) -> np.ndarray:
    """Evaluate e_i(z)=u_i+L_i|z| with nonnegative coefficients."""
    base=_vector('intercepts',intercepts,nonnegative=True)
    matrix=np.asarray(sensitivities,dtype=float)
    z=_vector('drift',drift)
    if matrix.shape!=(len(base),len(z)) or not np.isfinite(matrix).all() or np.any(matrix<0):
        raise ValueError('sensitivities must be a finite nonnegative action-by-drift matrix')
    return base+matrix@np.abs(z)


@dataclass(frozen=True)
class BoxCertificate:
    origin_permitted: bool
    radii: tuple[float,...]
    offsets: tuple[float,...]
    tolerance: float
    limiting_slacks: tuple[float,...]
    assumptions: tuple[str,...]=(
        'fixed finite action set and selected action',
        'simultaneous uniform cost-error envelopes over the entire region',
        'same scalar cost defines the stated regret',
    )


def conservative_box_certificate(estimates, intercepts, sensitivities, selected: int,
                                 tolerance: float, maximum_radii, *, codebook_id: int = 1) -> BoxCertificate:
    """Construct a sufficient symmetric box and its receiver-checkable supports.

    For every competitor j, the box enforces
    (L_selected+L_j)|z| <= tolerance-(c_selected-c_j+u_selected+u_j).
    Axis normals in codebook v1 ensure the encoded polytope is a subset of this
    box; the two diagonal supports are redundant exact supports of the box.
    """
    costs=_vector('estimates',estimates)
    base=_vector('intercepts',intercepts,nonnegative=True)
    matrix=np.asarray(sensitivities,dtype=float)
    caps=_vector('maximum_radii',maximum_radii,nonnegative=True)
    if (costs.shape!=base.shape or matrix.shape!=(len(costs),len(caps))
            or not np.isfinite(matrix).all() or np.any(matrix<0)
            or type(selected) is not int or not 0<=selected<len(costs)
            or not math.isfinite(tolerance) or tolerance<0):
        raise ValueError('invalid certificate inputs')
    slacks=[];constraints=[]
    for j in range(len(costs)):
        if j==selected: continue
        slack=tolerance-(costs[selected]-costs[j]+base[selected]+base[j])
        slacks.append(float(slack));constraints.append(matrix[selected]+matrix[j])
    if any(value<0 for value in slacks):
        radii=np.zeros_like(caps);origin=False
    else:
        radii=caps.copy();origin=True
        for slack,weights in zip(slacks,constraints):
            active=np.flatnonzero(weights>0)
            if not len(active): continue
            allowances=slack/(len(active)*weights[active])
            radii[active]=np.minimum(radii[active],allowances)
    normals=get_normal_codebook(codebook_id)
    if len(caps)!=len(normals[0]): raise ValueError('drift dimension/codebook mismatch')
    offsets=tuple(float(np.abs(np.asarray(row))@radii) for row in normals)
    return BoxCertificate(origin,tuple(map(float,radii)),offsets,float(tolerance),tuple(slacks))


def verify_box_certificate(certificate: BoxCertificate, estimates, intercepts,
                           sensitivities, selected: int, *, atol: float = 1e-12) -> bool:
    """Algebraically verify every box corner through its weighted-L1 maximum."""
    if not certificate.origin_permitted: return not any(certificate.radii)
    costs=_vector('estimates',estimates);base=_vector('intercepts',intercepts,nonnegative=True)
    matrix=np.asarray(sensitivities,dtype=float);radii=np.asarray(certificate.radii)
    if matrix.shape!=(len(costs),len(radii)): return False
    for j in range(len(costs)):
        if j==selected: continue
        worst=costs[selected]-costs[j]+base[selected]+base[j]+(matrix[selected]+matrix[j])@radii
        if worst>certificate.tolerance+atol: return False
    return True
