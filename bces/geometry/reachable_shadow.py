"""Development-only conservative footprint overapproximation during query gaps.

This is a conditional *geometric* inclusion, not a regret or driving-safety
certificate. The stated vector-norm error/acceleration bounds must hold for
the actor; no such bounds have yet been validated for deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from bces.oracle.world import WorldObject


@dataclass(frozen=True)
class ReachabilityAssumptions:
    position_error_m: float
    velocity_error_mps: float
    acceleration_bound_mps2: float

    def __post_init__(self) -> None:
        for name in ("position_error_m", "velocity_error_mps", "acceleration_bound_mps2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def center_error_radius_m(self, elapsed_s: float) -> float:
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        return (self.position_error_m + self.velocity_error_mps * elapsed_s
                + 0.5 * self.acceleration_bound_mps2 * elapsed_s * elapsed_s)


def reachable_shadow(
    message_object: WorldObject,
    *,
    age_s: float,
    horizon_s: float,
    assumptions: ReachabilityAssumptions,
) -> WorldObject:
    """Return a square containing every bounded actor footprint up to horizon.

    Assume the true reference center/velocity differ from message values by
    vector norms at most position_error_m/velocity_error_mps, and true
    acceleration has vector norm at most acceleration_bound_mps2. Assume
    the object's true footprint always fits within the message rectangle's
    half-diagonal about its center, allowing arbitrary heading changes.
    Then for each 0 <= s <= horizon_s, the true footprint lies within
    ``reachable_shadow(...).propagated(s)``. It is an overapproximation;
    the planner still needs a separate safe-action/continuous-time argument.
    """
    if not math.isfinite(age_s) or age_s < 0:
        raise ValueError("age_s must be finite and nonnegative")
    if not math.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("horizon_s must be finite and nonnegative")
    elapsed = age_s + horizon_s
    if not math.isfinite(elapsed):
        raise ValueError("age plus horizon overflowed")
    center_radius = assumptions.center_error_radius_m(elapsed)
    footprint_radius = 0.5 * math.hypot(message_object.length_m, message_object.width_m)
    half_side = center_radius + footprint_radius
    if not math.isfinite(half_side) or half_side <= 0:
        raise ValueError("reachable footprint is not finite and positive")
    predicted = message_object.propagated(age_s)
    return replace(predicted,
                   heading_rad=0.0,
                   length_m=2.0 * half_side,
                   width_m=2.0 * half_side,
                   source="development_reachable_shadow")
