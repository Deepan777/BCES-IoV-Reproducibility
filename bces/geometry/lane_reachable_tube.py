"""Conditional map-constrained actor-center reachability for development.

Unlike an isotropic ball, this assumes the actor remains near one of all
matched mapped lane continuations. It makes no claim that this premise or
the motion bounds hold in real traffic. No route-outcome label is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


Point = tuple[float, float]


def _distance_point_segment(x: float, y: float, first: Point, second: Point) -> float:
    dx, dy = second[0] - first[0], second[1] - first[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(x - first[0], y - first[1])
    fraction = max(0.0, min(1.0,
                            ((x - first[0]) * dx + (y - first[1]) * dy) / length_sq))
    return math.hypot(x - first[0] - fraction * dx,
                      y - first[1] - fraction * dy)


@dataclass(frozen=True)
class PolylinePath:
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("path needs at least two points")
        if any(not all(math.isfinite(value) for value in point) for point in self.points):
            raise ValueError("path coordinates must be finite")
        if self.length_m <= 0:
            raise ValueError("path length must be positive")

    @property
    def segment_lengths(self) -> tuple[float, ...]:
        return tuple(math.dist(a, b) for a, b in zip(self.points, self.points[1:]))

    @property
    def length_m(self) -> float:
        return sum(self.segment_lengths)

    def position_at(self, progress_m: float) -> Point:
        if not math.isfinite(progress_m) or not 0 <= progress_m <= self.length_m:
            raise ValueError("progress outside mapped path")
        remaining = progress_m
        for first, second, length in zip(self.points, self.points[1:], self.segment_lengths):
            if length == 0:
                continue
            if remaining <= length:
                fraction = remaining / length
                return (first[0] + fraction * (second[0] - first[0]),
                        first[1] + fraction * (second[1] - first[1]))
            remaining -= length
        return self.points[-1]

    def nearest_progress(self, point: Point) -> tuple[float, float]:
        best = (math.inf, 0.0)
        progress = 0.0
        for first, second, length in zip(self.points, self.points[1:], self.segment_lengths):
            if length == 0:
                continue
            dx, dy = second[0] - first[0], second[1] - first[1]
            fraction = max(0.0, min(1.0,
                                    ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy)
                                    / (length * length)))
            x = first[0] + fraction * dx
            y = first[1] + fraction * dy
            distance = math.hypot(point[0] - x, point[1] - y)
            if distance < best[0]:
                best = (distance, progress + fraction * length)
            progress += length
        return best

    def remaining_from(self, progress_m: float) -> PolylinePath:
        first = self.position_at(progress_m)
        retained = [first]
        progress = 0.0
        for point, length in zip(self.points[1:], self.segment_lengths):
            progress += length
            if progress > progress_m + 1e-9 and math.dist(retained[-1], point) > 1e-9:
                retained.append(point)
        if len(retained) < 2:
            raise ValueError("no mapped path remains after current position")
        return PolylinePath(tuple(retained))

    def min_distance_over_interval(self, point: Point, low_m: float, high_m: float) -> float:
        """Distance to mapped arc interval; beyond-map motion becomes a ball."""
        if (not math.isfinite(low_m) or not math.isfinite(high_m)
                or low_m < 0 or high_m < low_m):
            raise ValueError("invalid longitudinal reach interval")
        end = self.length_m
        best = math.inf
        if low_m <= end:
            low, high = low_m, min(high_m, end)
            if high == low:
                best = math.dist(point, self.position_at(low))
            else:
                progress = 0.0
                for first, second, length in zip(self.points, self.points[1:], self.segment_lengths):
                    start = max(low, progress)
                    stop = min(high, progress + length)
                    if length > 0 and start <= stop:
                        a = (first[0] + (start - progress) / length * (second[0] - first[0]),
                             first[1] + (start - progress) / length * (second[1] - first[1]))
                        b = (first[0] + (stop - progress) / length * (second[0] - first[0]),
                             first[1] + (stop - progress) / length * (second[1] - first[1]))
                        best = min(best, _distance_point_segment(*point, a, b))
                    progress += length
        if high_m > end:
            # After the represented path ends, no future route may be assumed.
            # A ball of the maximum residual travel conservatively encloses it.
            best = min(best, max(0.0, math.dist(point, self.points[-1]) - (high_m - end)))
        return best


def longitudinal_reach_m(
    *, speed_mps: float, elapsed_s: float, position_error_m: float,
    velocity_error_mps: float, acceleration_bound_mps2: float,
) -> tuple[float, float]:
    values = (speed_mps, elapsed_s, position_error_m,
              velocity_error_mps, acceleration_bound_mps2)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("reach inputs must be finite and nonnegative")
    nominal = speed_mps * elapsed_s
    uncertainty = (position_error_m + velocity_error_mps * elapsed_s
                   + 0.5 * acceleration_bound_mps2 * elapsed_s * elapsed_s)
    return max(0.0, nominal - uncertainty), nominal + uncertainty
