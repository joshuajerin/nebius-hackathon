"""Manifest-selected coarse routes through the two-workbench factory aisle."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from factory_sre.contracts import Pose2D, WorkbenchManifest

from .navigation import GridMap, astar, hero_factory_grid


@dataclass(frozen=True, slots=True)
class RoutePlan:
    asset_id: str
    start: Pose2D
    goal: Pose2D
    waypoints: tuple[Pose2D, ...]
    length_m: float

    @property
    def final_error_m(self) -> float:
        final = self.waypoints[-1]
        return hypot(final.x - self.goal.x, final.y - self.goal.y)


def _compress(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Keep endpoints and grid-direction changes, not every 10 cm cell."""
    if len(points) <= 2:
        return points
    output = [points[0]]
    previous_direction: tuple[int, int] | None = None
    for index in range(1, len(points)):
        dx = round(points[index][0] - points[index - 1][0], 6)
        dy = round(points[index][1] - points[index - 1][1], 6)
        direction = (0 if dx == 0 else (1 if dx > 0 else -1), 0 if dy == 0 else (1 if dy > 0 else -1))
        if previous_direction is not None and direction != previous_direction:
            output.append(points[index - 1])
        previous_direction = direction
    output.append(points[-1])
    return output


def plan_to_manifest(
    manifest: WorkbenchManifest,
    *,
    start: Pose2D = Pose2D(-0.5, -2.5, 0.0),
    grid: GridMap | None = None,
) -> RoutePlan:
    occupancy = grid or hero_factory_grid()
    points = _compress(astar(occupancy, (start.x, start.y), (manifest.approach_pose.x, manifest.approach_pose.y)))
    waypoints: list[Pose2D] = []
    for index, point in enumerate(points):
        if index + 1 < len(points):
            nxt = points[index + 1]
            from math import atan2

            yaw = atan2(nxt[1] - point[1], nxt[0] - point[0])
        else:
            yaw = manifest.approach_pose.yaw
        waypoints.append(Pose2D(point[0], point[1], yaw))
    length = sum(
        hypot(right.x - left.x, right.y - left.y)
        for left, right in zip(waypoints, waypoints[1:], strict=False)
    )
    return RoutePlan(manifest.asset_id, start, manifest.approach_pose, tuple(waypoints), length)


def validate_route(plan: RoutePlan, grid: GridMap | None = None) -> tuple[str, ...]:
    occupancy = grid or hero_factory_grid()
    failures: list[str] = []
    if plan.asset_id == "":
        failures.append("route has no asset identity")
    if not plan.waypoints:
        failures.append("route has no waypoints")
        return tuple(failures)
    if plan.final_error_m > occupancy.resolution_m:
        failures.append(f"route misses approach pose by {plan.final_error_m:.3f} m")
    for waypoint in plan.waypoints:
        if not occupancy.free(occupancy.world_to_cell(waypoint.x, waypoint.y)):
            failures.append(f"occupied waypoint at ({waypoint.x:.2f}, {waypoint.y:.2f})")
    return tuple(failures)
