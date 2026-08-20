"""Greedy route planning across a changing set of workstation failures."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot

from factory_sre.contracts import Pose2D
from factory_sre.sim.factory_navigation import RoutePlan
from factory_sre.sim.navigation import GridMap, astar
from factory_sre.sim.tagged_box_grid import plan_to_service_box, service_box_for_number
from factory_sre.sim.tagged_box_grid import tagged_box_occupancy


REST_POSE = Pose2D(-0.55, 1.60, 0.0)
DOCK_STANDOFF_M = 0.36


@dataclass(frozen=True, slots=True)
class FailureRoute:
    asset_ids: tuple[str, ...]
    tag_ids: tuple[int, ...]
    legs: tuple[RoutePlan, ...]
    total_length_m: float


def docking_pose_for_asset(asset_id: str) -> Pose2D:
    """Return the close, tag-facing stance used only after coarse travel."""
    box = service_box_for_number(_box_number(asset_id))
    return Pose2D(
        box.tag_center[0],
        box.tag_center[1] + box.face_sign_y * DOCK_STANDOFF_M,
        box.approach_yaw,
    )


def _compress(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    compact = [points[0]]
    previous_direction: tuple[int, int] | None = None
    for index in range(1, len(points)):
        dx = round(points[index][0] - points[index - 1][0], 6)
        dy = round(points[index][1] - points[index - 1][1], 6)
        direction = (
            0 if dx == 0 else (1 if dx > 0 else -1),
            0 if dy == 0 else (1 if dy > 0 else -1),
        )
        if previous_direction is not None and direction != previous_direction:
            compact.append(points[index - 1])
        previous_direction = direction
    compact.append(points[-1])
    return compact


def plan_between(
    start: Pose2D,
    goal: Pose2D,
    *,
    asset_id: str,
    grid: GridMap | None = None,
) -> RoutePlan:
    """Plan any aisle leg, including final docking and return-to-rest legs."""
    occupancy = grid or tagged_box_occupancy()
    points = _compress(astar(occupancy, (start.x, start.y), (goal.x, goal.y)))
    # Preserve the exact metric endpoint instead of the occupancy cell center.
    points[-1] = (goal.x, goal.y)
    waypoints: list[Pose2D] = []
    for index, point in enumerate(points):
        if index + 1 < len(points):
            following = points[index + 1]
            yaw = atan2(following[1] - point[1], following[0] - point[0])
        else:
            yaw = goal.yaw
        waypoints.append(Pose2D(point[0], point[1], yaw))
    return RoutePlan(
        asset_id=asset_id,
        start=start,
        goal=goal,
        waypoints=tuple(waypoints),
        length_m=sum(
            hypot(right.x - left.x, right.y - left.y)
            for left, right in zip(waypoints, waypoints[1:], strict=False)
        ),
    )


def _box_number(asset_id: str) -> int:
    prefix = "service-box-"
    if not asset_id.startswith(prefix):
        raise ValueError(f"unsupported workstation identity: {asset_id}")
    try:
        number = int(asset_id.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"unsupported workstation identity: {asset_id}") from exc
    if number not in range(1, 9):
        raise ValueError(f"unsupported workstation identity: {asset_id}")
    return number


def plan_failure_route(
    failures: tuple[str, ...] | list[str],
    *,
    start: Pose2D = REST_POSE,
) -> FailureRoute:
    """Visit every failed station using deterministic nearest-next planning."""
    remaining = set(failures)
    if len(remaining) != len(failures):
        raise ValueError("failure list cannot contain duplicates")
    ordered_ids: list[str] = []
    tag_ids: list[int] = []
    legs: list[RoutePlan] = []
    current = start
    while remaining:
        candidates = []
        for asset_id in sorted(remaining, key=_box_number):
            box = service_box_for_number(_box_number(asset_id))
            leg = plan_to_service_box(box.tag_id, start=current)
            candidates.append((leg.length_m, box.box_number, asset_id, box.tag_id, leg))
        _, _, asset_id, tag_id, selected = min(candidates)
        ordered_ids.append(asset_id)
        tag_ids.append(tag_id)
        legs.append(selected)
        current = selected.goal
        remaining.remove(asset_id)
    return FailureRoute(
        asset_ids=tuple(ordered_ids),
        tag_ids=tuple(tag_ids),
        legs=tuple(legs),
        total_length_m=sum(leg.length_m for leg in legs),
    )
