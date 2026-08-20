from __future__ import annotations

import pytest

from factory_sre.live.planner import (
    REST_POSE,
    docking_pose_for_asset,
    plan_between,
    plan_failure_route,
)
from factory_sre.sim.tagged_box_grid import tagged_box_occupancy


def test_failure_planner_visits_every_station_once_and_is_deterministic() -> None:
    failures = ["service-box-7", "service-box-2", "service-box-5"]
    first = plan_failure_route(failures, start=REST_POSE)
    second = plan_failure_route(failures, start=REST_POSE)

    assert first == second
    assert set(first.asset_ids) == set(failures)
    assert len(first.asset_ids) == len(set(first.asset_ids)) == 3
    assert set(first.tag_ids) == {31, 34, 36}
    assert first.total_length_m == pytest.approx(sum(leg.length_m for leg in first.legs))
    assert all(leg.asset_id == asset_id for leg, asset_id in zip(first.legs, first.asset_ids, strict=True))


def test_failure_planner_handles_empty_queue() -> None:
    route = plan_failure_route([])
    assert route.asset_ids == ()
    assert route.legs == ()
    assert route.total_length_m == 0.0


@pytest.mark.parametrize("bad_id", ["station-2", "service-box-x", "service-box-9"])
def test_failure_planner_rejects_unknown_identity(bad_id: str) -> None:
    with pytest.raises(ValueError, match="unsupported workstation identity"):
        plan_failure_route([bad_id])


def test_failure_planner_rejects_duplicate_incidents() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        plan_failure_route(["service-box-2", "service-box-2"])


def test_docking_stances_face_each_row_from_the_central_aisle() -> None:
    lower = docking_pose_for_asset("service-box-1")
    upper = docking_pose_for_asset("service-box-5")

    assert (lower.x, lower.y) == pytest.approx((0.0, 1.17))
    assert lower.yaw == pytest.approx(-3.141592653589793 / 2.0)
    assert (upper.x, upper.y) == pytest.approx((0.0, 2.03))
    assert upper.yaw == pytest.approx(3.141592653589793 / 2.0)


def test_return_to_rest_is_collision_free_and_exact() -> None:
    occupancy = tagged_box_occupancy()
    start = docking_pose_for_asset("service-box-8")
    route = plan_between(start, REST_POSE, asset_id="rest", grid=occupancy)

    assert route.goal == REST_POSE
    assert route.waypoints[-1] == REST_POSE
    assert route.length_m > 0.0
    assert all(occupancy.free(occupancy.world_to_cell(pose.x, pose.y)) for pose in route.waypoints)
