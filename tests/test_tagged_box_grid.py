from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pupil_apriltags import Detector

from factory_sre.contracts import Pose2D
from factory_sre.sim.factory_navigation import validate_route
from factory_sre.sim.tagged_box_grid import (
    UNIFORM_BOX_COLOR,
    UNIFORM_STATUS_COLOR,
    plan_to_service_box,
    service_box_for_number,
    service_box_for_tag,
    tagged_box_occupancy,
    tagged_service_boxes,
)


def test_service_boxes_form_two_rows_of_four_with_unique_tags() -> None:
    boxes = tagged_service_boxes()
    assert len(boxes) == 8
    assert [box.box_number for box in boxes] == list(range(1, 9))
    assert {box.tag_id for box in boxes} == set(range(30, 38))
    assert sorted({box.center[0] for box in boxes}) == [0.0, 1.6, 3.2, 4.8]
    assert sorted({box.center[1] for box in boxes}) == [0.4, 2.8]
    assert {box.tag_center[2] for box in boxes} == {0.28}
    assert [box.box_number for box in boxes if box.offline] == [8]


def test_all_terminal_tags_face_the_central_aisle() -> None:
    for box in tagged_service_boxes():
        assert box.approach_xy == (box.center[0], 1.6)
        route = plan_to_service_box(box.tag_id)
        pre_approach = route.waypoints[-2]
        assert pre_approach.x == pytest.approx(box.approach_xy[0])
        assert pre_approach.y == pytest.approx(
            box.approach_xy[1] + 0.30 * box.face_sign_y
        )
        if box.box_number <= 4:
            assert box.tag_center[1] > box.center[1]
            assert box.approach_yaw == pytest.approx(-np.pi / 2.0)
        else:
            assert box.tag_center[1] < box.center[1]
            assert box.approach_yaw == pytest.approx(np.pi / 2.0)


def test_box_eight_offline_incident_selects_tag_37_without_redirect() -> None:
    box = service_box_for_number(8)
    route = plan_to_service_box(box.tag_id)
    assert box.asset_id == "service-box-8"
    assert box.tag_id == 37
    assert box.offline is True
    assert route.asset_id == box.asset_id
    assert route.goal == Pose2D(4.8, 1.6, np.pi / 2.0)
    assert box.tag_center[1] - route.goal.y == pytest.approx(0.79)


def test_box_two_can_be_the_only_offline_incident_without_colour_change() -> None:
    boxes = tagged_service_boxes(offline_box_number=2)
    target = service_box_for_number(2, offline_box_number=2)
    assert [box.box_number for box in boxes if box.offline] == [2]
    assert target.tag_id == 31
    assert target.asset_id == "service-box-2"
    assert UNIFORM_BOX_COLOR == (0.18, 0.27, 0.36)
    assert UNIFORM_STATUS_COLOR == (0.03, 0.72, 0.18)


@pytest.mark.parametrize("offline_box_number", [0, 9])
def test_offline_box_number_must_resolve_to_a_real_box(offline_box_number: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 8"):
        tagged_service_boxes(offline_box_number=offline_box_number)


@pytest.mark.parametrize("tag_id", range(30, 38))
def test_canonical_tag_texture_detects_as_its_declared_identity(tag_id: int) -> None:
    image_path = Path("assets/tags") / f"tag36h11_{tag_id}.png"
    image = np.asarray(Image.open(image_path).convert("L"))
    detections = Detector(families="tag36h11").detect(image)
    assert [int(result.tag_id) for result in detections] == [tag_id]


@pytest.mark.parametrize("tag_id", range(30, 38))
def test_every_tag_selects_a_collision_free_terminal_approach(tag_id: int) -> None:
    box = service_box_for_tag(tag_id)
    grid = tagged_box_occupancy()
    route = plan_to_service_box(tag_id, grid=grid)
    assert route.asset_id == box.asset_id
    assert route.final_error_m < 1e-9
    assert validate_route(route, grid) == ()


def test_unknown_tag_never_redirects_to_a_neighbor() -> None:
    with pytest.raises(KeyError, match="unknown service-box AprilTag"):
        plan_to_service_box(99)
