"""Eight service cuboids arranged as a 2x4 navigation landmark grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, hypot, pi

from factory_sre.contracts import Pose2D

from .factory_navigation import RoutePlan
from .navigation import GridMap, astar, inflate

UNIFORM_BOX_COLOR = (0.18, 0.27, 0.36)
UNIFORM_STATUS_COLOR = (0.03, 0.72, 0.18)


@dataclass(frozen=True, slots=True)
class TaggedServiceBox:
    box_number: int
    asset_id: str
    tag_id: int
    center: tuple[float, float, float]
    tag_center: tuple[float, float, float]
    approach_xy: tuple[float, float]
    approach_yaw: float
    face_sign_y: int
    offline: bool


def tagged_service_boxes(*, offline_box_number: int = 8) -> tuple[TaggedServiceBox, ...]:
    if offline_box_number not in range(1, 9):
        raise ValueError("offline_box_number must be between 1 and 8")
    boxes = []
    # Leave a 1.6 m clear central aisle between the 0.8 m-deep boxes.  The
    # previous 1.2 m aisle left the Go2 base only one occupancy cell from the
    # inflated safety boundary during normal gait sway.
    for row, y in enumerate((0.4, 2.8)):
        for column, x in enumerate((0.0, 1.6, 3.2, 4.8)):
            index = row * 4 + column
            box_number = index + 1
            face_sign_y = 1 if row == 0 else -1
            boxes.append(
                TaggedServiceBox(
                    box_number=box_number,
                    asset_id=f"service-box-{box_number}",
                    tag_id=30 + index,
                    center=(x, y, 0.45),
                    tag_center=(x, y + face_sign_y * 0.41, 0.28),
                    approach_xy=(x, 1.6),
                    approach_yaw=-pi / 2.0 if row == 0 else pi / 2.0,
                    face_sign_y=face_sign_y,
                    offline=box_number == offline_box_number,
                )
            )
    return tuple(boxes)


def service_box_for_tag(tag_id: int, *, offline_box_number: int = 8) -> TaggedServiceBox:
    for box in tagged_service_boxes(offline_box_number=offline_box_number):
        if box.tag_id == tag_id:
            return box
    raise KeyError(f"unknown service-box AprilTag: {tag_id}")


def service_box_for_number(box_number: int, *, offline_box_number: int = 8) -> TaggedServiceBox:
    for box in tagged_service_boxes(offline_box_number=offline_box_number):
        if box.box_number == box_number:
            return box
    raise KeyError(f"unknown service-box number: {box_number}")


def tagged_box_occupancy() -> GridMap:
    """Return the 10 cm map used for coarse travel between the eight boxes."""
    resolution = 0.10
    origin = (-1.0, -3.0)
    raw: set[tuple[int, int]] = set()
    for box in tagged_service_boxes():
        center_x = round((box.center[0] - origin[0]) / resolution)
        center_y = round((box.center[1] - origin[1]) / resolution)
        for dx in range(-5, 6):
            for dy in range(-4, 5):
                raw.add((center_x + dx, center_y + dy))
    return GridMap(70, 80, resolution, origin, inflate(raw, radius_cells=3))


def plan_to_service_box(
    tag_id: int,
    *,
    start: Pose2D = Pose2D(2.4, 1.6, 0.0),
    grid: GridMap | None = None,
) -> RoutePlan:
    """Plan using occupancy, while the tag remains the terminal identity."""
    box = service_box_for_tag(tag_id)
    occupancy = grid or tagged_box_occupancy()
    # Enter the terminal pose along the service face normal.  Asking the
    # quadruped policy to rotate 90 degrees in place under the arm payload is
    # both slow and inaccurate; a short, collision-free final leg naturally
    # aligns the learned locomotion policy with the tag before it stops.
    pre_approach_xy = (
        box.approach_xy[0],
        box.approach_xy[1] + 0.30 * box.face_sign_y,
    )
    points = astar(occupancy, (start.x, start.y), pre_approach_xy)
    terminal_points = astar(occupancy, pre_approach_xy, box.approach_xy)
    points.extend(terminal_points[1:])
    compact = [points[0]]
    previous_direction: tuple[int, int] | None = None
    for index in range(1, len(points)):
        dx = round(points[index][0] - points[index - 1][0], 6)
        dy = round(points[index][1] - points[index - 1][1], 6)
        direction = (0 if dx == 0 else (1 if dx > 0 else -1), 0 if dy == 0 else (1 if dy > 0 else -1))
        if previous_direction is not None and direction != previous_direction:
            compact.append(points[index - 1])
        previous_direction = direction
    if compact[-1] != points[-1]:
        compact.append(points[-1])
    waypoints = []
    for index, point in enumerate(compact):
        if index + 1 < len(compact):
            following = compact[index + 1]
            yaw = atan2(following[1] - point[1], following[0] - point[0])
        else:
            yaw = box.approach_yaw
        waypoints.append(Pose2D(point[0], point[1], yaw))
    goal = Pose2D(box.approach_xy[0], box.approach_xy[1], box.approach_yaw)
    length = sum(
        hypot(right.x - left.x, right.y - left.y)
        for left, right in zip(waypoints, waypoints[1:], strict=False)
    )
    return RoutePlan(box.asset_id, start, goal, tuple(waypoints), length)


_DIGIT_SEGMENTS = {
    1: "bc",
    2: "abdeg",
    3: "abcdg",
    4: "bcfg",
    5: "acdfg",
    6: "acdefg",
    7: "abc",
    8: "abcdefg",
}


def _number_label(stage, root: str, box: TaggedServiceBox) -> None:
    """Author an unmistakable seven-segment box number on the aisle face."""
    from .scene import _box

    face_y = box.center[1] + box.face_sign_y * 0.412
    depth = 0.012
    _box(
        stage,
        root + "/NumberBackplate",
        (box.center[0], face_y, 0.67),
        (0.24, depth, 0.28),
        (0.94, 0.94, 0.90),
        collision=False,
    )
    segments = {
        "a": ((0.0, 0.765), (0.13, depth, 0.025)),
        "b": ((0.065, 0.715), (0.025, depth, 0.10)),
        "c": ((0.065, 0.625), (0.025, depth, 0.10)),
        "d": ((0.0, 0.575), (0.13, depth, 0.025)),
        "e": ((-0.065, 0.625), (0.025, depth, 0.10)),
        "f": ((-0.065, 0.715), (0.025, depth, 0.10)),
        "g": ((0.0, 0.67), (0.13, depth, 0.025)),
    }
    for segment in _DIGIT_SEGMENTS[box.box_number]:
        (offset_x, z), size = segments[segment]
        _box(
            stage,
            root + f"/Number/Segment_{segment.upper()}",
            (box.center[0] + offset_x, face_y + box.face_sign_y * 0.008, z),
            size,
            (0.015, 0.02, 0.025),
            collision=False,
        )


def build_tagged_box_grid(*, incident_box_number: int = 8) -> dict[str, object]:
    import antioch
    from isaacsim.core.experimental.objects import GroundPlane
    from pxr import UsdGeom, UsdLux

    from .scene import _box, _tag

    stage = antioch.stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    GroundPlane("/World/Ground", templates=None)
    report = []
    boxes = tagged_service_boxes(offline_box_number=incident_box_number)
    for box in boxes:
        root = f"/World/TaggedBoxes/{box.asset_id.replace('-', '_')}"
        _box(stage, root + "/Cuboid", box.center, (1.0, 0.8, 0.9), UNIFORM_BOX_COLOR)
        tag_faces = {
            "north": ((box.center[0], box.center[1] + 0.41, 0.28), "y", 1),
            "south": ((box.center[0], box.center[1] - 0.41, 0.28), "y", -1),
            "east": ((box.center[0] + 0.51, box.center[1], 0.28), "x", 1),
            "west": ((box.center[0] - 0.51, box.center[1], 0.28), "x", -1),
        }
        for face_name, (center, axis, sign) in tag_faces.items():
            _tag(
                stage,
                root + f"/Tags/{face_name.title()}Tag{box.tag_id}",
                center,
                box.tag_id,
                face_axis=axis,
                face_sign=sign,
            )
        # A bright registration strip makes each service face readable from
        # the aisle without adding decorative factory clutter.
        _box(
            stage,
            root + "/TagBackplate",
            (
                box.tag_center[0],
                box.tag_center[1] - box.face_sign_y * 0.015,
                box.tag_center[2],
            ),
            (0.22, 0.025, 0.22),
            (0.92, 0.92, 0.88),
            collision=False,
        )
        _number_label(stage, root, box)
        _box(
            stage,
            root + "/StatusBeacon",
            (box.center[0] + 0.32, box.center[1], 0.94),
            (0.10, 0.10, 0.08),
            UNIFORM_STATUS_COLOR,
            collision=False,
        )
        item = asdict(box)
        item["status"] = "offline" if box.offline else "online"
        item["body_color_rgb"] = list(UNIFORM_BOX_COLOR)
        item["status_color_rgb"] = list(UNIFORM_STATUS_COLOR)
        item["tag_faces"] = {name: list(values[0]) for name, values in tag_faces.items()}
        report.append(item)
    UsdLux.DomeLight.Define(stage, "/World/BoxGridDome").CreateIntensityAttr(300.0)
    UsdLux.DistantLight.Define(stage, "/World/BoxGridKey").CreateIntensityAttr(850.0)
    return {
        "layout": "2x4-central-aisle",
        "robot_start": [2.4, 1.6, 0.50],
        "incident": {
            "asset_id": f"service-box-{incident_box_number}",
            "tag_id": 29 + incident_box_number,
            "status": "offline",
        },
        "tag_family": "tag36h11",
        "tag_size_m": 0.120,
        "tag_faces_per_box": 4,
        "boxes": report,
    }
