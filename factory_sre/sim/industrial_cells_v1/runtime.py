"""Uniform, wide industrial pick-and-place cells for the simulation demo.

This scene is intentionally separate from the compact tagged-box navigation
grid.  It reuses the same service-box and AprilTag identities, but owns its
wide physical layout so the eight UR5e cells can be viewed and simulated as
independent factory workcells.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import Any

from .contracts import CellSnapshot, IndustrialCellConfig


UR5E_USD = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
BASIC_BLOCK_USD = "/Isaac/Props/Blocks/basic_block.usd"
GO2_POLICY = "/Isaac/Samples/Policies/go2/physx_policy.pt"
GO2_POLICY_ENV = "/Isaac/Samples/Policies/go2/physx_env.yaml"
ROOT_PATH = "/World/IndustrialCellsV1"
COLUMN_SPACING_M = 2.6
ROW_SPACING_M = 3.2
SERVICE_DOCK_RADIUS_M = 0.040
SERVICE_DOCK_HEIGHT_M = 0.025
GO2_START_POSITION_M = (3.9, -1.55, 0.50)

_BODY_COLOR = (0.10, 0.20, 0.28)
_BEACON_COLOR = (0.03, 0.72, 0.20)
_FIXTURE_COLOR = (0.16, 0.18, 0.20)
_BLOCK_COLOR = (0.96, 0.60, 0.08)

# These are deliberately compact presentation poses for a six-DOF UR5e. They
# are joint-space choreography, not a general IK or collision planner.
_POSES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("home", (0.0, -1.35, 1.55, -1.78, -1.57, 0.0)),
    ("pre_pick", (-0.34, -1.22, 1.72, -1.95, -1.57, 0.10)),
    ("pick", (-0.34, -1.52, 2.02, -2.08, -1.57, 0.10)),
    ("lift", (-0.34, -1.18, 1.70, -1.92, -1.57, 0.10)),
    ("bowl", (0.42, -1.18, 1.65, -1.88, -1.57, 0.34)),
    ("release", (0.42, -1.46, 1.95, -2.04, -1.57, 0.34)),
)


@dataclass(frozen=True, slots=True)
class _WideCellSpec:
    asset_id: str
    box_number: int
    tag_id: int
    center_m: tuple[float, float, float]
    face_sign_y: int

    @property
    def tag_center_m(self) -> tuple[float, float, float]:
        x, y, _ = self.center_m
        return (x, y + self.face_sign_y * 0.51, 0.32)

    @property
    def tag_faces(self) -> tuple[tuple[str, tuple[float, float, float], str, int], ...]:
        """Cardinal AprilTag poses, ordered north/south/east/west."""

        x, y, _ = self.center_m
        return (
            ("North", (x, y + 0.51, 0.32), "y", 1),
            ("South", (x, y - 0.51, 0.32), "y", -1),
            ("East", (x + 0.51, y, 0.32), "x", 1),
            ("West", (x - 0.51, y, 0.32), "x", -1),
        )


def _wide_cell_specs() -> tuple[_WideCellSpec, ...]:
    """Stable service identities in the industrial-only 2.6 m x 3.2 m grid."""

    specs: list[_WideCellSpec] = []
    for row, y in enumerate((0.0, ROW_SPACING_M)):
        for column, x in enumerate(tuple(COLUMN_SPACING_M * index for index in range(4))):
            index = row * 4 + column
            specs.append(
                _WideCellSpec(
                    asset_id=f"service-box-{index + 1}",
                    box_number=index + 1,
                    tag_id=30 + index,
                    center_m=(x, y, 0.45),
                    face_sign_y=1 if row == 0 else -1,
                )
            )
    return tuple(specs)


@dataclass(slots=True)
class _CellRuntime:
    asset_id: str
    box_number: int
    center_m: tuple[float, float, float]
    root_path: str
    ur5e_path: str
    articulation_root: str
    display_block_path: str
    service_dock_path: str
    tag_paths: tuple[str, ...]
    phase_offset_s: float
    robot: Any | None = None
    phase: str = "home"
    block_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _lerp(left: tuple[float, ...], right: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    return tuple(a + (b - a) * alpha for a, b in zip(left, right, strict=True))


def _smoothstep(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def _host_array(value: Any):
    """Convert Isaac's NumPy, Warp, or torch pose result to a host array."""

    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class IndustrialCellScene:
    """Live stage controller for eight uniformly active factory cells."""

    def __init__(
        self,
        config: IndustrialCellConfig,
        cells: list[_CellRuntime],
        warehouse_shell_path: str,
        lighting_paths: dict[str, tuple[str, ...]],
        composite_report: Any,
        go2_policy: Any,
        stow_drive_count: int,
    ) -> None:
        self.config = config
        self._cells = {cell.asset_id: cell for cell in cells}
        self.warehouse_shell_path = warehouse_shell_path
        self.lighting_paths = lighting_paths
        self.composite_report = composite_report
        self.go2_policy = go2_policy
        self.stow_drive_count = stow_drive_count
        self._go2_chassis = None
        self._so101_base = None
        self._so101_articulation = None
        self._body_command = None
        self._elapsed_s = 0.0

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(self._cells)

    def prepare_runtime(self) -> None:
        """Create wrappers before ``World.reset()`` binds PhysX tensor views."""

        from isaacsim.core.experimental.prims import Articulation, RigidPrim

        for cell in self._cells.values():
            if cell.robot is None:
                cell.robot = Articulation(paths=cell.articulation_root)
        if self._go2_chassis is None:
            self._go2_chassis = RigidPrim(self.composite_report.go2_chassis)
        if self._so101_base is None:
            self._so101_base = RigidPrim(self.composite_report.so101_base)
        if self._so101_articulation is None:
            self._so101_articulation = Articulation(paths=self.composite_report.so101_base)

    def initialize_runtime(self) -> None:
        """Verify each referenced UR5e becomes a usable six-DOF articulation."""

        import torch

        self.prepare_runtime()
        for cell in self._cells.values():
            robot = cell.robot
            assert robot is not None
            if not robot.is_physics_tensor_entity_valid():
                raise RuntimeError(f"UR5e PhysX tensor view is unavailable: {cell.articulation_root}")
            if robot.num_dofs != 6:
                raise RuntimeError(f"UR5e must expose six DOFs, got {robot.num_dofs}: {cell.articulation_root}")
        if not self._go2_chassis.is_physics_tensor_entity_valid():
            raise RuntimeError("Go2 chassis requires a live PhysX tensor view")
        if not self._so101_base.is_physics_tensor_entity_valid():
            raise RuntimeError("SO-101 base requires a live PhysX tensor view")
        if not self._so101_articulation.is_physics_tensor_entity_valid():
            raise RuntimeError("SO-101 articulation requires a live PhysX tensor view")
        if self._so101_articulation.num_dofs != 6:
            raise RuntimeError(
                f"SO-101 must expose six DOFs, got {self._so101_articulation.num_dofs}"
            )
        self.go2_policy.initialize()
        self.go2_policy.post_reset()
        self._body_command = torch.zeros(3, dtype=torch.float32, device="cuda")

    def step(self, dt_s: float) -> None:
        """Advance all asynchronous scripted pick-and-place loops."""

        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        if self._body_command is None:
            raise RuntimeError("initialize_runtime() must be called before step()")
        self.go2_policy.forward(dt_s, self._parked_body_command())
        self.step_workcells(dt_s)

    def step_workcells(self, dt_s: float) -> None:
        """Advance the eight workcells without commanding the mobile base."""

        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative")
        self._elapsed_s += dt_s
        for cell in self._cells.values():
            local_time = (self._elapsed_s + cell.phase_offset_s) % self.config.loop_period_s
            segment_duration = self.config.loop_period_s / len(_POSES)
            segment_index = int(local_time / segment_duration) % len(_POSES)
            next_index = (segment_index + 1) % len(_POSES)
            alpha = _smoothstep((local_time % segment_duration) / segment_duration)
            phase_name, current_pose = _POSES[segment_index]
            _, next_pose = _POSES[next_index]
            cell.phase = phase_name
            self._set_joint_targets(cell, _lerp(current_pose, next_pose, alpha))
            self._set_block_position(cell, self._block_path_position(cell, segment_index, alpha))

    def _parked_body_command(self):
        """Use small policy commands to cancel payload-induced standing drift."""

        positions, orientations = self.go2_policy.robot.get_world_poses()
        position = _host_array(positions)[0]
        orientation = _host_array(orientations)[0]
        w, x, y, z = (float(value) for value in orientation)
        yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        dx = GO2_START_POSITION_M[0] - float(position[0])
        dy = GO2_START_POSITION_M[1] - float(position[1])
        if dx * dx + dy * dy < 0.0004:
            dx = 0.0
            dy = 0.0
        body_x = cos(yaw) * dx + sin(yaw) * dy
        body_y = -sin(yaw) * dx + cos(yaw) * dy
        yaw_error = _wrap_angle(-yaw)
        self._body_command[0] = max(-0.22, min(0.22, 0.9 * body_x))
        self._body_command[1] = max(-0.22, min(0.22, 0.9 * body_y))
        self._body_command[2] = max(-0.65, min(0.65, 2.0 * yaw_error))
        return self._body_command

    def snapshot(self) -> dict[str, object]:
        """Return JSON-safe state for scenario reports and future agents."""

        cells: list[CellSnapshot] = []
        for cell in self._cells.values():
            positions: tuple[float, ...] | None = None
            if cell.robot is not None:
                positions = tuple(float(value) for value in cell.robot.get_dof_positions().numpy()[0])
            cells.append(
                CellSnapshot(
                    asset_id=cell.asset_id,
                    box_number=cell.box_number,
                    phase=cell.phase,
                    block_position_m=cell.block_position_m,
                    articulation_root=cell.articulation_root,
                    service_dock_path=cell.service_dock_path,
                    tag_paths=cell.tag_paths,
                    joint_positions_rad=positions,
                )
            )
        mobile_manipulator = self.composite_report.to_dict()
        mobile_manipulator["stow_drive_count"] = self.stow_drive_count
        mobile_manipulator["start_position_m"] = list(GO2_START_POSITION_M)
        if self._go2_chassis is not None and self._so101_base is not None:
            chassis_positions, chassis_orientations = self._go2_chassis.get_world_poses()
            arm_positions, arm_orientations = self._so101_base.get_world_poses()
            mobile_manipulator["chassis_position_m"] = [
                float(value) for value in chassis_positions.numpy()[0]
            ]
            mobile_manipulator["chassis_orientation_wxyz"] = [
                float(value) for value in chassis_orientations.numpy()[0]
            ]
            mobile_manipulator["arm_base_position_m"] = [
                float(value) for value in arm_positions.numpy()[0]
            ]
            mobile_manipulator["arm_base_orientation_wxyz"] = [
                float(value) for value in arm_orientations.numpy()[0]
            ]
        if self._so101_articulation is not None:
            mobile_manipulator["arm_joint_positions_rad"] = [
                float(value)
                for value in self._so101_articulation.get_dof_positions().numpy()[0]
            ]
        return {
            "loop_period_s": self.config.loop_period_s,
            "column_spacing_m": COLUMN_SPACING_M,
            "row_spacing_m": ROW_SPACING_M,
            "warehouse_shell_path": self.warehouse_shell_path,
            "lighting_paths": {
                name: list(paths) for name, paths in self.lighting_paths.items()
            },
            "mobile_manipulator": mobile_manipulator,
            "cells": [
                {
                    "asset_id": cell.asset_id,
                    "box_number": cell.box_number,
                    "center_m": list(self._cells[cell.asset_id].center_m),
                    "phase": cell.phase,
                    "block_position_m": list(cell.block_position_m),
                    "articulation_root": cell.articulation_root,
                    "service_dock_path": cell.service_dock_path,
                    "tag_paths": list(cell.tag_paths),
                    "joint_positions_rad": None
                    if cell.joint_positions_rad is None
                    else list(cell.joint_positions_rad),
                }
                for cell in cells
            ],
        }

    def _set_joint_targets(self, cell: _CellRuntime, target: tuple[float, ...]) -> None:
        if cell.robot is None:
            return
        import numpy as np

        cell.robot.set_dof_position_targets(np.asarray(target, dtype=np.float32).reshape(1, -1))

    def _source_position(self, cell: _CellRuntime) -> tuple[float, float, float]:
        x, y, z = cell.center_m
        return (x - 0.18, y - 0.08, z + 0.60)

    def _bowl_position(self, cell: _CellRuntime) -> tuple[float, float, float]:
        x, y, z = cell.center_m
        return (x + 0.18, y + 0.12, z + 0.60)

    def _block_path_position(self, cell: _CellRuntime, segment_index: int, alpha: float) -> tuple[float, float, float]:
        source = self._source_position(cell)
        lifted_source = (source[0], source[1], source[2] + 0.30)
        bowl = self._bowl_position(cell)
        lifted_bowl = (bowl[0], bowl[1], bowl[2] + 0.30)
        path = (source, source, lifted_source, lifted_bowl, bowl, source)
        return _lerp(path[segment_index], path[(segment_index + 1) % len(path)], alpha)

    def _set_block_position(self, cell: _CellRuntime, position: tuple[float, float, float]) -> None:
        from pxr import Gf, UsdGeom

        operations = UsdGeom.Xformable(_stage_prim(cell.display_block_path)).GetOrderedXformOps()
        operations[0].Set(
            Gf.Vec3d(
                position[0] - cell.center_m[0],
                position[1] - cell.center_m[1],
                position[2] - 0.90,
            )
        )
        cell.block_position_m = position


def _stage_prim(path: str):
    import antioch

    prim = antioch.stage().GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"industrial cell prim is missing: {path}")
    return prim


def _material_path(name: str) -> str:
    return f"{ROOT_PATH}/Materials/{name}"


def _make_material(
    stage: Any,
    path: str,
    color: tuple[float, float, float],
    *,
    metallic: float = 0.0,
    emissive: bool = False,
):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.30 if metallic else 0.34)
    if emissive:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim: Any, material_path: str) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(prim.GetStage(), material_path))


def _xform(stage: Any, path: str, translation: tuple[float, float, float]):
    from pxr import Gf, UsdGeom

    prim = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*translation))
    return prim


def _box(
    stage: Any,
    path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    material_path: str,
    *,
    collision: bool = False,
    rotation_deg: tuple[float, float, float] | None = None,
):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    transform = UsdGeom.Xformable(cube)
    transform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotation_deg is not None:
        transform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_deg))
    transform.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_material(cube.GetPrim(), material_path)
    return cube


def _find_articulation_root(stage: Any, root_path: str) -> str:
    from pxr import Usd, UsdPhysics

    roots = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one UR5e articulation root below {root_path}, found {roots}")
    return roots[0]


def _disable_rigid_bodies(stage: Any, root_path: str) -> None:
    from pxr import Usd, UsdPhysics

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)


def _author_bowl(stage: Any, root: str) -> None:
    """Make a visibly open shallow collection bowl from native USD geometry."""

    _box(stage, f"{root}/Floor", (0.18, 0.12, 0.035), (0.25, 0.20, 0.025), _material_path("Bowl"))
    for index in range(8):
        angle = index * (2.0 * pi / 8.0)
        _box(
            stage,
            f"{root}/Wall{index}",
            (0.18 + 0.12 * cos(angle), 0.12 + 0.09 * sin(angle), 0.075),
            (0.045 if index % 2 == 0 else 0.025, 0.025 if index % 2 == 0 else 0.045, 0.07),
            _material_path("Bowl"),
        )


def _author_service_dock(stage: Any, spec: _WideCellSpec) -> str:
    """Place a neutral circular service dock directly above the aisle tag."""

    from pxr import Gf, UsdGeom

    path = f"{ROOT_PATH}/ServiceBoxes/{spec.asset_id.replace('-', '_')}/ServiceDock"
    tag_x, tag_y, tag_z = spec.tag_center_m
    dock = UsdGeom.Cylinder.Define(stage, path)
    dock.CreateRadiusAttr(SERVICE_DOCK_RADIUS_M)
    dock.CreateHeightAttr(SERVICE_DOCK_HEIGHT_M)
    transform = UsdGeom.Xformable(dock)
    transform.AddTranslateOp().Set(Gf.Vec3d(tag_x, tag_y + spec.face_sign_y * 0.020, tag_z + 0.16))
    transform.AddRotateXOp().Set(90.0)
    _bind_material(dock.GetPrim(), _material_path("ServiceDock"))
    center = UsdGeom.Cylinder.Define(stage, f"{path}/Center")
    center.CreateRadiusAttr(0.012)
    center.CreateHeightAttr(0.028)
    center_transform = UsdGeom.Xformable(center)
    center_transform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    _bind_material(center.GetPrim(), _material_path("ServiceDockCenter"))
    return path


def _author_cardinal_tags(stage: Any, box_root: str, spec: _WideCellSpec) -> tuple[str, ...]:
    """Add one outward-facing AprilTag and backing plate to each box face."""

    from factory_sre.sim.scene import _tag

    paths: list[str] = []
    for face_name, center, axis, sign in spec.tag_faces:
        if axis == "y":
            backplate_center = (center[0], center[1] - sign * 0.016, center[2])
            backplate_size = (0.22, 0.025, 0.22)
        else:
            backplate_center = (center[0] - sign * 0.016, center[1], center[2])
            backplate_size = (0.025, 0.22, 0.22)
        _box(
            stage,
            f"{box_root}/TagBackplates/{face_name}",
            backplate_center,
            backplate_size,
            _material_path("Fixture"),
        )
        path = f"{box_root}/Tags/{face_name}Tag{spec.tag_id}"
        _tag(stage, path, center, spec.tag_id, face_axis=axis, face_sign=sign)
        paths.append(path)
    return tuple(paths)


def _author_visual_gripper(stage: Any, ur5e_path: str) -> None:
    from pxr import Usd

    candidates = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(ur5e_path)) if prim.GetName() in {"tool0", "wrist_3_link"}]
    if not candidates:
        return
    tool = candidates[0]
    gripper = str(tool.GetPath()) + "/FactorySREVisualGripper"
    _xform(stage, gripper, (0.0, 0.0, 0.045))
    _box(stage, f"{gripper}/Palm", (0.0, 0.0, 0.025), (0.06, 0.08, 0.05), _material_path("Gripper"))
    _box(stage, f"{gripper}/LeftFinger", (0.0, 0.040, 0.085), (0.025, 0.018, 0.12), _material_path("Gripper"))
    _box(stage, f"{gripper}/RightFinger", (0.0, -0.040, 0.085), (0.025, 0.018, 0.12), _material_path("Gripper"))


def _author_warehouse_shell(stage: Any) -> str:
    """Build a bright high-bay warehouse without freestanding warehouse props."""

    shell = f"{ROOT_PATH}/WarehouseShell"
    _xform(stage, shell, (0.0, 0.0, 0.0))

    _box(
        stage,
        f"{shell}/Floor",
        (3.9, 1.6, -0.10),
        (14.0, 10.0, 0.20),
        _material_path("WarehouseFloor"),
        collision=True,
    )
    for name, position, size in (
        ("BackWall", (3.9, 6.25, 2.80), (14.0, 0.18, 5.60)),
        ("LeftWall", (-2.92, 1.6, 2.80), (0.18, 9.10, 5.60)),
        ("RightWall", (10.72, 1.6, 2.80), (0.18, 9.10, 5.60)),
        ("Roof", (3.9, 1.6, 5.68), (14.0, 9.10, 0.18)),
    ):
        material = "WarehouseRoof" if name == "Roof" else "WarehouseWall"
        _box(
            stage,
            f"{shell}/{name}",
            position,
            size,
            _material_path(material),
            collision=True,
        )

    # Visible wall-panel seams make the shell read as an insulated industrial
    # building rather than a featureless grey room.
    for index, x in enumerate((-1.8, -0.2, 1.4, 3.0, 4.6, 6.2, 7.8, 9.4)):
        _box(
            stage,
            f"{shell}/WallPanels/BackSeam{index}",
            (x, 6.14, 2.8),
            (0.025, 0.025, 5.35),
            _material_path("WarehousePanelSeam"),
        )
    for side_name, x in (("Left", -2.81), ("Right", 10.61)):
        for index, y in enumerate((-1.8, -0.2, 1.4, 3.0, 4.6)):
            _box(
                stage,
                f"{shell}/WallPanels/{side_name}Seam{index}",
                (x, y, 2.8),
                (0.025, 0.025, 5.35),
                _material_path("WarehousePanelSeam"),
            )

    # Perimeter steelwork and roof trusses. Only the corner posts reach the
    # open front, so the overview and navigation aisle remain unobstructed.
    for side, x in (("Left", -2.72), ("Right", 10.52)):
        for end, y in (("Front", -2.78), ("Rear", 6.05)):
            _box(
                stage,
                f"{shell}/SteelFrame/{side}{end}Column",
                (x, y, 2.75),
                (0.24, 0.24, 5.50),
                _material_path("WarehouseFrame"),
                collision=True,
            )
    _box(
        stage,
        f"{shell}/SteelFrame/FrontHeader",
        (3.9, -2.78, 5.24),
        (13.50, 0.26, 0.30),
        _material_path("WarehouseFrame"),
        collision=True,
    )
    for index, y in enumerate((-2.35, -0.35, 1.65, 3.65, 5.65)):
        _box(
            stage,
            f"{shell}/RoofBeams/CrossBeam{index}",
            (3.9, y, 5.18),
            (13.45, 0.14, 0.18),
            _material_path("WarehouseFrame"),
        )
        _box(
            stage,
            f"{shell}/RoofTrusses/LeftSlope{index}",
            (0.60, y, 4.88),
            (6.64, 0.09, 0.11),
            _material_path("WarehouseFrame"),
            rotation_deg=(0.0, -5.2, 0.0),
        )
        _box(
            stage,
            f"{shell}/RoofTrusses/RightSlope{index}",
            (7.20, y, 4.88),
            (6.64, 0.09, 0.11),
            _material_path("WarehouseFrame"),
            rotation_deg=(0.0, 5.2, 0.0),
        )

    # Clerestory glazing and roll-up doors are architectural wall details;
    # they add warehouse scale without introducing inventory or props.
    for index, x in enumerate((-0.2, 3.9, 8.0)):
        _box(
            stage,
            f"{shell}/Clerestory/Window{index}",
            (x, 6.13, 4.45),
            (2.85, 0.035, 0.62),
            _material_path("WarehouseGlass"),
        )
    for door_index, x in enumerate((-0.1, 3.9, 7.9)):
        door_root = f"{shell}/LoadingDoors/Door{door_index}"
        _box(
            stage,
            f"{door_root}/Panel",
            (x, 6.13, 1.55),
            (2.45, 0.04, 2.95),
            _material_path("WarehouseDoor"),
        )
        for rib_index, z in enumerate((0.35, 0.80, 1.25, 1.70, 2.15, 2.60)):
            _box(
                stage,
                f"{door_root}/Rib{rib_index}",
                (x, 6.10, z),
                (2.35, 0.035, 0.035),
                _material_path("WarehouseFrame"),
            )

    for index, x in enumerate((3.25, 4.55)):
        _box(
            stage,
            f"{shell}/AisleMarkings/Line{index}",
            (x, 1.55, 0.012),
            (0.055, 8.10, 0.012),
            _material_path("SafetyYellow"),
        )
    return shell


def _author_lighting(stage: Any) -> dict[str, tuple[str, ...]]:
    """Author visible high-bay fixtures plus four-direction interior fill."""

    from pxr import Gf, UsdGeom

    from pxr import UsdLux

    root = f"{ROOT_PATH}/Lighting"
    _xform(stage, root, (0.0, 0.0, 0.0))
    ceiling_paths: list[str] = []
    for row, y in enumerate((-1.05, 1.55, 4.15)):
        for column, x in enumerate((-0.65, 2.35, 5.45, 8.45)):
            fixture = f"{root}/Ceiling/Fixture{row}_{column}"
            _box(
                stage,
                f"{fixture}/Housing",
                (x, y, 5.25),
                (1.55, 0.48, 0.07),
                _material_path("LightHousing"),
            )
            _box(
                stage,
                f"{fixture}/Lens",
                (x, y, 5.20),
                (1.38, 0.34, 0.025),
                _material_path("LightLens"),
            )
            light_path = f"{fixture}/Light"
            light = UsdLux.RectLight.Define(stage, light_path)
            light.CreateWidthAttr(1.35)
            light.CreateHeightAttr(0.32)
            light.CreateIntensityAttr(5_500.0)
            light.CreateColorAttr(Gf.Vec3f(0.92, 0.96, 1.0))
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, y, 5.16))
            ceiling_paths.append(light_path)

    fill_paths: list[str] = []
    for name, position, rotation, housing_size, lens_offset in (
        ("Front", (3.9, -2.35, 2.8), (90.0, 0.0, 0.0), (1.30, 0.08, 0.42), (0.0, 0.05, 0.0)),
        ("Rear", (3.9, 5.95, 3.0), (-90.0, 0.0, 0.0), (1.30, 0.08, 0.42), (0.0, -0.05, 0.0)),
        ("Left", (-2.62, 1.6, 3.0), (0.0, -90.0, 0.0), (0.08, 1.30, 0.42), (0.05, 0.0, 0.0)),
        ("Right", (10.42, 1.6, 3.0), (0.0, 90.0, 0.0), (0.08, 1.30, 0.42), (-0.05, 0.0, 0.0)),
    ):
        fixture = f"{root}/Fill/{name}Fixture"
        _box(
            stage,
            f"{fixture}/Housing",
            position,
            housing_size,
            _material_path("LightHousing"),
        )
        lens_position = tuple(
            coordinate + offset
            for coordinate, offset in zip(position, lens_offset, strict=True)
        )
        lens_size = (
            1.10 if housing_size[0] > 0.1 else 0.025,
            1.10 if housing_size[1] > 0.1 else 0.025,
            0.26,
        )
        _box(
            stage,
            f"{fixture}/Lens",
            lens_position,
            lens_size,
            _material_path("LightLens"),
        )
        path = f"{fixture}/Light"
        fill = UsdLux.RectLight.Define(stage, path)
        fill.CreateWidthAttr(1.10)
        fill.CreateHeightAttr(0.26)
        fill.CreateIntensityAttr(3_200.0)
        fill.CreateColorAttr(Gf.Vec3f(0.78, 0.86, 1.0))
        fill_xform = UsdGeom.Xformable(fill)
        fill_xform.AddTranslateOp().Set(Gf.Vec3d(*lens_position))
        fill_xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        fill_paths.append(path)

    ambient_path = f"{root}/Ambient"
    ambient = UsdLux.DomeLight.Define(stage, ambient_path)
    ambient.CreateIntensityAttr(260.0)
    ambient.CreateColorAttr(Gf.Vec3f(0.72, 0.78, 0.86))
    return {
        "ceiling": tuple(ceiling_paths),
        "fill": tuple(fill_paths),
        "ambient": (ambient_path,),
    }


def _author_mobile_manipulator(assets_root: str) -> tuple[Any, Any, int]:
    """Build the policy-controlled Go2 with the pinned SO-101 roof payload."""

    import antioch
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy

    from factory_sre.sim.composite import (
        author_go2_so101_composite,
        configure_so101_stow_drives,
    )

    policy = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        position=list(GO2_START_POSITION_M),
        policy_path=assets_root + GO2_POLICY,
        env_config_path=assets_root + GO2_POLICY_ENV,
    )
    report = author_go2_so101_composite(
        mode="controller_isolated",
        initial_chassis_position_m=GO2_START_POSITION_M,
    )
    stow_drive_count = configure_so101_stow_drives(
        antioch.stage(), report.so101_movable_joints
    )
    if stow_drive_count != 6:
        raise RuntimeError(f"expected six SO-101 stow drives, configured {stow_drive_count}")
    return policy, report, stow_drive_count


def build_industrial_cell_grid(config: IndustrialCellConfig | None = None) -> IndustrialCellScene:
    """Compose eight active UR5e cells in their own wide industrial layout."""

    import antioch
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, UsdGeom

    config = config or IndustrialCellConfig()
    stage = antioch.stage()
    assets_root = get_assets_root_path()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, ROOT_PATH)
    UsdGeom.Xform.Define(stage, f"{ROOT_PATH}/Materials")
    for name, color, metallic in (
        ("ServiceBox", _BODY_COLOR, 0.0),
        ("Beacon", _BEACON_COLOR, 0.0),
        ("Fixture", _FIXTURE_COLOR, 0.0),
        ("Bowl", (0.18, 0.48, 0.66), 0.0),
        ("Tray", (0.23, 0.23, 0.25), 0.0),
        ("Block", _BLOCK_COLOR, 0.0),
        ("Gripper", (0.07, 0.07, 0.08), 0.0),
        ("ServiceDock", (0.44, 0.48, 0.52), 0.85),
        ("ServiceDockCenter", (0.11, 0.13, 0.15), 0.90),
        ("WarehouseFloor", (0.20, 0.23, 0.25), 0.0),
        ("WarehouseWall", (0.42, 0.45, 0.47), 0.0),
        ("WarehouseRoof", (0.24, 0.27, 0.30), 0.0),
        ("WarehouseFrame", (0.12, 0.14, 0.16), 0.85),
        ("WarehousePanelSeam", (0.24, 0.27, 0.30), 0.6),
        ("WarehouseGlass", (0.30, 0.55, 0.72), 0.15),
        ("WarehouseDoor", (0.32, 0.36, 0.39), 0.65),
        ("SafetyYellow", (0.96, 0.66, 0.04), 0.0),
        ("LightHousing", (0.16, 0.18, 0.20), 0.75),
    ):
        _make_material(stage, _material_path(name), color, metallic=metallic)
    _make_material(
        stage,
        _material_path("LightLens"),
        (0.92, 0.96, 1.0),
        emissive=True,
    )
    warehouse_shell_path = _author_warehouse_shell(stage)
    lighting_paths = _author_lighting(stage)

    cells: list[_CellRuntime] = []
    specs = _wide_cell_specs()
    for index, spec in enumerate(specs):
        box_root = f"{ROOT_PATH}/ServiceBoxes/{spec.asset_id.replace('-', '_')}"
        _xform(stage, box_root, (0.0, 0.0, 0.0))
        _box(stage, f"{box_root}/Cuboid", spec.center_m, (1.0, 0.8, 0.9), _material_path("ServiceBox"), collision=True)
        tag_paths = _author_cardinal_tags(stage, box_root, spec)
        _box(
            stage,
            f"{box_root}/StatusBeacon",
            (spec.center_m[0] + 0.32, spec.center_m[1], 0.94),
            (0.10, 0.10, 0.08),
            _material_path("Beacon"),
        )
        service_dock_path = _author_service_dock(stage, spec)

        cell_root = f"{ROOT_PATH}/Cells/{spec.asset_id.replace('-', '_')}"
        _xform(stage, cell_root, (spec.center_m[0], spec.center_m[1], 0.90))
        _box(stage, f"{cell_root}/Fixture", (0.0, 0.0, 0.035), (0.86, 0.66, 0.07), _material_path("Fixture"), collision=True)
        _box(stage, f"{cell_root}/SourceTray/Floor", (-0.18, -0.08, 0.075), (0.18, 0.15, 0.018), _material_path("Tray"))
        for tray_index, (x, y, sx, sy) in enumerate(((-0.26, -0.08, 0.018, 0.15), (-0.10, -0.08, 0.018, 0.15), (-0.18, -0.145, 0.18, 0.018))):
            _box(stage, f"{cell_root}/SourceTray/Wall{tray_index}", (x, y, 0.10), (sx, sy, 0.07), _material_path("Tray"))
        _author_bowl(stage, f"{cell_root}/CollectionBowl")
        source_root = _xform(stage, f"{cell_root}/SourceBlock", (-0.18, -0.08, 0.125))
        display_root = _xform(stage, f"{cell_root}/DisplayBlock", (-0.18, -0.08, 0.15))
        ur5e_path = f"{cell_root}/UR5e"
        stage_utils.add_reference_to_stage(usd_path=assets_root + UR5E_USD, path=ur5e_path)
        stage_utils.add_reference_to_stage(usd_path=assets_root + BASIC_BLOCK_USD, path=f"{source_root.GetPath()}/Asset")
        stage_utils.add_reference_to_stage(usd_path=assets_root + BASIC_BLOCK_USD, path=f"{display_root.GetPath()}/Asset")
        ur5e = stage.GetPrimAtPath(ur5e_path)
        UsdGeom.Xformable(ur5e).MakeMatrixXform().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.0, 0.0, 0.075)))
        cells.append(
            _CellRuntime(
                asset_id=spec.asset_id,
                box_number=spec.box_number,
                center_m=spec.center_m,
                root_path=cell_root,
                ur5e_path=ur5e_path,
                articulation_root="",
                display_block_path=str(display_root.GetPath()),
                service_dock_path=service_dock_path,
                tag_paths=tag_paths,
                phase_offset_s=(index / len(specs)) * config.loop_period_s,
                block_position_m=(spec.center_m[0] - 0.18, spec.center_m[1] - 0.08, 1.05),
            )
        )

    stage.Load()
    for _ in range(8):
        app_utils.update_app()
    for cell in cells:
        cell.articulation_root = _find_articulation_root(stage, cell.ur5e_path)
        _disable_rigid_bodies(stage, f"{cell.root_path}/SourceBlock")
        _disable_rigid_bodies(stage, cell.display_block_path)
        _bind_material(stage.GetPrimAtPath(f"{cell.root_path}/SourceBlock"), _material_path("Block"))
        _bind_material(stage.GetPrimAtPath(cell.display_block_path), _material_path("Block"))
        _author_visual_gripper(stage, cell.ur5e_path)

    go2_policy, composite_report, stow_drive_count = _author_mobile_manipulator(assets_root)
    scene = IndustrialCellScene(
        config,
        cells,
        warehouse_shell_path,
        lighting_paths,
        composite_report,
        go2_policy,
        stow_drive_count,
    )
    for cell in cells:
        scene._set_block_position(cell, scene._source_position(cell))
    return scene
