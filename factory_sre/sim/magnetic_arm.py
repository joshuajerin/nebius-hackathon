"""SO-101 magnetic puck manipulation using Isaac Sim's Robot Poser IK."""

from __future__ import annotations

from dataclasses import dataclass


def _quat_conjugate(quaternion):
    import numpy as np

    quaternion = np.asarray(quaternion, dtype=float)
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


def _quat_multiply(left, right):
    import numpy as np

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quat_rotate(quaternion, vector):
    import numpy as np

    quaternion = np.asarray(quaternion, dtype=float)
    vector = np.asarray(vector, dtype=float)
    vector_quaternion = np.concatenate(([0.0], vector))
    return _quat_multiply(
        _quat_multiply(quaternion, vector_quaternion),
        _quat_conjugate(quaternion),
    )[1:]


@dataclass(frozen=True, slots=True)
class MagneticIKResult:
    success: bool
    joints: dict[str, float]
    target_position_base_m: tuple[float, float, float]
    target_orientation_base_wxyz: tuple[float, float, float, float]


class MagneticArmController:
    """Drive the magnetic contact frame to a world-space AprilTag target."""

    def __init__(self, stage, composite_report) -> None:
        import numpy as np
        import omni.kit.app
        from isaacsim.core.experimental.prims import RigidPrim
        from pxr import UsdGeom
        from usd.schema.isaac.robot_schema import ApplyRobotAPI, ApplySiteAPI, Classes
        from usd.schema.isaac.robot_schema.utils import PopulateRobotSchemaFromArticulation

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        if not extension_manager.is_extension_enabled("isaacsim.robot.poser"):
            extension_manager.set_extension_enabled_immediate("isaacsim.robot.poser", True)
        from isaacsim.robot.poser import RobotPoser

        self.stage = stage
        self.report = composite_report
        self.robot_prim = stage.GetPrimAtPath(composite_report.so101_base)
        self.start_prim = self.robot_prim
        self.end_prim = stage.GetPrimAtPath(composite_report.magnetic_contact_frame)
        self.terminal_prim = stage.GetPrimAtPath(composite_report.end_effector)
        self.chassis_prim = stage.GetPrimAtPath(composite_report.go2_chassis)
        if not all(
            prim.IsValid()
            for prim in (self.robot_prim, self.end_prim, self.terminal_prim, self.chassis_prim)
        ):
            raise RuntimeError("magnetic arm controller requires composed base, jaw, chassis, and contact frame")
        if not self.end_prim.HasAPI(Classes.SITE_API.value):
            ApplySiteAPI(self.end_prim)
        if not self.robot_prim.HasAPI(Classes.ROBOT_API.value):
            ApplyRobotAPI(self.robot_prim)
        PopulateRobotSchemaFromArticulation(
            stage,
            self.robot_prim,
            self.robot_prim,
            detect_sites=True,
        )
        self.poser = RobotPoser(
            stage,
            self.robot_prim,
            self.start_prim,
            self.end_prim,
        )
        if self.poser.chain is None or not self.poser.chain.joints:
            raise RuntimeError("Robot Poser could not discover the SO-101 magnetic-tool chain")

        cache = UsdGeom.XformCache()
        terminal_world = cache.GetLocalToWorldTransform(self.terminal_prim)
        contact_world = cache.GetLocalToWorldTransform(self.end_prim)
        contact_in_terminal = contact_world * terminal_world.GetInverse()
        self._contact_offset_terminal = np.asarray(
            contact_in_terminal.ExtractTranslation(), dtype=float
        )
        contact_rotation = contact_in_terminal.ExtractRotation().GetQuat()
        imaginary = contact_rotation.GetImaginary()
        self._contact_orientation_terminal = np.asarray(
            [contact_rotation.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
            dtype=float,
        )
        # Declare live tensor views before the world reset binds PhysX actors.
        self.base_body = RigidPrim(composite_report.so101_base)
        self.terminal_body = RigidPrim(composite_report.end_effector)
        self.chassis_body = RigidPrim(composite_report.go2_chassis)
        self._solution: dict[str, float] | None = None

    def ready(self) -> bool:
        return all(
            body.is_physics_tensor_entity_valid()
            for body in (self.base_body, self.terminal_body, self.chassis_body)
        )

    @staticmethod
    def _pose(body):
        import numpy as np

        positions, orientations = body.get_world_poses()
        return (
            np.asarray(positions.numpy()[0], dtype=float),
            np.asarray(orientations.numpy()[0], dtype=float),
        )

    def solve(self, target_world_position_m) -> MagneticIKResult:
        import numpy as np
        from usd.schema.isaac.robot_schema.math import Transform

        if not self.ready():
            raise RuntimeError("magnetic arm tensor views are not initialized")
        base_position, base_orientation = self._pose(self.base_body)
        _, chassis_orientation = self._pose(self.chassis_body)
        base_inverse = _quat_conjugate(base_orientation)
        target_position_base = _quat_rotate(
            base_inverse,
            np.asarray(target_world_position_m, dtype=float) - base_position,
        )
        # The circular face's local +X normal stays aligned with the Go2's
        # forward axis, which already points at the selected tag after the
        # navigation terminal-yaw gate.
        target_orientation_base = _quat_multiply(base_inverse, chassis_orientation)
        result = self.poser.solve_ik(
            Transform(target_position_base, target_orientation_base),
            tolerance=0.004,
            iters=300,
            lam=0.01,
        )
        self._solution = dict(result.joints) if result.success else None
        return MagneticIKResult(
            success=bool(result.success),
            joints=dict(result.joints),
            target_position_base_m=tuple(float(value) for value in target_position_base),
            target_orientation_base_wxyz=tuple(
                float(value) for value in target_orientation_base
            ),
        )

    def apply_solution(self) -> None:
        if self._solution is None:
            raise RuntimeError("cannot apply a magnetic IK solution before a successful solve")
        self.poser.apply_pose(self._solution)

    def stow(self) -> None:
        self._solution = {path: 0.0 for path in self.report.so101_movable_joints}
        self.poser.apply_pose(self._solution)

    def contact_world_pose(self):
        terminal_position, terminal_orientation = self._pose(self.terminal_body)
        position = terminal_position + _quat_rotate(
            terminal_orientation,
            self._contact_offset_terminal,
        )
        orientation = _quat_multiply(
            terminal_orientation,
            self._contact_orientation_terminal,
        )
        return position, orientation
