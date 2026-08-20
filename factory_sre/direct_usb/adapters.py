"""Deterministic adapters for the direct-USB mission and local certification."""

from __future__ import annotations

from dataclasses import dataclass, field

from factory_sre.contracts import DockObservation, Pose2D, Transform3D


@dataclass(slots=True)
class DryLocomotion:
    """A controllable stand-in for Nav2 plus the Go2 velocity adapter."""

    navigation_ok: bool = True
    stabilization_ok: bool = True
    is_stable: bool = True
    last_goal: Pose2D | None = None
    stopped: bool = False

    def navigate(self, pose: Pose2D) -> bool:
        self.last_goal = pose
        return self.navigation_ok

    def stabilize(self) -> bool:
        return self.stabilization_ok

    def stable(self) -> bool:
        return self.is_stable

    def emergency_stop(self) -> None:
        self.stopped = True


@dataclass(slots=True)
class DryArm:
    """A guarded-insertion stand-in with injectable contact conditions."""

    asset_id: str
    standoff_ok: bool = True
    servo_ok: bool = True
    contact: bool = True
    depth_m: float = 0.006
    axial_force_n: float = 3.0
    retreat_ok: bool = True
    last_target: Transform3D | None = None
    stopped: bool = False
    retreated: bool = False

    def move_to_standoff(self) -> bool:
        return self.standoff_ok

    def visual_servo(self, pose: Transform3D) -> bool:
        self.last_target = pose
        return self.servo_ok

    def guarded_insert(self, pose: Transform3D) -> DockObservation:
        self.last_target = pose
        return DockObservation(
            asset_id=self.asset_id,
            estimated_pose=pose,
            covariance=(0.0,) * 36,
            contact=self.contact,
            depth_m=self.depth_m,
            axial_force_n=self.axial_force_n,
            link_epoch=0,
        )

    def retreat(self) -> bool:
        self.retreated = True
        return self.retreat_ok

    def emergency_stop(self) -> None:
        self.stopped = True


@dataclass(slots=True)
class DryCamera:
    """Returns calibrated AprilTag poses keyed by physical printed size."""

    detections: dict[float, tuple[tuple[int, Transform3D], ...]] = field(default_factory=dict)

    def detect_tags(self, family: str, size_m: float) -> tuple[tuple[int, Transform3D], ...]:
        if family != "tag36h11":
            return ()
        return self.detections.get(size_m, ())

