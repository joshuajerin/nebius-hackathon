"""Tag registration and guarded magnetic-dock controller contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, sqrt

from factory_sre.contracts import Transform3D, WorkbenchManifest


class WrongAssetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DockingLimits:
    maximum_position_error_m: float = 0.005
    maximum_angle_error_deg: float = 3.0
    maximum_axial_force_n: float = 8.0
    minimum_contact_depth_m: float = 0.005
    maximum_penetration_m: float = 0.010


@dataclass(frozen=True, slots=True)
class RegisteredDockTarget:
    asset_id: str
    tag_id: int
    world_tag: Transform3D
    world_dock: Transform3D
    position_error_m: float
    angle_error_deg: float


@dataclass(frozen=True, slots=True)
class MagneticDockResult:
    asset_id: str
    engaged: bool
    position_error_m: float
    angle_error_deg: float
    axial_force_n: float
    contact_depth_m: float
    penetration_m: float
    terminal_reason: str


def _distance(left: Transform3D, right: Transform3D) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left.translation, right.translation, strict=True)))


def _angle_degrees(left: Transform3D, right: Transform3D) -> float:
    lq = left.quaternion_wxyz
    rq = right.quaternion_wxyz
    ln = sqrt(sum(value * value for value in lq))
    rn = sqrt(sum(value * value for value in rq))
    if ln == 0.0 or rn == 0.0:
        raise ValueError("pose quaternion must be non-zero")
    dot = abs(sum(a * b for a, b in zip(lq, rq, strict=True)) / (ln * rn))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * acos(dot) * 180.0 / 3.141592653589793


def register_dock_target(
    manifest: WorkbenchManifest,
    *,
    observed_tag_id: int,
    world_tag: Transform3D,
    hidden_world_dock: Transform3D,
) -> RegisteredDockTarget:
    """Register the service dock without accepting a neighboring tag."""
    if observed_tag_id != manifest.tag_id:
        raise WrongAssetError(
            f"expected tag {manifest.tag_id} for {manifest.asset_id}, observed {observed_tag_id}"
        )
    estimate = world_tag.compose(manifest.tag_to_dock)
    return RegisteredDockTarget(
        asset_id=manifest.asset_id,
        tag_id=observed_tag_id,
        world_tag=world_tag,
        world_dock=estimate,
        position_error_m=_distance(estimate, hidden_world_dock),
        angle_error_deg=_angle_degrees(estimate, hidden_world_dock),
    )


def guarded_magnetic_dock(
    registered: RegisteredDockTarget,
    *,
    reached_pose: Transform3D,
    axial_force_n: float,
    contact_depth_m: float,
    penetration_m: float,
    limits: DockingLimits = DockingLimits(),
) -> MagneticDockResult:
    position_error = _distance(reached_pose, registered.world_dock)
    angle_error = _angle_degrees(reached_pose, registered.world_dock)
    reason = "ENGAGED"
    if axial_force_n > limits.maximum_axial_force_n:
        reason = "FORCE_LIMIT"
    elif penetration_m > limits.maximum_penetration_m:
        reason = "PENETRATION_LIMIT"
    elif contact_depth_m < limits.minimum_contact_depth_m:
        reason = "NO_CONTACT"
    elif position_error > limits.maximum_position_error_m:
        reason = "POSITION_ERROR"
    elif angle_error > limits.maximum_angle_error_deg:
        reason = "ANGLE_ERROR"
    return MagneticDockResult(
        asset_id=registered.asset_id,
        engaged=reason == "ENGAGED",
        position_error_m=position_error,
        angle_error_deg=angle_error,
        axial_force_n=axial_force_n,
        contact_depth_m=contact_depth_m,
        penetration_m=penetration_m,
        terminal_reason=reason,
    )
