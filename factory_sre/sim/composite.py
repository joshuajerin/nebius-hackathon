"""Go2 + SO-101 composite authoring for the approved Factory SRE demo.

This module is deliberately separate from ``factory_sre.direct_usb``.  It
authors the mobile manipulator without changing the pinned source assets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Literal

from .circular_effector import author_circular_service_effector
from .config import SO101_ASSET, SO101_VERSION


GO2_ROOT = "/World/Go2"
SO101_ROOT = "/World/SO101"
PREFERRED_GO2_CHASSIS = "/World/Go2/Geometry/base"
PREFERRED_SO101_BASE = "/World/SO101/base"
PREFERRED_SO101_ROOT_JOINT = "/World/SO101/root_joint"


@dataclass(frozen=True, slots=True)
class CompositeReport:
    mode: str
    go2_chassis: str
    so101_base: str
    mount_joint: str
    mount_plate: str
    mount_pedestal: str
    mount_offset_m: tuple[float, float, float]
    mount_yaw_deg: float
    chassis_aabb_before_mount: tuple[float, float, float, float, float, float]
    mount_body0: tuple[str, ...]
    mount_body1: tuple[str, ...]
    end_effector: str
    end_effector_asset: str
    end_effector_kind: str
    magnetic_contact_frame: str
    magnetic_contact_radius_m: float
    hidden_gripper_geometry: tuple[str, ...]
    disabled_gripper_colliders: tuple[str, ...]
    wrist_camera: str
    articulation_roots: tuple[str, ...]
    go2_movable_joints: tuple[str, ...]
    so101_movable_joints: tuple[str, ...]
    payload_mass_kg: float
    gravity_compensated: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def measure_mount_stack(stage, report: CompositeReport) -> dict[str, object]:
    """Measure visible mount continuity instead of inferring it from a joint.

    The fixed joint proves kinematic attachment. These bounds independently
    prove that the rendered/collision geometry forms a continuous stack from
    chassis shell through pedestal and plate into the SO-101 base.
    """
    import numpy as np
    from isaacsim.core.experimental.utils import bounds as bounds_utils
    from pxr import UsdGeom, UsdPhysics

    def nearest_rigid_body(prim) -> str | None:
        candidate = prim
        while candidate and candidate.IsValid():
            if candidate.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(candidate.GetPath())
            candidate = candidate.GetParent()
        return None

    def owned_boundables(root_path: str, owner_path: str, excluded: tuple[str, ...]) -> list:
        return [
            prim
            for prim in _subtree(stage, root_path)
            if prim.IsA(UsdGeom.Boundable)
            and nearest_rigid_body(prim) == owner_path
            and not any(str(prim.GetPath()).startswith(prefix) for prefix in excluded)
        ]

    bbox_cache = bounds_utils.create_bbox_cache(use_extents_hint=False)
    chassis_geometry = owned_boundables(
        report.go2_chassis,
        report.go2_chassis,
        (report.mount_plate, report.mount_pedestal),
    )
    arm_base_geometry = owned_boundables(report.so101_base, report.so101_base, ())
    aabbs = {
        # Captured before the adapter geometry is authored, so the chassis
        # bound cannot accidentally prove contact against the adapter itself.
        "chassis": np.asarray(report.chassis_aabb_before_mount, dtype=float),
        "pedestal": bounds_utils.compute_aabb(report.mount_pedestal, bbox_cache=bbox_cache),
        "plate": bounds_utils.compute_aabb(report.mount_plate, bbox_cache=bbox_cache),
        # The pinned SO-101 base is reference-backed, so its composed bound is
        # authoritative even when the referenced meshes are not traversable
        # as ordinary child Boundable prims in this layer.
        "arm_base": (
            bounds_utils.compute_combined_aabb(arm_base_geometry, bbox_cache=bbox_cache)
            if arm_base_geometry
            else bounds_utils.compute_aabb(report.so101_base, bbox_cache=bbox_cache)
        ),
    }

    def interface(lower_name: str, upper_name: str) -> dict[str, float | str]:
        lower = aabbs[lower_name]
        upper = aabbs[upper_name]
        signed_vertical_overlap = float(lower[5] - upper[2])
        return {
            "lower": lower_name,
            "upper": upper_name,
            "gap_m": max(0.0, -signed_vertical_overlap),
            "vertical_overlap_m": max(0.0, signed_vertical_overlap),
            "x_overlap_m": max(0.0, float(min(lower[3], upper[3]) - max(lower[0], upper[0]))),
            "y_overlap_m": max(0.0, float(min(lower[4], upper[4]) - max(lower[1], upper[1]))),
        }

    return {
        "aabbs": {name: np.asarray(bounds, dtype=float).tolist() for name, bounds in aabbs.items()},
        "geometry_counts": {
            "chassis": len(chassis_geometry),
            "arm_base": len(arm_base_geometry),
        },
        "arm_base_bounds_source": "owned boundables" if arm_base_geometry else "composed base prim",
        "interfaces": {
            "chassis_to_pedestal": interface("chassis", "pedestal"),
            "pedestal_to_plate": interface("pedestal", "plate"),
            "plate_to_arm_base": interface("plate", "arm_base"),
        },
    }


def _subtree(stage, root_path: str):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"missing prim: {root_path}")
    return list(Usd.PrimRange(root))


def _first_rigid_body(stage, root_path: str, preferred: str):
    from pxr import UsdPhysics

    candidate = stage.GetPrimAtPath(preferred)
    if candidate.IsValid() and candidate.HasAPI(UsdPhysics.RigidBodyAPI):
        return candidate
    for prim in _subtree(stage, root_path):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    raise RuntimeError(f"no rigid body under {root_path}")


def _root_fixed_joint(stage):
    from pxr import UsdPhysics

    preferred = stage.GetPrimAtPath(PREFERRED_SO101_ROOT_JOINT)
    if preferred.IsValid() and preferred.IsA(UsdPhysics.FixedJoint):
        return UsdPhysics.FixedJoint(preferred)
    base_path = str(_first_rigid_body(stage, SO101_ROOT, PREFERRED_SO101_BASE).GetPath())
    for prim in _subtree(stage, SO101_ROOT):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        bodies = tuple(str(path) for path in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets())
        if base_path in bodies or not joint.GetBody0Rel().GetTargets():
            return UsdPhysics.FixedJoint(prim)
    raise RuntimeError("SO-101 has no fixed root joint to repurpose")


def _terminal_body(stage):
    """Return the final rigid body reached by the SO-101 joint graph."""
    from pxr import UsdPhysics

    rigid_paths = {
        str(prim.GetPath())
        for prim in _subtree(stage, SO101_ROOT)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    parent_paths: set[str] = set()
    child_paths: list[str] = []
    jaw_child: str | None = None
    for prim in _subtree(stage, SO101_ROOT):
        if not prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        parent_paths.update(path for path in body0 if path in rigid_paths)
        child_paths.extend(path for path in body1 if path in rigid_paths)
        if prim.GetName().lower() == "jaw" and body1:
            jaw_child = body1[0]
    if jaw_child and stage.GetPrimAtPath(jaw_child).IsValid():
        return stage.GetPrimAtPath(jaw_child)
    leaves = [path for path in child_paths if path not in parent_paths]
    if leaves:
        return stage.GetPrimAtPath(leaves[-1])
    if child_paths:
        return stage.GetPrimAtPath(child_paths[-1])
    return _first_rigid_body(stage, SO101_ROOT, PREFERRED_SO101_BASE)


def align_wrist_camera_to_chassis(
    stage,
    camera_path: str,
    end_effector_path: str,
    chassis_path: str,
) -> str:
    """Recalibrate the fixed jaw camera from the live parked-arm pose.

    The service tag is 0.28 m above the floor and the chassis is planted about
    0.80 m from its face.  Computing the optical axis from the real parked jaw
    position removes lateral parallax while chassis up keeps the image upright.
    """
    from math import sqrt

    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Get(stage, camera_path)
    end_effector = stage.GetPrimAtPath(end_effector_path)
    chassis = stage.GetPrimAtPath(chassis_path)
    if not camera or not end_effector.IsValid() or not chassis.IsValid():
        raise RuntimeError("wrist camera calibration frames are missing")
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    # The SO-101's parked jaw axis points downward, so attaching a camera with
    # a fixed jaw-local Euler rotation produces either a sideways gripper view
    # or a view of the floor.  Author the child transform from an explicit
    # chassis-relative optical frame instead: local -Z looks straight out of
    # the Go2 front and local +Y is world/chassis up.  The optical center stays
    # at the wrist with a small forward/up clearance from the jaw housing.
    cache = UsdGeom.XformCache()
    chassis_world = cache.GetLocalToWorldTransform(chassis)
    jaw_world = cache.GetLocalToWorldTransform(end_effector)
    jaw_in_chassis = jaw_world * chassis_world.GetInverse()
    jaw_position = jaw_in_chassis.ExtractTranslation()
    eye = (
        float(jaw_position[0]) + 0.08,
        float(jaw_position[1]),
        float(jaw_position[2]) + 0.04,
    )
    target = (0.80, 0.0, 0.28 - float(chassis_world.ExtractTranslation()[2]))
    raw_forward = tuple(target[index] - eye[index] for index in range(3))
    forward_length = sqrt(sum(value * value for value in raw_forward))
    if forward_length <= 1e-6:
        raise RuntimeError("parked wrist camera target is degenerate")
    forward = tuple(value / forward_length for value in raw_forward)
    raw_right = (forward[1], -forward[0], 0.0)
    right_length = sqrt(raw_right[0] * raw_right[0] + raw_right[1] * raw_right[1])
    right = tuple(value / right_length for value in raw_right)
    up = (
        right[1] * forward[2],
        -right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    camera_in_chassis = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        -forward[0], -forward[1], -forward[2], 0.0,
        eye[0], eye[1], eye[2],
        1.0,
    )
    camera_world = camera_in_chassis * chassis_world
    camera_local = camera_world * jaw_world.GetInverse()
    xform.MakeMatrixXform().Set(camera_local)
    return camera_path


def _author_wrist_camera(stage, end_effector, chassis) -> str:
    from pxr import Gf, UsdGeom

    camera_path = str(end_effector.GetPath()) + "/FactorySREWristCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(2.8)
    camera.CreateHorizontalApertureAttr(4.8)
    camera.CreateVerticalApertureAttr(3.6)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 10.0))
    return align_wrist_camera_to_chassis(
        stage,
        camera_path,
        str(end_effector.GetPath()),
        str(chassis.GetPath()),
    )


def aim_wrist_camera_at_world_point(
    stage,
    camera_path: str,
    end_effector_path: str,
    target_world: tuple[float, float, float],
) -> str:
    """Point the wrist inspection view at a manifest target with zero roll."""
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera.Get(stage, camera_path)
    end_effector = stage.GetPrimAtPath(end_effector_path)
    if not camera or not end_effector.IsValid():
        raise RuntimeError("wrist inspection frames are missing")
    jaw_world = UsdGeom.XformCache().GetLocalToWorldTransform(end_effector)
    eye_world = jaw_world.Transform(Gf.Vec3d(0.08, 0.0, 0.04))
    target = Gf.Vec3d(*target_world)
    if (target - eye_world).GetLength() <= 1e-6:
        raise RuntimeError("wrist inspection target is degenerate")
    # Gf.SetLookAt authors a world-to-camera view matrix.  USD cameras look
    # down local -Z, so invert it to obtain the camera-to-world transform.
    camera_world = Gf.Matrix4d().SetLookAt(
        eye_world,
        target,
        Gf.Vec3d(0.0, 0.0, 1.0),
    ).GetInverse()
    camera_local = camera_world * jaw_world.GetInverse()
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.MakeMatrixXform().Set(camera_local)
    return camera_path


def _set_opposite_joint_frame(stage, cache, joint_prim, *, body0base: bool) -> None:
    """Inline Isaac Sim 6.0.1 RobotAssembler joint-frame alignment math."""
    from pxr import Gf, UsdPhysics

    joint = UsdPhysics.Joint(joint_prim)
    body0_paths = joint.GetBody0Rel().GetTargets()
    body1_paths = joint.GetBody1Rel().GetTargets()
    body0 = stage.GetPrimAtPath(body0_paths[0]) if body0_paths else None
    body1 = stage.GetPrimAtPath(body1_paths[0]) if body1_paths else None
    if body0base:
        reference_body, reference_position, reference_rotation, opposite_body = (
            body0,
            joint.GetLocalPos0Attr().Get(),
            joint.GetLocalRot0Attr().Get(),
            body1,
        )
    else:
        reference_body, reference_position, reference_rotation, opposite_body = (
            body1,
            joint.GetLocalPos1Attr().Get(),
            joint.GetLocalRot1Attr().Get(),
            body0,
        )
    reference_world = cache.GetLocalToWorldTransform(reference_body) if reference_body else Gf.Matrix4d(1.0)
    opposite_world = cache.GetLocalToWorldTransform(opposite_body) if opposite_body else Gf.Matrix4d(1.0)
    reference_local = Gf.Transform()
    reference_local.SetRotation(Gf.Rotation(Gf.Quatd(reference_rotation)))
    reference_local.SetTranslation(Gf.Vec3d(reference_position))
    relative = reference_local * Gf.Transform(reference_world) * Gf.Transform(opposite_world.GetInverse())
    position = Gf.Vec3f(relative.GetTranslation())
    rotation = Gf.Quatf(relative.GetRotation().GetQuat())
    if body0base:
        joint.GetLocalPos1Attr().Set(position)
        joint.GetLocalRot1Attr().Set(rotation)
    else:
        joint.GetLocalPos0Attr().Set(position)
        joint.GetLocalRot0Attr().Set(rotation)


def configure_so101_stow_drives(
    stage,
    joint_paths: tuple[str, ...],
    *,
    stiffness: float = 120.0,
    damping: float = 12.0,
    maximum_force: float = 35.0,
) -> int:
    """Hold the arm's authored zero pose while the mobile base navigates.

    The arm remains articulated; manipulation code can replace these targets
    after the base plants.  This prevents inertial motion of an otherwise
    gravity-compensated arm from swinging the jaw-mounted camera in transit.
    """
    from pxr import UsdPhysics

    configured = 0
    for path in joint_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(maximum_force)
        configured += 1
    return configured


def author_go2_so101_composite(
    *,
    mode: Literal["merged", "controller_isolated"] = "merged",
    mount_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.125),
    mount_yaw_deg: float = 90.0,
    initial_chassis_position_m: tuple[float, float, float] | None = None,
    so101_payload_mass_kg: float = 1.2,
    gravity_compensated: bool = True,
) -> CompositeReport:
    """Load and attach the pinned SO-101 to an already-authored Go2.

    ``merged`` makes one reduced-coordinate articulation.  It is the strict
    structural gate. ``controller_isolated`` retains the arm articulation and
    excludes the physical mount joint from both controller trees; this is the
    fallback used only if the supplied 12-DOF locomotion policy rejects added
    arm DOFs.
    """
    import antioch
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.utils import bounds as bounds_utils
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    stage = antioch.stage()
    if not stage.GetPrimAtPath(GO2_ROOT).IsValid():
        raise RuntimeError("author the Go2 at /World/Go2 before mounting the SO-101")
    if not stage.GetPrimAtPath(SO101_ROOT).IsValid():
        local_so101 = os.environ.get("FACTORY_SRE_SO101_USD")
        if local_so101:
            local_path = Path(local_so101)
            if not local_path.is_file():
                raise RuntimeError(f"vendored SO-101 asset is missing: {local_path}")
            stage_utils.add_reference_to_stage(
                usd_path=str(local_path),
                path=SO101_ROOT,
            )
        else:
            antioch.load_asset(SO101_ASSET, prim_path=SO101_ROOT, version=SO101_VERSION)
    stage.Load()
    for _ in range(3):
        app_utils.update_app()

    go2_chassis = _first_rigid_body(stage, GO2_ROOT, PREFERRED_GO2_CHASSIS)
    so101_base = _first_rigid_body(stage, SO101_ROOT, PREFERRED_SO101_BASE)
    arm_bodies = [
        prim for prim in _subtree(stage, SO101_ROOT) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if not arm_bodies or so101_payload_mass_kg <= 0.0:
        raise ValueError("SO-101 payload mass must be positive and have rigid links")
    per_link_mass = float(so101_payload_mass_kg) / len(arm_bodies)
    for body in arm_bodies:
        mass = UsdPhysics.MassAPI(body) if body.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(body)
        attribute = mass.GetMassAttr() or mass.CreateMassAttr()
        attribute.Set(per_link_mass)
        if gravity_compensated:
            physx_body = (
                PhysxSchema.PhysxRigidBodyAPI(body)
                if body.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
                else PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            )
            physx_body.GetDisableGravityAttr().Set(True)
    source_mount = _root_fixed_joint(stage)
    if mode == "controller_isolated":
        # Keep both controller trees intact and connect them with an external
        # constraint. The inherited root joint is a fixed-to-world joint; an
        # authored jointEnabled=false still gets created by PhysX and can drag
        # the payload toward its original world pose after the Go2 walks.
        # Deactivating this composed prim matches the existing bundle loader
        # and leaves the source asset itself untouched.
        source_mount.GetPrim().SetActive(False)
        mount = UsdPhysics.FixedJoint.Define(stage, "/World/FactorySRECompositeMount")
    else:
        mount = source_mount

    # The asset root and its rigid ``base`` frame are not coincident. Align
    # the actual rigid base to the roof target instead of merely translating
    # /World/SO101, which can leave the visible arm detached from its joint.
    cache = UsdGeom.XformCache()
    chassis_world = cache.GetLocalToWorldTransform(go2_chassis)
    if initial_chassis_position_m is not None:
        authored_position = chassis_world.ExtractTranslation()
        expected_position = Gf.Vec3d(*initial_chassis_position_m)
        if (authored_position - expected_position).GetLength() > 0.05:
            raise RuntimeError(
                "Go2 authored chassis position does not match composite start: "
                f"{tuple(authored_position)} vs {initial_chassis_position_m}"
            )
    arm_root = stage.GetPrimAtPath(SO101_ROOT)
    arm_root_world = cache.GetLocalToWorldTransform(arm_root)
    arm_body_world = cache.GetLocalToWorldTransform(so101_base)
    arm_body_in_root = arm_body_world * arm_root_world.GetInverse()
    # SO-101's native service face points across the Go2 roof. Rotate that
    # mounting frame a quarter-turn so the arm base faces the Go2 front.
    mount_rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(mount_yaw_deg))
    mount_frame = Gf.Transform()
    mount_frame.SetTranslation(Gf.Vec3d(*mount_offset_m))
    mount_frame.SetRotation(mount_rotation)
    desired_body_world = mount_frame.GetMatrix() * chassis_world
    desired_root_world = arm_body_in_root.GetInverse() * desired_body_world
    UsdGeom.Xformable(arm_root).MakeMatrixXform().Set(desired_root_world)

    # Snapshot the robot bound before any adapter geometry exists. This gives
    # the continuity validator a non-circular chassis surface measurement.
    chassis_aabb_before_mount = tuple(
        float(value)
        for value in bounds_utils.compute_aabb(go2_chassis, include_children=True)
    )

    # Visible roof plate: visual evidence for the attachment, while the fixed
    # joint below remains the physical load path.
    mount_plate_path = str(go2_chassis.GetPath()) + "/FactorySREArmMountPlate"
    mount_plate = UsdGeom.Cube.Define(stage, mount_plate_path)
    mount_plate.CreateSizeAttr(1.0)
    plate_xform = UsdGeom.Xformable(mount_plate)
    plate_xform.AddTranslateOp().Set(
        Gf.Vec3d(mount_offset_m[0], mount_offset_m[1], mount_offset_m[2] + 0.005)
    )
    # The SO-101 reference geometry begins 30.08 mm above its rigid-body
    # origin. A 50 mm adapter block spans pedestal top (z=0.105) through that
    # measured base surface (z≈0.155), eliminating the rendered air gap.
    plate_xform.AddScaleOp().Set(Gf.Vec3f(0.18, 0.14, 0.05))
    mount_plate.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.10, 0.12)])
    UsdPhysics.CollisionAPI.Apply(mount_plate.GetPrim())
    # The Go2 shell is curved, so a flat plate alone can still look airborne.
    # This rigid pedestal overlaps the chassis actor at its bottom and meets
    # the plate at its top, forming one continuous collision-bearing adapter.
    mount_pedestal_path = str(go2_chassis.GetPath()) + "/FactorySREArmMountPedestal"
    mount_pedestal = UsdGeom.Cube.Define(stage, mount_pedestal_path)
    mount_pedestal.CreateSizeAttr(1.0)
    pedestal_xform = UsdGeom.Xformable(mount_pedestal)
    pedestal_xform.AddTranslateOp().Set(Gf.Vec3d(mount_offset_m[0], mount_offset_m[1], 0.065))
    pedestal_xform.AddScaleOp().Set(Gf.Vec3f(0.10, 0.09, 0.08))
    mount_pedestal.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.10, 0.12)])
    UsdPhysics.CollisionAPI.Apply(mount_pedestal.GetPrim())
    mount.CreateBody0Rel().SetTargets([go2_chassis.GetPath()])
    mount.CreateBody1Rel().SetTargets([so101_base.GetPath()])
    mount.CreateLocalPos0Attr(Gf.Vec3f(*mount_offset_m))
    mount.CreateLocalRot0Attr(Gf.Quatf(mount_rotation.GetQuat()))
    mount.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    mount.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Isaac Sim's robot assembler applies both safeguards because otherwise a
    # newly constrained pair can explode from contact overlap or stale
    # JointStateAPI values.
    filtering = UsdPhysics.FilteredPairsAPI.Apply(go2_chassis)
    filtering.CreateFilteredPairsRel().AddTarget(Sdf.Path(SO101_ROOT))
    for root_path in (GO2_ROOT, SO101_ROOT):
        for prim in _subtree(stage, root_path):
            for property_name in ("state:angular:physics:position", "state:angular:physics:velocity"):
                if prim.HasProperty(property_name):
                    prim.GetProperty(property_name).Set(0.0)

    if mode == "controller_isolated":
        cache = UsdGeom.XformCache()
        _set_opposite_joint_frame(stage, cache, mount.GetPrim(), body0base=False)
        _set_opposite_joint_frame(stage, cache, mount.GetPrim(), body0base=True)

    if mode == "merged":
        mount.CreateExcludeFromArticulationAttr(False)
        if so101_base.HasAPI(UsdPhysics.ArticulationRootAPI):
            so101_base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        so101_root = stage.GetPrimAtPath(SO101_ROOT)
        if so101_root.HasAPI(UsdPhysics.ArticulationRootAPI):
            so101_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    else:
        mount.CreateExcludeFromArticulationAttr(True)

    end_effector = _terminal_body(stage)
    circular_effector = author_circular_service_effector(
        stage,
        end_effector,
        go2_chassis,
    )
    camera_path = _author_wrist_camera(stage, end_effector, go2_chassis)
    for _ in range(2):
        app_utils.update_app()

    roots = tuple(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    go2_joints = tuple(
        str(prim.GetPath())
        for prim in _subtree(stage, GO2_ROOT)
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    )
    arm_joints = tuple(
        str(prim.GetPath())
        for prim in _subtree(stage, SO101_ROOT)
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    )
    joint = UsdPhysics.Joint(mount.GetPrim())
    return CompositeReport(
        mode=mode,
        go2_chassis=str(go2_chassis.GetPath()),
        so101_base=str(so101_base.GetPath()),
        mount_joint=str(mount.GetPath()),
        mount_plate=mount_plate_path,
        mount_pedestal=mount_pedestal_path,
        mount_offset_m=tuple(float(value) for value in mount_offset_m),
        mount_yaw_deg=float(mount_yaw_deg),
        chassis_aabb_before_mount=chassis_aabb_before_mount,
        mount_body0=tuple(str(path) for path in joint.GetBody0Rel().GetTargets()),
        mount_body1=tuple(str(path) for path in joint.GetBody1Rel().GetTargets()),
        end_effector=str(end_effector.GetPath()),
        end_effector_asset=circular_effector.prim_path,
        end_effector_kind="magnetic_circular_service_puck",
        magnetic_contact_frame=circular_effector.contact_frame,
        magnetic_contact_radius_m=circular_effector.contact_radius_m,
        hidden_gripper_geometry=circular_effector.hidden_source_geometry,
        disabled_gripper_colliders=circular_effector.disabled_source_colliders,
        wrist_camera=camera_path,
        articulation_roots=roots,
        go2_movable_joints=go2_joints,
        so101_movable_joints=arm_joints,
        payload_mass_kg=float(so101_payload_mass_kg),
        gravity_compensated=gravity_compensated,
    )
