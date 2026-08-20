"""Deployable Isaac Sim 6.0.1 asset composition for the Factory SRE MVP.

This module is intentionally asset-only. It composes stock Isaac assets by
reference, attaches Antioch's SO-101 to the policy-compatible Go2
articulation, and adds a plain calibrated-camera schema to the SO-101 wrist.
All simulator imports remain lazy for Antioch local discovery.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


STOCK_ASSETS = {
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "go2": "/Isaac/Samples/Policies/go2/go2.usda",
    "ur5e": "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "d455": "/Isaac/Sensors/RealSense/D455/rsd455.usd",
    "nanoscan3": "/Isaac/Sensors/SICK/nanoScan3/SICK_nanoScan3.usd",
    "inspector83x": "/Isaac/Sensors/SICK/Inspector83x/SICK_Inspector83x.usd",
}
GO2_POLICY = "/Isaac/Samples/Policies/go2/physx_policy.pt"
GO2_POLICY_ENV = "/Isaac/Samples/Policies/go2/physx_env.yaml"
SO101_NAME = "so101_antioch"
SO101_VERSION = "1.3.2"


@dataclass(frozen=True, slots=True)
class MountSpecification:
    """Physical mount measurements; replace prototype values from final CAD."""

    chassis_to_arm_base_m: tuple[float, float, float]
    plate_size_m: tuple[float, float, float]
    plate_mass_kg: float
    measured_from_hardware: bool


@dataclass(frozen=True, slots=True)
class WristCameraSpecification:
    """Pinhole camera parameters and its measured pose on the SO-101 wrist."""

    wrist_translation_m: tuple[float, float, float]
    wrist_rotation_xyz_deg: tuple[float, float, float]
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    clipping_range_m: tuple[float, float]
    calibrated_from_hardware: bool


# These dimensions are explicit prototype geometry, not a claim about the
# user's physical bracket. Validation reports remain fail-closed until they are
# replaced by measured CAD values.
PROTOTYPE_MOUNT = MountSpecification(
    chassis_to_arm_base_m=(0.0, 0.0, 0.205),
    plate_size_m=(0.160, 0.120, 0.012),
    plate_mass_kg=0.22,
    measured_from_hardware=False,
)
PROTOTYPE_WRIST_CAMERA = WristCameraSpecification(
    wrist_translation_m=(0.035, 0.0, 0.025),
    wrist_rotation_xyz_deg=(0.0, 90.0, 0.0),
    focal_length_mm=2.8,
    horizontal_aperture_mm=4.8,
    vertical_aperture_mm=3.6,
    clipping_range_m=(0.03, 10.0),
    calibrated_from_hardware=False,
)


def _reference(stage, assets_root: str, relative_path: str, prim_path: str):
    import isaacsim.core.experimental.utils.stage as stage_utils

    stage_utils.add_reference_to_stage(usd_path=assets_root + relative_path, path=prim_path)
    return stage.GetPrimAtPath(prim_path)


def _set_world_transform(stage, prim_path: str, matrix) -> None:
    from pxr import UsdGeom

    UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).MakeMatrixXform().Set(matrix)


def _author_mount_geometry(stage, mount: MountSpecification) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    path = "/World/FactorySRE/Go2/Geometry/base/FactorySREArmMount"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, mount.chassis_to_arm_base_m[2] / 2.0))
    xform.AddScaleOp().Set(Gf.Vec3f(*mount.plate_size_m))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateMassAttr(mount.plate_mass_kg)
    cube.GetPrim().CreateAttribute("factorySre:measuredFromHardware", Sdf.ValueTypeNames.Bool).Set(
        mount.measured_from_hardware
    )
    return path


def _attach_so101(stage, mount: MountSpecification) -> dict[str, object]:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    go2_body_path = "/World/FactorySRE/Go2/Geometry/base"
    arm_root_path = "/World/FactorySRE/SO101"
    arm_body_path = f"{arm_root_path}/base"
    old_root_joint_path = f"{arm_root_path}/root_joint"

    old_root_joint = stage.GetPrimAtPath(old_root_joint_path)
    if old_root_joint.IsValid():
        old_root_joint.SetActive(False)

    arm_body = stage.GetPrimAtPath(arm_body_path)
    if arm_body.HasAPI(UsdPhysics.ArticulationRootAPI):
        arm_body.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    cache = UsdGeom.XformCache()
    go2_body_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(go2_body_path))
    arm_root_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(arm_root_path))
    arm_body_world = cache.GetLocalToWorldTransform(arm_body)
    arm_body_in_root = arm_body_world * arm_root_world.GetInverse()
    mount_offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*mount.chassis_to_arm_base_m))
    desired_arm_body_world = mount_offset * go2_body_world
    desired_arm_root_world = arm_body_in_root.GetInverse() * desired_arm_body_world
    _set_world_transform(stage, arm_root_path, desired_arm_root_world)

    joint_path = "/World/FactorySRE/Go2SO101MountJoint"
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(go2_body_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(arm_body_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*mount.chassis_to_arm_base_m))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return {
        "go2_body": go2_body_path,
        "arm_body": arm_body_path,
        "fixed_joint": joint_path,
        "deactivated_arm_world_joint": old_root_joint_path,
    }


def _attach_plain_wrist_camera(stage, spec: WristCameraSpecification) -> dict[str, str]:
    from pxr import Gf, Sdf, UsdGeom

    mount_path = "/World/FactorySRE/SO101/wrist/FactorySRECameraMount"
    camera_path = f"{mount_path}/Camera"
    mount = UsdGeom.Xform.Define(stage, mount_path)
    transform = UsdGeom.Xformable(mount)
    transform.AddTranslateOp().Set(Gf.Vec3d(*spec.wrist_translation_m))
    transform.AddRotateXYZOp().Set(Gf.Vec3f(*spec.wrist_rotation_xyz_deg))
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(spec.focal_length_mm)
    camera.CreateHorizontalApertureAttr(spec.horizontal_aperture_mm)
    camera.CreateVerticalApertureAttr(spec.vertical_aperture_mm)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*spec.clipping_range_m))
    camera.GetPrim().CreateAttribute("factorySre:calibratedFromHardware", Sdf.ValueTypeNames.Bool).Set(
        spec.calibrated_from_hardware
    )
    return {"mount": mount_path, "camera": camera_path}


def _attach_go2_sensors(stage, assets_root: str) -> dict[str, str]:
    from pxr import Gf, UsdGeom

    body_path = "/World/FactorySRE/Go2/Geometry/base"
    placements = {
        "d455": ((0.245, 0.0, 0.055), (0.0, 90.0, 0.0)),
        "nanoscan3": ((0.0, 0.0, 0.115), (0.0, 0.0, 0.0)),
    }
    result: dict[str, str] = {}
    for name, (translation, rotation) in placements.items():
        path = f"{body_path}/FactorySRE_{name}"
        prim = _reference(stage, assets_root, STOCK_ASSETS[name], path)
        xform = UsdGeom.Xformable(prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        result[name] = path
    return result


def _author_semantic_frames(stage) -> dict[str, str]:
    from pxr import UsdGeom

    frames = {
        "base": "/World/FactorySRE/Go2/Geometry/base/FactorySREFrames/base",
        "arm_mount": "/World/FactorySRE/Go2/Geometry/base/FactorySREFrames/arm_mount",
        "wrist_camera": "/World/FactorySRE/SO101/wrist/FactorySRECameraMount",
        "usb_tip": "/World/FactorySRE/SO101/gripper/FactorySREFrames/usb_tip",
    }
    for path in frames.values():
        UsdGeom.Xform.Define(stage, path)
    return frames


def build_direct_usb_asset_bundle(
    mount: MountSpecification,
    wrist_camera: WristCameraSpecification,
) -> dict[str, object]:
    """Compose all reusable assets into the current live Isaac stage."""

    import antioch
    import isaacsim.core.experimental.utils.app as app_utils
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = antioch.stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    assets_root = get_assets_root_path()

    UsdGeom.Xform.Define(stage, "/World/FactorySRE")
    _reference(stage, assets_root, STOCK_ASSETS["warehouse"], "/World/FactorySRE/Warehouse")
    _reference(stage, assets_root, STOCK_ASSETS["go2"], "/World/FactorySRE/Go2")
    _reference(stage, assets_root, STOCK_ASSETS["ur5e"], "/World/FactorySRE/Workcell/UR5e")
    _reference(
        stage,
        assets_root,
        STOCK_ASSETS["inspector83x"],
        "/World/FactorySRE/Workcell/Inspector83x",
    )
    antioch.load_asset(SO101_NAME, prim_path="/World/FactorySRE/SO101", version=SO101_VERSION)
    stage.Load()
    for _ in range(5):
        app_utils.update_app()

    mount_geometry = _author_mount_geometry(stage, mount)
    attachment = _attach_so101(stage, mount)
    camera = _attach_plain_wrist_camera(stage, wrist_camera)
    sensors = _attach_go2_sensors(stage, assets_root)
    frames = _author_semantic_frames(stage)
    stage.Load()
    for _ in range(5):
        app_utils.update_app()

    go2_prims = list(Usd.PrimRange(stage.GetPrimAtPath("/World/FactorySRE/Go2")))
    so101_prims = list(Usd.PrimRange(stage.GetPrimAtPath("/World/FactorySRE/SO101")))
    composite_prims = go2_prims + so101_prims
    roots = [str(p.GetPath()) for p in composite_prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints = [
        str(p.GetPath())
        for p in composite_prims
        if p.IsA(UsdPhysics.Joint) and p.IsActive()
    ]
    mount_joint = stage.GetPrimAtPath(attachment["fixed_joint"])
    if mount_joint.IsValid() and mount_joint.IsActive():
        joints.append(str(mount_joint.GetPath()))
    workcell_prims = list(Usd.PrimRange(stage.GetPrimAtPath("/World/FactorySRE/Workcell/UR5e")))
    workcell_roots = [
        str(p.GetPath()) for p in workcell_prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    return {
        "assets_root": assets_root,
        "stock_assets": {name: assets_root + path for name, path in STOCK_ASSETS.items()},
        "go2_policy": assets_root + GO2_POLICY,
        "go2_policy_env": assets_root + GO2_POLICY_ENV,
        "so101": f"{SO101_NAME}@{SO101_VERSION}",
        "mount_specification": asdict(mount),
        "wrist_camera_specification": asdict(wrist_camera),
        "mount_geometry": mount_geometry,
        "attachment": attachment,
        "camera": camera,
        "sensors": sensors,
        "frames": frames,
        "articulation_roots": roots,
        "active_joints": joints,
        "workcell_articulation_roots": workcell_roots,
        "valid_prims": {
            "warehouse": stage.GetPrimAtPath("/World/FactorySRE/Warehouse").IsValid(),
            "go2": stage.GetPrimAtPath("/World/FactorySRE/Go2").IsValid(),
            "so101": stage.GetPrimAtPath("/World/FactorySRE/SO101").IsValid(),
            "ur5e": stage.GetPrimAtPath("/World/FactorySRE/Workcell/UR5e").IsValid(),
            "d455": stage.GetPrimAtPath(sensors["d455"]).IsValid(),
            "nanoscan3": stage.GetPrimAtPath(sensors["nanoscan3"]).IsValid(),
            "inspector83x": stage.GetPrimAtPath(
                "/World/FactorySRE/Workcell/Inspector83x"
            ).IsValid(),
            "wrist_camera": stage.GetPrimAtPath(camera["camera"]).IsValid(),
        },
        "physical_measurements_ready": (
            mount.measured_from_hardware and wrist_camera.calibrated_from_hardware
        ),
    }
