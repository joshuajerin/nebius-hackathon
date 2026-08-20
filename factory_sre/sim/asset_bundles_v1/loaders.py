"""Lazy Isaac Sim loaders for independently composable Factory SRE bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .contracts import (
    BundleConfiguration,
    BundleId,
    BundleLoadReport,
    MobileManipulatorConfiguration,
    WarehouseConfiguration,
    WorkcellConfiguration,
)


ISAAC_PATHS = {
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "go2": "/Isaac/Samples/Policies/go2/go2.usda",
    "go2_policy": "/Isaac/Samples/Policies/go2/physx_policy.pt",
    "go2_policy_env": "/Isaac/Samples/Policies/go2/physx_env.yaml",
    "ur5e": "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "d455": "/Isaac/Sensors/RealSense/D455/rsd455.usd",
    "nanoscan3": "/Isaac/Sensors/SICK/nanoScan3/SICK_nanoScan3.usd",
    "inspector83x": "/Isaac/Sensors/SICK/Inspector83x/SICK_Inspector83x.usd",
}


def _validate_root_path(stage, root_path: str) -> None:
    from pxr import Sdf

    path = Sdf.Path(root_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath():
        raise ValueError(f"bundle root must be an absolute USD prim path: {root_path}")
    if stage.GetPrimAtPath(root_path).IsValid():
        raise ValueError(f"bundle root is already occupied: {root_path}")


def _reference(stage, assets_root: str, asset_path: str, prim_path: str):
    import isaacsim.core.experimental.utils.stage as stage_utils

    stage_utils.add_reference_to_stage(usd_path=assets_root + asset_path, path=prim_path)
    return stage.GetPrimAtPath(prim_path)


def _update_stage(stage, frames: int = 4) -> None:
    import isaacsim.core.experimental.utils.app as app_utils

    stage.Load()
    for _ in range(frames):
        app_utils.update_app()


def _box(stage, path: str, position, size, *, mass_kg: float | None = None) -> str:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if mass_kg is not None:
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(mass_kg)
    return path


def _set_matrix_transform(prim, *, translation, rotation_xyz_deg=(0.0, 0.0, 0.0)) -> None:
    from pxr import Gf, UsdGeom

    scale = Gf.Matrix4d(1.0)
    rotate_x = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_xyz_deg[0]))
    rotate_y = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_xyz_deg[1]))
    rotate_z = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_xyz_deg[2]))
    translate = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*translation))
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(scale * rotate_x * rotate_y * rotate_z * translate)


def _load_mobile(root_path: str, config: MobileManipulatorConfiguration) -> BundleLoadReport:
    import antioch
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = antioch.stage()
    assets_root = get_assets_root_path()
    UsdGeom.Xform.Define(stage, root_path)
    go2_path = f"{root_path}/Go2"
    so101_path = f"{root_path}/SO101"
    _reference(stage, assets_root, ISAAC_PATHS["go2"], go2_path)
    antioch.load_asset("so101_antioch", prim_path=so101_path, version="1.3.2")
    _update_stage(stage)

    go2_body_path = f"{go2_path}/Geometry/base"
    arm_body_path = f"{so101_path}/base"
    arm_world_joint_path = f"{so101_path}/root_joint"
    arm_world_joint = stage.GetPrimAtPath(arm_world_joint_path)
    if arm_world_joint.IsValid():
        arm_world_joint.SetActive(False)
    arm_body = stage.GetPrimAtPath(arm_body_path)
    if arm_body.HasAPI(UsdPhysics.ArticulationRootAPI):
        arm_body.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    cache = UsdGeom.XformCache()
    go2_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(go2_body_path))
    arm_root_world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(so101_path))
    arm_body_world = cache.GetLocalToWorldTransform(arm_body)
    arm_body_in_root = arm_body_world * arm_root_world.GetInverse()
    mount_offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*config.chassis_to_arm_base_m))
    desired_body_world = mount_offset * go2_world
    desired_root_world = arm_body_in_root.GetInverse() * desired_body_world
    UsdGeom.Xformable(stage.GetPrimAtPath(so101_path)).MakeMatrixXform().Set(desired_root_world)

    mount_joint_path = f"{root_path}/Go2SO101MountJoint"
    joint = UsdPhysics.FixedJoint.Define(stage, mount_joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(go2_body_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(arm_body_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*config.chassis_to_arm_base_m))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    mount_path = _box(
        stage,
        f"{go2_body_path}/FactorySREArmMount",
        (0.0, 0.0, config.chassis_to_arm_base_m[2] / 2.0),
        config.mount_plate_size_m,
        mass_kg=config.mount_plate_mass_kg,
    )
    d455_path = f"{go2_body_path}/FactorySRE_D455"
    nanoscan_path = f"{go2_body_path}/FactorySRE_NanoScan3"
    d455 = _reference(stage, assets_root, ISAAC_PATHS["d455"], d455_path)
    nanoscan = _reference(stage, assets_root, ISAAC_PATHS["nanoscan3"], nanoscan_path)
    _set_matrix_transform(d455, translation=(0.245, 0.0, 0.055), rotation_xyz_deg=(0.0, 90.0, 0.0))
    _set_matrix_transform(nanoscan, translation=(0.0, 0.0, 0.115))

    camera_mount_path = f"{so101_path}/wrist/FactorySRECameraMount"
    camera_path = f"{camera_mount_path}/Camera"
    camera_mount = UsdGeom.Xform.Define(stage, camera_mount_path)
    camera_xform = UsdGeom.Xformable(camera_mount)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(*config.wrist_camera_translation_m))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(*config.wrist_camera_rotation_xyz_deg))
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(config.camera_focal_length_mm)
    camera.CreateHorizontalApertureAttr(config.camera_horizontal_aperture_mm)
    camera.CreateVerticalApertureAttr(config.camera_vertical_aperture_mm)
    camera.CreateClippingRangeAttr(Gf.Vec2f(*config.camera_clipping_range_m))

    tool_path = _box(
        stage,
        f"{so101_path}/gripper/FactorySREUsbTool",
        (0.0, 0.0, config.usb_tool_size_m[2] / 2.0),
        config.usb_tool_size_m,
        mass_kg=config.usb_tool_mass_kg,
    )
    usb_tip_path = f"{so101_path}/gripper/FactorySREFrames/usb_tip"
    usb_tip = UsdGeom.Xform.Define(stage, usb_tip_path)
    UsdGeom.Xformable(usb_tip).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, config.usb_tool_size_m[2]))
    base_frame_path = f"{go2_body_path}/FactorySREFrames/base"
    arm_mount_frame_path = f"{go2_body_path}/FactorySREFrames/arm_mount"
    UsdGeom.Xform.Define(stage, base_frame_path)
    arm_mount_frame = UsdGeom.Xform.Define(stage, arm_mount_frame_path)
    UsdGeom.Xformable(arm_mount_frame).AddTranslateOp().Set(Gf.Vec3d(*config.chassis_to_arm_base_m))
    _update_stage(stage)

    ready = config.mount_measured and config.wrist_camera_calibrated and config.usb_tool_measured
    warnings = () if ready else (
        "prototype mount, wrist-camera pose, or USB-tool geometry is not physical-hardware calibrated",
    )
    return BundleLoadReport(
        bundle_id=BundleId.MOBILE_MANIPULATOR,
        root_path=root_path,
        prim_paths={
            "go2": go2_path,
            "so101": so101_path,
            "mount_joint": mount_joint_path,
            "mount": mount_path,
            "d455": d455_path,
            "nanoscan3": nanoscan_path,
            "wrist_camera": camera_path,
            "usb_tool": tool_path,
            "usb_tip": usb_tip_path,
            "base_frame": base_frame_path,
            "arm_mount_frame": arm_mount_frame_path,
        },
        asset_sources={
            "go2": assets_root + ISAAC_PATHS["go2"],
            "go2_policy": assets_root + ISAAC_PATHS["go2_policy"],
            "go2_policy_env": assets_root + ISAAC_PATHS["go2_policy_env"],
            "so101": "so101_antioch@1.3.2",
            "d455": assets_root + ISAAC_PATHS["d455"],
            "nanoscan3": assets_root + ISAAC_PATHS["nanoscan3"],
        },
        deployment_ready=ready,
        warnings=warnings,
        metadata={"expected_active_joint_count": 19, "expected_articulation_roots": 1},
    )


def _tag_material(stage, material_path: str, image_path: Path):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(image_path)))
    reader = UsdShade.Shader.Define(stage, f"{material_path}/Reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _tag(stage, frame_path: str, tag_id: int, size_m: float, image_path: Path) -> str:
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    mesh_path = f"{frame_path}/Tag{tag_id}"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    half = size_m / 2.0
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-half, 0.0, -half),
            Gf.Vec3f(half, 0.0, -half),
            Gf.Vec3f(half, 0.0, half),
            Gf.Vec3f(-half, 0.0, half),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    material = _tag_material(stage, f"{mesh_path}/Material", image_path)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh_path


def _load_workcell(root_path: str, config: WorkcellConfiguration) -> BundleLoadReport:
    import antioch
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, UsdGeom

    stage = antioch.stage()
    assets_root = get_assets_root_path()
    UsdGeom.Xform.Define(stage, root_path)
    table_top_z = config.table_height_m - config.table_size_m[2] / 2.0
    table_path = _box(stage, f"{root_path}/Workbench/Table", (0.0, 0.0, table_top_z), config.table_size_m)
    for index, (x, y) in enumerate(((-0.78, -0.34), (-0.78, 0.34), (0.78, -0.34), (0.78, 0.34))):
        _box(stage, f"{root_path}/Workbench/Leg{index}", (x, y, config.table_height_m / 2.0), (0.07, 0.07, config.table_height_m))

    panel_path = _box(
        stage,
        f"{root_path}/Workbench/DiagnosticPanel/Panel",
        (0.0, -0.455, 0.98),
        config.panel_size_m,
    )
    coarse_frame_path = f"{root_path}/Workbench/DiagnosticPanel/coarse_tag"
    fine_frame_path = f"{root_path}/Workbench/DiagnosticPanel/fine_tag"
    port_frame_path = f"{root_path}/Workbench/DiagnosticPanel/usb_port"
    coarse_frame = UsdGeom.Xform.Define(stage, coarse_frame_path)
    fine_frame = UsdGeom.Xform.Define(stage, fine_frame_path)
    port_frame = UsdGeom.Xform.Define(stage, port_frame_path)
    UsdGeom.Xformable(coarse_frame).AddTranslateOp().Set(Gf.Vec3d(-0.17, -0.474, 1.01))
    fine_position = (0.17, -0.474, 1.01)
    UsdGeom.Xformable(fine_frame).AddTranslateOp().Set(Gf.Vec3d(*fine_position))
    port_position = tuple(a + b for a, b in zip(fine_position, config.fine_tag_to_port_m, strict=True))
    UsdGeom.Xformable(port_frame).AddTranslateOp().Set(Gf.Vec3d(*port_position))
    tag_root = Path.cwd() / "assets" / "tags"
    coarse_tag_path = _tag(
        stage,
        coarse_frame_path,
        config.coarse_tag_id,
        config.coarse_tag_size_m,
        tag_root / f"tag36h11_{config.coarse_tag_id}.png",
    )
    fine_tag_path = _tag(
        stage,
        fine_frame_path,
        config.fine_tag_id,
        config.fine_tag_size_m,
        tag_root / f"tag36h11_{config.fine_tag_id}.png",
    )
    port_geometry_path = _box(
        stage,
        f"{port_frame_path}/PortBezel",
        (0.0, 0.0, 0.0),
        (0.016, 0.008, 0.008),
    )

    ur5e_path = f"{root_path}/UR5e"
    inspector_path = f"{root_path}/Inspector83x"
    ur5e = _reference(stage, assets_root, ISAAC_PATHS["ur5e"], ur5e_path)
    inspector = _reference(stage, assets_root, ISAAC_PATHS["inspector83x"], inspector_path)
    _set_matrix_transform(ur5e, translation=(0.0, 0.10, config.table_height_m))
    _set_matrix_transform(inspector, translation=(0.0, -0.15, 1.65), rotation_xyz_deg=(90.0, 0.0, 0.0))
    _update_stage(stage)

    ready = config.panel_measured and config.tag_to_port_calibrated
    warnings = () if ready else (
        "prototype workbench/panel geometry or tag-to-port transform is not hardware calibrated",
    )
    return BundleLoadReport(
        bundle_id=BundleId.WORKCELL,
        root_path=root_path,
        prim_paths={
            "table": table_path,
            "panel": panel_path,
            "ur5e": ur5e_path,
            "inspector83x": inspector_path,
            "coarse_tag": coarse_tag_path,
            "fine_tag": fine_tag_path,
            "coarse_tag_frame": coarse_frame_path,
            "fine_tag_frame": fine_frame_path,
            "usb_port_frame": port_frame_path,
            "usb_port_geometry": port_geometry_path,
        },
        asset_sources={
            "ur5e": assets_root + ISAAC_PATHS["ur5e"],
            "inspector83x": assets_root + ISAAC_PATHS["inspector83x"],
            "coarse_tag_texture": str(tag_root / f"tag36h11_{config.coarse_tag_id}.png"),
            "fine_tag_texture": str(tag_root / f"tag36h11_{config.fine_tag_id}.png"),
        },
        deployment_ready=ready,
        warnings=warnings,
        metadata={
            "coarse_tag_id": config.coarse_tag_id,
            "coarse_tag_size_m": config.coarse_tag_size_m,
            "fine_tag_id": config.fine_tag_id,
            "fine_tag_size_m": config.fine_tag_size_m,
            "fine_tag_to_port_m": config.fine_tag_to_port_m,
            "expected_articulation_roots": 1,
        },
    )


def _load_warehouse(root_path: str, config: WarehouseConfiguration) -> BundleLoadReport:
    import antioch
    from isaacsim.storage.native import get_assets_root_path

    stage = antioch.stage()
    assets_root = get_assets_root_path()
    warehouse = _reference(stage, assets_root, ISAAC_PATHS["warehouse"], root_path)
    _set_matrix_transform(
        warehouse,
        translation=config.translation_m,
        rotation_xyz_deg=config.rotation_xyz_deg,
    )
    _update_stage(stage)
    return BundleLoadReport(
        bundle_id=BundleId.WAREHOUSE,
        root_path=root_path,
        prim_paths={"warehouse": root_path},
        asset_sources={"warehouse": assets_root + ISAAC_PATHS["warehouse"]},
        deployment_ready=True,
        metadata={"stock_reference": True},
    )


def load_bundle(
    bundle_id: BundleId | str,
    prim_path: str,
    configuration: BundleConfiguration | None = None,
) -> BundleLoadReport:
    """Load one requested bundle without importing unrelated scene assets."""

    import antioch
    from pxr import UsdGeom

    selected = BundleId(bundle_id)
    stage = antioch.stage()
    _validate_root_path(stage, prim_path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    if selected == BundleId.MOBILE_MANIPULATOR:
        if configuration is not None and not isinstance(configuration, MobileManipulatorConfiguration):
            raise TypeError("mobile_manipulator requires MobileManipulatorConfiguration")
        return _load_mobile(prim_path, configuration or MobileManipulatorConfiguration())
    if selected == BundleId.WORKCELL:
        if configuration is not None and not isinstance(configuration, WorkcellConfiguration):
            raise TypeError("workcell requires WorkcellConfiguration")
        return _load_workcell(prim_path, configuration or WorkcellConfiguration())
    if configuration is not None and not isinstance(configuration, WarehouseConfiguration):
        raise TypeError("warehouse requires WarehouseConfiguration")
    return _load_warehouse(prim_path, configuration or WarehouseConfiguration())


def load_factory_sre_scene(
    root_path: str = "/World/FactorySREBundles",
    configurations: Mapping[BundleId, BundleConfiguration] | None = None,
) -> dict[BundleId, BundleLoadReport]:
    """Compose the same three independent loaders under non-overlapping roots."""

    import antioch
    from pxr import UsdGeom

    stage = antioch.stage()
    _validate_root_path(stage, root_path)
    UsdGeom.Xform.Define(stage, root_path)
    supplied = configurations or {}
    paths = {
        BundleId.WAREHOUSE: f"{root_path}/Warehouse",
        BundleId.MOBILE_MANIPULATOR: f"{root_path}/MobileManipulator",
        BundleId.WORKCELL: f"{root_path}/Workcell",
    }
    return {
        bundle_id: load_bundle(bundle_id, path, supplied.get(bundle_id))
        for bundle_id, path in paths.items()
    }
