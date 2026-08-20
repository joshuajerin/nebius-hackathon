"""Recorded visual demonstration for the active-only wide factory overlay."""

from __future__ import annotations

import antioch

from factory_sre.sim.industrial_cells_v1 import IndustrialCellConfig, build_industrial_cell_grid


PROFILE = antioch.BootProfile(
    physics_dt=1.0 / 200.0,
    render_dt=1.0 / 50.0,
    physics_engine="physx",
    render_quality="performance",
    viewport=(1280, 720),
)


def _frame_checks(run: antioch.ScenarioRun, rgb, *, label: str) -> bool:
    import numpy as np

    mean = float(rgb.mean())
    contrast = float(rgb.std())
    subject_pixels = int(np.count_nonzero((rgb[..., 1] > 80) | (rgb[..., 0] > 80)))
    run.check(f"{label} frame has useful exposure", 10.0 <= mean <= 220.0, detail=f"mean {mean:.1f}")
    run.check(f"{label} frame has useful contrast", contrast >= 5.0, detail=f"std {contrast:.1f}")
    run.check(
        f"{label} frame contains visible subjects",
        subject_pixels >= 3_000,
        detail=f"subject pixels {subject_pixels}",
    )
    return 10.0 <= mean <= 220.0 and contrast >= 5.0 and subject_pixels >= 3_000


def _overview_lighting_checks(run: antioch.ScenarioRun, rgb) -> None:
    import numpy as np

    height, width = rgb.shape[:2]
    quadrants = (
        rgb[: height // 2, : width // 2],
        rgb[: height // 2, width // 2 :],
        rgb[height // 2 :, : width // 2],
        rgb[height // 2 :, width // 2 :],
    )
    means = [float(quadrant.mean()) for quadrant in quadrants]
    clipped_ratio = float(np.mean(np.all(rgb >= 250, axis=-1)))
    run.check(
        "warehouse lighting reaches every image quadrant",
        min(means) >= 18.0 and max(means) - min(means) <= 110.0,
        detail=f"quadrant means {[round(value, 1) for value in means]}",
    )
    run.check(
        "warehouse lighting avoids clipped highlights",
        clipped_ratio <= 0.08,
        detail=f"clipped ratio {clipped_ratio:.4f}",
    )


def _maximum_roll_pitch_deg(quaternion) -> float:
    from math import asin, atan2, degrees

    w, x, y, z = (float(value) for value in quaternion)
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return max(abs(degrees(roll)), abs(degrees(pitch)))


@antioch.scenario(
    name="industrial_cells_scene_smoke",
    description="Render a bright high-bay warehouse with eight active UR5e cells and a parked Go2-SO101 composite.",
    tags=("smoke", "scene", "industrial", "ur5e", "pick-place", "composite"),
    sim=PROFILE,
    capture=True,
)
def industrial_cells_scene_smoke(run: antioch.ScenarioRun) -> None:
    import numpy as np
    from PIL import Image
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Usd, UsdGeom

    SimulationManager.set_physics_sim_device("cuda")
    scene = build_industrial_cell_grid(IndustrialCellConfig())
    world = antioch.world()
    scene.prepare_runtime()
    world.reset()
    scene.initialize_runtime()
    set_camera_view(
        eye=[10.4, -7.5, 4.35],
        target=[3.9, 1.6, 1.35],
        camera_prim_path="/OmniverseKit_Persp",
    )

    phase_history: dict[str, set[str]] = {asset_id: set() for asset_id in scene.cell_ids}
    joint_samples: dict[str, list[np.ndarray]] = {asset_id: [] for asset_id in scene.cell_ids}
    chassis_positions: list[np.ndarray] = []
    chassis_orientations: list[np.ndarray] = []
    arm_base_positions: list[np.ndarray] = []
    arm_joint_samples: list[np.ndarray] = []
    for step in range(1_800):
        scene.step(1.0 / 200.0)
        world.step(render=True)
        if step % 30 == 0:
            snapshot = scene.snapshot()
            for cell in snapshot["cells"]:
                phase_history[cell["asset_id"]].add(cell["phase"])
                if cell["joint_positions_rad"] is not None:
                    joint_samples[cell["asset_id"]].append(np.asarray(cell["joint_positions_rad"], dtype=float))
            if step >= 400:
                mobile = snapshot["mobile_manipulator"]
                chassis_positions.append(np.asarray(mobile["chassis_position_m"], dtype=float))
                chassis_orientations.append(
                    np.asarray(mobile["chassis_orientation_wxyz"], dtype=float)
                )
                arm_base_positions.append(np.asarray(mobile["arm_base_position_m"], dtype=float))
                arm_joint_samples.append(
                    np.asarray(mobile["arm_joint_positions_rad"], dtype=float)
                )

    snapshot = scene.snapshot()
    cells = snapshot["cells"]
    articulations = {cell["articulation_root"] for cell in cells}
    service_docks = {cell["service_dock_path"] for cell in cells}
    tag_paths = [path for cell in cells for path in cell["tag_paths"]]
    active_motion = {
        asset_id: max(
            (float(np.linalg.norm(right - left)) for left, right in zip(samples, samples[1:], strict=False)),
            default=0.0,
        )
        for asset_id, samples in joint_samples.items()
    }
    centers = np.asarray([cell["center_m"] for cell in cells], dtype=float)
    unique_x = np.unique(centers[:, 0])
    unique_y = np.unique(centers[:, 1])
    stage = antioch.stage()
    warehouse_shell = stage.GetPrimAtPath(snapshot["warehouse_shell_path"])
    mobile = snapshot["mobile_manipulator"]
    ceiling_lights = snapshot["lighting_paths"]["ceiling"]
    fill_lights = snapshot["lighting_paths"]["fill"]
    root_layer_text = stage.GetRootLayer().ExportToString().lower()
    forbidden_warehouse_prims = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(warehouse_shell)
        if any(token in prim.GetName().lower() for token in ("shelf", "pallet", "crate", "forklift"))
    ]
    dock_dimensions = [
        (
            float(UsdGeom.Cylinder(stage.GetPrimAtPath(path)).GetRadiusAttr().Get()),
            float(UsdGeom.Cylinder(stage.GetPrimAtPath(path)).GetHeightAttr().Get()),
        )
        for path in service_docks
    ]
    cardinal_tag_checks = []
    for cell in cells:
        x, y, _ = cell["center_m"]
        expected_faces = (
            ((x, y + 0.51, 0.32), "North"),
            ((x, y - 0.51, 0.32), "South"),
            ((x + 0.51, y, 0.32), "East"),
            ((x - 0.51, y, 0.32), "West"),
        )
        for path, (expected_center, face_name) in zip(cell["tag_paths"], expected_faces, strict=True):
            points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(), dtype=float)
            cardinal_tag_checks.append(
                face_name in path
                and path.endswith(str(29 + cell["box_number"]))
                and bool(np.allclose(points.mean(axis=0), expected_center, atol=1e-6))
            )

    run.check("scene contains eight industrial cells", len(cells) == 8, detail=str(len(cells)))
    run.check(
        "cells are inside a complete high-bay warehouse shell",
        warehouse_shell.IsValid()
        and all(
            stage.GetPrimAtPath(f"{snapshot['warehouse_shell_path']}/{name}").IsValid()
            for name in (
                "Floor",
                "BackWall",
                "LeftWall",
                "RightWall",
                "Roof",
                "SteelFrame",
                "RoofBeams",
                "RoofTrusses",
                "Clerestory",
                "LoadingDoors",
                "AisleMarkings",
            )
        ),
        detail=snapshot["warehouse_shell_path"],
    )
    run.check(
        "stock warehouse environment and warehouse props are absent",
        "simple_warehouse" not in root_layer_text
        and "warehouse_multiple_shelves" not in root_layer_text
        and not forbidden_warehouse_prims,
        detail=str(forbidden_warehouse_prims),
    )
    run.check(
        "warehouse has twelve ceiling lights",
        len(ceiling_lights) == 12
        and all(stage.GetPrimAtPath(path).IsValid() for path in ceiling_lights),
        detail=str(ceiling_lights),
    )
    run.check(
        "warehouse has four-direction fill lighting",
        len(fill_lights) == 4 and all(stage.GetPrimAtPath(path).IsValid() for path in fill_lights),
        detail=str(fill_lights),
    )
    run.check(
        "one policy-compatible Go2 and one SO-101 are present",
        stage.GetPrimAtPath("/World/Go2").IsValid()
        and stage.GetPrimAtPath("/World/SO101").IsValid()
        and len(mobile["go2_movable_joints"]) == 12
        and len(mobile["so101_movable_joints"]) == 6,
        detail=f"{len(mobile['go2_movable_joints'])} + {len(mobile['so101_movable_joints'])} joints",
    )
    run.check(
        "SO-101 is physically mounted to the Go2 chassis",
        stage.GetPrimAtPath(mobile["mount_joint"]).IsValid()
        and tuple(mobile["mount_body0"]) == (mobile["go2_chassis"],)
        and tuple(mobile["mount_body1"]) == (mobile["so101_base"],),
        detail=mobile["mount_joint"],
    )
    run.check(
        "SO-101 uses its ordinary wrist camera and circular service tool",
        stage.GetPrimAtPath(mobile["wrist_camera"]).IsA(UsdGeom.Camera)
        and stage.GetPrimAtPath(mobile["end_effector_asset"]).IsValid()
        and mobile["end_effector_kind"] == "magnetic_circular_service_puck",
        detail=f"{mobile['wrist_camera']} + {mobile['end_effector_asset']}",
    )
    run.check(
        "all six SO-101 stow drives are configured",
        mobile["stow_drive_count"] == 6,
        detail=str(mobile["stow_drive_count"]),
    )
    run.check("each cell owns a unique UR5e articulation", len(articulations) == 8, detail=str(sorted(articulations)))
    run.check("each cell has a unique circular service dock", len(service_docks) == 8, detail=str(sorted(service_docks)))
    run.check(
        "service docks use the 80 millimetre puck dimensions",
        all(np.allclose(dimensions, (0.040, 0.025), atol=1e-6) for dimensions in dock_dimensions),
        detail=str(dock_dimensions),
    )
    run.check("each cell exposes four cardinal AprilTags", all(len(cell["tag_paths"]) == 4 for cell in cells), detail=str(tag_paths))
    run.check("scene contains 32 unique AprilTags", len(tag_paths) == 32 and len(set(tag_paths)) == 32, detail=str(len(tag_paths)))
    run.check("AprilTags retain cell identity and cardinal placement", all(cardinal_tag_checks), detail=str(cardinal_tag_checks))
    run.check("industrial grid has four wide columns", len(unique_x) == 4, detail=str(unique_x.tolist()))
    run.check("industrial grid has two wide rows", len(unique_y) == 2, detail=str(unique_y.tolist()))
    run.check(
        "industrial columns are 2.6 metres apart",
        bool(np.allclose(np.diff(unique_x), 2.6, atol=1e-6)),
        detail=str(np.diff(unique_x).tolist()),
    )
    run.check(
        "industrial rows are 3.2 metres apart",
        bool(np.allclose(np.diff(unique_y), 3.2, atol=1e-6)),
        detail=str(np.diff(unique_y).tolist()),
    )
    run.check(
        "all cells execute pick lift bowl and release phases",
        all({"pick", "lift", "bowl", "release"}.issubset(phase_history[cell["asset_id"]]) for cell in cells),
        detail=str({asset_id: sorted(phases) for asset_id, phases in phase_history.items()}),
    )
    run.check(
        "all UR5e joints visibly move",
        all(active_motion[cell["asset_id"]] > 0.03 for cell in cells),
        detail=str(active_motion),
    )
    run.check(
        "snapshot exposes no per-cell mode state",
        all("mode" not in cell for cell in cells),
        detail=str(snapshot.keys()),
    )

    chassis_array = np.asarray(chassis_positions, dtype=float)
    arm_base_array = np.asarray(arm_base_positions, dtype=float)
    arm_joint_array = np.asarray(arm_joint_samples, dtype=float)
    horizontal_drift_m = float(
        np.max(np.linalg.norm(chassis_array[:, :2] - chassis_array[0, :2], axis=1))
    )
    maximum_start_error_m = float(
        np.max(
            np.linalg.norm(
                chassis_array[:, :2]
                - np.asarray(mobile["start_position_m"], dtype=float)[:2],
                axis=1,
            )
        )
    )
    maximum_tilt_deg = max(
        _maximum_roll_pitch_deg(quaternion) for quaternion in chassis_orientations
    )
    mount_separations = np.linalg.norm(arm_base_array - chassis_array, axis=1)
    maximum_mount_variation_m = float(np.max(np.abs(mount_separations - mount_separations[0])))
    maximum_arm_joint_variation_rad = float(
        np.max(np.ptp(arm_joint_array, axis=0))
    )
    run.check(
        "parked Go2 remains at a standing chassis height",
        float(chassis_array[:, 2].min()) >= 0.18
        and float(chassis_array[:, 2].max()) <= 0.65,
        detail=f"z {chassis_array[:, 2].min():.3f}..{chassis_array[:, 2].max():.3f} m",
    )
    run.check(
        "parked Go2 remains level and does not drift",
        maximum_tilt_deg <= 10.0 and maximum_start_error_m <= 0.10,
        detail=(
            f"tilt {maximum_tilt_deg:.2f} deg, "
            f"max start error {maximum_start_error_m:.3f} m, "
            f"sample drift {horizontal_drift_m:.3f} m"
        ),
    )
    run.check(
        "SO-101 remains attached and stowed",
        maximum_mount_variation_m <= 0.03 and maximum_arm_joint_variation_rad <= 0.10,
        detail=(
            f"mount variation {maximum_mount_variation_m:.4f} m, "
            f"max arm joint variation {maximum_arm_joint_variation_rad:.3f} rad"
        ),
    )

    overview = antioch.capture_viewport()
    run.check("wide industrial overview captured", overview is not None, detail="1280x720")
    if overview is not None:
        overview_rgb = np.asarray(overview)[..., :3]
        if _frame_checks(run, overview_rgb, label="wide industrial overview"):
            antioch.Logger("factory_sre/industrial_cells").image("camera/overview", overview_rgb)
        _overview_lighting_checks(run, overview_rgb)
        overview_path = "/tmp/wide_industrial_cells_overview.png"
        Image.fromarray(overview_rgb).save(overview_path)
        run.add_artifact(overview_path, name="wide-industrial-cells-overview.png", content_type="image/png")

    representative = cells[0]
    dock_x, dock_y, _ = representative["center_m"]
    set_camera_view(
        eye=[dock_x, dock_y + 1.70, 0.95],
        target=[dock_x, dock_y + 0.53, 0.48],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(20):
        scene.step(1.0 / 200.0)
        world.step(render=True)
    closeup = antioch.capture_viewport()
    run.check("service-dock closeup captured", closeup is not None, detail=representative["asset_id"])
    if closeup is not None:
        closeup_rgb = np.asarray(closeup)[..., :3]
        if _frame_checks(run, closeup_rgb, label="service-dock closeup"):
            antioch.Logger("factory_sre/industrial_cells").image("camera/service_dock", closeup_rgb)
        closeup_path = "/tmp/service_dock_closeup.png"
        Image.fromarray(closeup_rgb).save(closeup_path)
        run.add_artifact(closeup_path, name="service-dock-closeup.png", content_type="image/png")

    set_camera_view(
        eye=[1.95, -3.55, 1.80],
        target=[3.9, -1.55, 0.78],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(30):
        scene.step(1.0 / 200.0)
        world.step(render=True)
    composite_closeup = antioch.capture_viewport()
    run.check("Go2-SO101 closeup captured", composite_closeup is not None, detail="1280x720")
    if composite_closeup is not None:
        composite_rgb = np.asarray(composite_closeup)[..., :3]
        if _frame_checks(run, composite_rgb, label="Go2-SO101 closeup"):
            antioch.Logger("factory_sre/industrial_cells").image(
                "camera/mobile_manipulator", composite_rgb
            )
        composite_path = "/tmp/go2_so101_warehouse_closeup.png"
        Image.fromarray(composite_rgb).save(composite_path)
        run.add_artifact(
            composite_path,
            name="go2-so101-warehouse-closeup.png",
            content_type="image/png",
        )
    run.add_result("industrial_cells", scene.snapshot())
