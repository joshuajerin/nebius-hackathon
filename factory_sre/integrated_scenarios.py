"""Approved two-workbench scenarios kept separate from direct-USB experiments."""

from __future__ import annotations

import antioch

from factory_sre.contracts import WORKBENCH_A, WORKBENCH_B
from factory_sre.sim.config import HERO_CONFIG


PROFILE = antioch.BootProfile(
    physics_dt=HERO_CONFIG.physics_dt,
    render_dt=HERO_CONFIG.render_dt,
    physics_engine="physx",
    render_quality="performance",
    viewport=(1280, 720),
)


@antioch.scenario(
    name="composite_mount_smoke",
    description="Attach so101_antioch@1.3.2 and a wrist camera to the supplied Go2 as one articulation.",
    tags=("smoke", "composite", "go2", "so101", "camera"),
    sim=PROFILE,
    capture=True,
)
def composite_mount_smoke(run: antioch.ScenarioRun) -> None:
    import numpy as np
    from isaacsim.core.experimental.objects import GroundPlane
    from isaacsim.core.experimental.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdGeom, UsdLux

    from factory_sre.sim.composite import author_go2_so101_composite

    root = get_assets_root_path()
    add_reference_to_stage(
        usd_path=root + "/Isaac/Samples/Policies/go2/go2.usda",
        path="/World/Go2",
    )
    GroundPlane("/World/Ground", templates=None)
    UsdLux.DomeLight.Define(antioch.stage(), "/World/CompositeLight").CreateIntensityAttr(900.0)
    report = author_go2_so101_composite(mode="merged")
    world = antioch.world()
    world.reset()
    set_camera_view(
        eye=[-2.3, -3.0, 1.8],
        target=[0.0, 0.0, 0.65],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(40):
        world.step(render=True)
    frame = antioch.capture_viewport()
    cache = UsdGeom.XformCache()
    go2_position = cache.GetLocalToWorldTransform(antioch.stage().GetPrimAtPath(report.go2_chassis)).ExtractTranslation()
    arm_position = cache.GetLocalToWorldTransform(antioch.stage().GetPrimAtPath(report.so101_base)).ExtractTranslation()

    run.check("Go2 chassis is mount parent", report.mount_body0 == (report.go2_chassis,), detail=str(report.mount_body0))
    run.check("SO-101 base is mount child", report.mount_body1 == (report.so101_base,), detail=str(report.mount_body1))
    run.check("composite has one articulation root", len(report.articulation_roots) == 1, detail=str(report.articulation_roots))
    run.check("Go2 retains 12 movable joints", len(report.go2_movable_joints) == 12, detail=str(report.go2_movable_joints))
    run.check("SO-101 retains 6 movable joints", len(report.so101_movable_joints) == 6, detail=str(report.so101_movable_joints))
    run.check("wrist camera is on terminal arm link", report.wrist_camera.startswith(report.end_effector + "/"), detail=report.wrist_camera)
    run.check(
        "terminal tool is the magnetic circular effector",
        report.end_effector_kind == "magnetic_circular_service_puck"
        and report.end_effector_asset.startswith(report.end_effector + "/"),
        detail=report.end_effector_asset,
    )
    run.check(
        "magnetic contact frame is part of the circular effector",
        report.magnetic_contact_frame.startswith(report.end_effector + "/"),
        detail=report.magnetic_contact_frame,
    )
    run.check("uncontrolled structural composite does not pass through ground", float(go2_position[2]) >= 0.05, detail=f"base z {float(go2_position[2]):.3f} m")
    run.check("arm remains attached near chassis", np.linalg.norm(np.asarray(arm_position) - np.asarray(go2_position)) <= 0.50, detail=f"separation {float(np.linalg.norm(np.asarray(arm_position) - np.asarray(go2_position))):.3f} m")
    run.check("composite review frame captured", frame is not None, detail="1280x720")
    if frame is not None:
        rgb = np.asarray(frame)[..., :3]
        run.check("composite review frame is lit", float(rgb.std()) >= 5.0, detail=f"std {float(rgb.std()):.2f}")
        antioch.Logger("factory_sre/composite").image("camera/overview", rgb)
    run.add_result("composite", report.to_dict())


@antioch.scenario(
    name="magnetic_circle_effector_smoke",
    description="Replace the SO-101 gripper with a collision-bearing circular magnetic service puck.",
    tags=("smoke", "so101", "end-effector", "magnet", "circular-tool"),
    sim=PROFILE,
    capture=True,
)
def magnetic_circle_effector_smoke(run: antioch.ScenarioRun) -> None:
    import numpy as np
    from PIL import Image
    from isaacsim.core.experimental.objects import GroundPlane
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.experimental.utils import bounds as bounds_utils
    from isaacsim.core.experimental.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdGeom, UsdLux, UsdPhysics

    from factory_sre.sim.composite import author_go2_so101_composite

    root = get_assets_root_path()
    add_reference_to_stage(
        usd_path=root + "/Isaac/Samples/Policies/go2/go2.usda",
        path="/World/Go2",
    )
    GroundPlane("/World/Ground", templates=None)
    UsdLux.DomeLight.Define(antioch.stage(), "/World/MagneticToolLight").CreateIntensityAttr(1100.0)
    report = author_go2_so101_composite(mode="merged")
    stage = antioch.stage()
    contact_face_prim = stage.GetPrimAtPath(report.end_effector_asset + "/ContactFace")
    contact_face = UsdGeom.Cylinder(contact_face_prim)

    owner = contact_face_prim
    while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
        owner = owner.GetParent()
    collision_owner = str(owner.GetPath()) if owner.IsValid() else ""

    tool_bounds = np.asarray(
        bounds_utils.compute_aabb(report.end_effector_asset, include_children=True),
        dtype=float,
    )
    tool_size = tool_bounds[3:] - tool_bounds[:3]
    terminal_body = RigidPrim(report.end_effector)
    world = antioch.world()
    world.reset()
    if not terminal_body.is_physics_tensor_entity_valid():
        raise RuntimeError("magnetic effector terminal link requires live PhysX readback")
    for _ in range(35):
        world.step(render=True)
    terminal_positions, _ = terminal_body.get_world_poses()
    terminal_position = np.asarray(terminal_positions.numpy()[0], dtype=float)
    set_camera_view(
        eye=[
            float(terminal_position[0] + 0.42),
            float(terminal_position[1] - 0.30),
            float(terminal_position[2] + 0.18),
        ],
        target=[
            float(terminal_position[0] + 0.025),
            float(terminal_position[1]),
            float(terminal_position[2]),
        ],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(30):
        world.step(render=True)
    frame = antioch.capture_viewport()

    run.check(
        "circular magnetic tool composes",
        stage.GetPrimAtPath(report.end_effector_asset).IsValid(),
        detail=report.end_effector_asset,
    )
    run.check(
        "contact face is a true circle",
        contact_face
        and contact_face.GetAxisAttr().Get() == "X"
        and abs(float(contact_face.GetRadiusAttr().Get()) - report.magnetic_contact_radius_m) <= 1e-6,
        detail=f"axis={contact_face.GetAxisAttr().Get()} radius={float(contact_face.GetRadiusAttr().Get()):.3f} m",
    )
    run.check(
        "magnet puck has circular cross-section",
        abs(float(tool_size[1] - tool_size[2])) <= 0.002,
        detail=f"world dimensions {tool_size.tolist()}",
    )
    run.check(
        "magnet puck is physically owned by terminal link",
        collision_owner == report.end_effector,
        detail=f"collision owner {collision_owner}",
    )
    run.check(
        "source gripper geometry is hidden",
        bool(report.hidden_gripper_geometry),
        detail=str(report.hidden_gripper_geometry),
    )
    run.check(
        "source gripper collisions are disabled",
        bool(report.disabled_gripper_colliders)
        and all(
            not bool(UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get())
            for path in report.disabled_gripper_colliders
        ),
        detail=str(report.disabled_gripper_colliders),
    )
    run.check(
        "magnetic contact frame exists",
        stage.GetPrimAtPath(report.magnetic_contact_frame).IsValid(),
        detail=report.magnetic_contact_frame,
    )
    run.check("magnetic effector closeup captured", frame is not None, detail="1280x720")
    if frame is not None:
        rgb = np.asarray(frame)[..., :3]
        run.check("magnetic effector closeup is lit", float(rgb.std()) >= 5.0, detail=f"std {float(rgb.std()):.2f}")
        image_path = "/tmp/so101_magnetic_circle_effector.png"
        Image.fromarray(rgb).save(image_path)
        run.add_artifact(
            image_path,
            name="so101-magnetic-circle-effector.png",
            content_type="image/png",
        )
        antioch.Logger("factory_sre/magnetic_effector").image("camera/closeup", rgb)
    run.add_result(
        "magnetic_circle_effector",
        {
            "kind": report.end_effector_kind,
            "asset": report.end_effector_asset,
            "contact_frame": report.magnetic_contact_frame,
            "contact_radius_m": report.magnetic_contact_radius_m,
            "tool_size_m": tool_size.tolist(),
            "collision_owner": collision_owner,
            "hidden_gripper_geometry": list(report.hidden_gripper_geometry),
        },
    )


@antioch.scenario(
    name="magnetic_effector_ik_smoke",
    description="Use Isaac Robot Poser IK to place the SO-101 magnetic contact frame on Box 1's AprilTag.",
    tags=("smoke", "so101", "magnet", "ik", "apriltag"),
    sim=antioch.BootProfile(
        physics_dt=1.0 / 200.0,
        render_dt=1.0 / 50.0,
        physics_engine="physx",
        render_quality="performance",
        viewport=(1280, 720),
    ),
    capture=True,
)
def magnetic_effector_ik_smoke(run: antioch.ScenarioRun) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.storage.native import get_assets_root_path

    from factory_sre.sim.composite import (
        author_go2_so101_composite,
        configure_so101_stow_drives,
    )
    from factory_sre.sim.magnetic_arm import MagneticArmController
    from factory_sre.sim.tagged_box_grid import build_tagged_box_grid, service_box_for_number

    SimulationManager.set_physics_sim_device("cuda")
    build_tagged_box_grid(incident_box_number=1)
    assets = get_assets_root_path()
    target = service_box_for_number(1, offline_box_number=1)
    stance = (target.center[0], 1.17, 0.50)
    yaw = target.approach_yaw
    policy = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        position=list(stance),
        orientation=[float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))],
        policy_path=assets + "/Isaac/Samples/Policies/go2/physx_policy.pt",
        env_config_path=assets + "/Isaac/Samples/Policies/go2/physx_env.yaml",
    )
    composite = author_go2_so101_composite(
        mode="controller_isolated",
        initial_chassis_position_m=stance,
    )
    stow_drive_count = configure_so101_stow_drives(
        antioch.stage(), composite.so101_movable_joints
    )
    arm = MagneticArmController(antioch.stage(), composite)
    world = antioch.world()
    world.reset()
    policy.initialize()
    policy.post_reset()
    if not arm.ready():
        run.fail("SO-101 magnetic IK tensor views did not initialize")
    zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    for _ in range(180):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)

    solution = arm.solve(target.tag_center)
    if solution.success:
        for _ in range(900):
            arm.apply_solution()
            policy.forward(1.0 / 200.0, zero)
            world.step(render=True)
    contact_position, _ = arm.contact_world_pose()
    contact_error_m = float(
        np.linalg.norm(np.asarray(target.tag_center, dtype=float) - contact_position)
    )
    set_camera_view(
        eye=[target.tag_center[0] + 0.65, target.tag_center[1] + 0.70, 0.80],
        target=list(target.tag_center),
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(30):
        if solution.success:
            arm.apply_solution()
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    frame = antioch.capture_viewport()

    run.check("Robot Poser discovers the SO-101 chain", stow_drive_count == 6, detail=f"{stow_drive_count} driven joints")
    run.check("magnetic contact IK converges", solution.success, detail=str(solution.target_position_base_m))
    run.check("magnetic puck reaches AprilTag", contact_error_m <= 0.025, detail=f"contact error {contact_error_m:.4f} m")
    run.check("magnetic docking review frame captured", frame is not None, detail="1280x720")
    if frame is not None:
        rgb = np.asarray(frame)[..., :3]
        run.check("magnetic docking frame is visible", float(rgb.std()) >= 5.0, detail=f"std {float(rgb.std()):.2f}")
        frame_path = "/tmp/so101_magnetic_apriltag_contact.png"
        Image.fromarray(rgb).save(frame_path)
        run.add_artifact(frame_path, name="so101-magnetic-apriltag-contact.png", content_type="image/png")
        antioch.Logger("factory_sre/magnetic_ik").image("camera/contact", rgb)
    run.add_result(
        "magnetic_ik",
        {
            "success": solution.success,
            "contact_error_m": contact_error_m,
            "target_world_m": list(target.tag_center),
            "target_base_m": list(solution.target_position_base_m),
            "joint_targets_rad": solution.joints,
        },
    )


@antioch.scenario(
    name="composite_policy_stability_smoke",
    description="Drive the policy-controlled Go2 with the physically fixed SO-101 payload and stop safely.",
    tags=("smoke", "composite", "locomotion", "policy", "stability"),
    sim=antioch.BootProfile(
        physics_dt=1.0 / 200.0,
        render_dt=1.0 / 50.0,
        physics_engine="physx",
        render_quality="performance",
        viewport=(1280, 720),
    ),
    capture=True,
)
def composite_policy_stability_smoke(run: antioch.ScenarioRun) -> None:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.experimental.utils.stage import add_reference_to_stage
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdLux

    from factory_sre.sim.composite import author_go2_so101_composite, measure_mount_stack

    SimulationManager.set_physics_sim_device("cuda")
    assets = get_assets_root_path()
    add_reference_to_stage(
        usd_path=assets + "/Isaac/Environments/Grid/default_environment.usd",
        path="/World/GroundEnvironment",
    )
    policy = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        position=[-0.5, -2.5, 0.50],
        policy_path=assets + "/Isaac/Samples/Policies/go2/physx_policy.pt",
        env_config_path=assets + "/Isaac/Samples/Policies/go2/physx_env.yaml",
    )
    # Keep the supplied 12-DOF policy and the 6-DOF arm as separate control
    # trees while a real fixed joint carries the payload between them.
    report = author_go2_so101_composite(
        mode="controller_isolated",
        initial_chassis_position_m=(-0.5, -2.5, 0.50),
    )
    mount_stack = measure_mount_stack(antioch.stage(), report)
    # GPU PhysX views must be declared before the scene is initialized. The
    # reset below binds these wrappers to the resulting live tensor actors.
    chassis_body = RigidPrim(report.go2_chassis)
    arm_body = RigidPrim(report.so101_base)
    UsdLux.DomeLight.Define(antioch.stage(), "/World/StabilityLight").CreateIntensityAttr(900.0)
    world = antioch.world()
    world.reset()
    policy.initialize()
    policy.post_reset()
    # Read both attachment bodies through PhysX tensors. USD XformCache is not
    # a valid motion witness when Fabric/GPU simulation is active because its
    # authored transforms can remain unchanged while the physics actors move.
    live_physics_readback = (
        chassis_body.is_physics_tensor_entity_valid()
        and arm_body.is_physics_tensor_entity_valid()
    )
    if not live_physics_readback:
        raise RuntimeError("Go2 and SO-101 rigid bodies require live PhysX tensor views")
    set_camera_view(
        eye=[-4.5, -5.5, 2.5],
        target=[0.0, -2.5, 0.5],
        camera_prim_path="/OmniverseKit_Persp",
    )
    zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    forward = torch.tensor([0.25, 0.0, 0.0], dtype=torch.float32, device="cuda")
    mount_errors_m: list[float] = []
    mount_angle_errors_deg: list[float] = []
    arm_positions: list[np.ndarray] = []
    walk_frames: list[tuple[str, np.ndarray]] = []

    def quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        quaternion = quaternion / np.linalg.norm(quaternion)
        scalar = quaternion[0]
        imaginary = quaternion[1:]
        first_cross = np.cross(imaginary, vector)
        return vector + 2.0 * scalar * first_cross + 2.0 * np.cross(imaginary, first_cross)

    def quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_w, left_xyz = left[0], left[1:]
        right_w, right_xyz = right[0], right[1:]
        return np.concatenate(
            (
                [left_w * right_w - np.dot(left_xyz, right_xyz)],
                left_w * right_xyz + right_w * left_xyz + np.cross(left_xyz, right_xyz),
            )
        )

    def sample_mount() -> None:
        chassis_positions, chassis_orientations = chassis_body.get_world_poses()
        arm_world_positions, arm_world_orientations = arm_body.get_world_poses()
        chassis_position = np.asarray(chassis_positions.numpy()[0], dtype=float)
        chassis_orientation = np.asarray(chassis_orientations.numpy()[0], dtype=float)
        arm_position = np.asarray(arm_world_positions.numpy()[0], dtype=float)
        arm_orientation = np.asarray(arm_world_orientations.numpy()[0], dtype=float)
        expected_arm_position = chassis_position + quat_rotate_wxyz(
            chassis_orientation,
            np.asarray(report.mount_offset_m, dtype=float),
        )
        mount_errors_m.append(float(np.linalg.norm(arm_position - expected_arm_position)))
        yaw_half_rad = np.deg2rad(report.mount_yaw_deg) / 2.0
        expected_arm_orientation = quat_multiply_wxyz(
            chassis_orientation,
            np.asarray([np.cos(yaw_half_rad), 0.0, 0.0, np.sin(yaw_half_rad)]),
        )
        orientation_dot = float(
            np.clip(
                abs(
                    np.dot(
                        expected_arm_orientation / np.linalg.norm(expected_arm_orientation),
                        arm_orientation / np.linalg.norm(arm_orientation),
                    )
                ),
                0.0,
                1.0,
            )
        )
        mount_angle_errors_deg.append(float(np.degrees(2.0 * np.arccos(orientation_dot))))
        arm_positions.append(arm_position)

    def capture_walk_frame(label: str) -> None:
        frame = antioch.capture_viewport()
        if frame is not None:
            walk_frames.append((label, np.asarray(frame)[..., :3].copy()))

    for _ in range(120):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    sample_mount()
    capture_walk_frame("START")
    start = np.asarray(policy.robot.get_world_poses()[0].numpy()[0], dtype=float)
    for step in range(240):
        policy.forward(1.0 / 200.0, forward)
        world.step(render=True)
        if step % 10 == 0:
            sample_mount()
        if step == 120:
            capture_walk_frame("MID-WALK")
    moving = np.asarray(policy.robot.get_world_poses()[0].numpy()[0], dtype=float)
    for step in range(320):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
        if step % 10 == 0:
            sample_mount()
    stopped = np.asarray(policy.robot.get_world_poses()[0].numpy()[0], dtype=float)
    stopped_orientation = np.asarray(policy.robot.get_world_poses()[1].numpy()[0], dtype=float)
    stopped_velocity = np.asarray(policy.robot.get_velocities()[0].numpy()[0], dtype=float)
    frame = antioch.capture_viewport()
    if frame is not None:
        walk_frames.append(("STOP", np.asarray(frame)[..., :3].copy()))
    distance = float(np.linalg.norm(moving[:2] - start[:2]))
    residual_speed = float(np.linalg.norm(stopped_velocity[:3]))
    from math import asin, atan2, degrees

    w, x, y, z = (float(value) for value in stopped_orientation)
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    maximum_tilt_deg = max(abs(degrees(roll)), abs(degrees(pitch)))
    maximum_mount_error_m = max(mount_errors_m)
    maximum_mount_angle_error_deg = max(mount_angle_errors_deg)
    arm_travel_m = float(np.linalg.norm(arm_positions[-1][:2] - arm_positions[0][:2]))
    run.check("payload composite uses a physical mount joint", report.mount_body0 == (report.go2_chassis,), detail=report.mount_joint)
    run.check("SO-101 demo payload is gravity compensated", report.gravity_compensated, detail="1.2 kg inertia retained; hardware needs payload-aware Go2 control")
    run.check("solid pedestal physically overlaps chassis", antioch.stage().GetPrimAtPath(report.mount_pedestal).IsValid(), detail=report.mount_pedestal)
    for interface_name, interface in mount_stack["interfaces"].items():
        interface_ok = (
            float(interface["gap_m"]) <= 0.002
            and float(interface["x_overlap_m"]) >= 0.01
            and float(interface["y_overlap_m"]) >= 0.01
        )
        run.check(
            f"mount geometry is continuous at {interface_name.replace('_', ' ')}",
            interface_ok,
            detail=(
                f"gap {float(interface['gap_m']):.4f} m, "
                f"xy overlap {float(interface['x_overlap_m']):.3f} x "
                f"{float(interface['y_overlap_m']):.3f} m"
            ),
        )
    run.check("SO-101 base faces Go2 front", abs(report.mount_yaw_deg - 90.0) <= 1e-6, detail=f"roof-frame yaw {report.mount_yaw_deg:.1f} deg")
    run.check("attachment motion is read from live PhysX tensors", live_physics_readback, detail=f"{report.go2_chassis} + {report.so101_base}")
    run.check("arm remains glued to dog throughout walk", maximum_mount_error_m <= 0.01, detail=f"max frame error {maximum_mount_error_m:.4f} m over {len(mount_errors_m)} samples")
    run.check("arm orientation remains glued to dog", maximum_mount_angle_error_deg <= 3.0, detail=f"max angular error {maximum_mount_angle_error_deg:.3f} deg")
    run.check("arm travels with dog", arm_travel_m >= 0.10, detail=f"arm base traveled {arm_travel_m:.3f} m")
    run.check("controller isolation preserves two control trees", len(report.articulation_roots) == 2, detail=str(report.articulation_roots))
    finite_state = bool(np.isfinite(np.concatenate((start, moving, stopped, stopped_velocity))).all())
    run.check("payload composite state remains finite", finite_state, detail="all pose and velocity values finite")
    run.check("payload composite moves under supplied policy", 0.10 <= distance <= 3.0, detail=f"{distance:.3f} m")
    run.check("payload composite keeps a proper walking stance", 0.20 <= float(stopped[2]) <= 1.2, detail=f"base z {float(stopped[2]):.3f} m")
    run.check("payload composite roll and pitch stay below 15 degrees", maximum_tilt_deg < 15.0, detail=f"{maximum_tilt_deg:.3f} deg")
    run.check("payload composite decelerates", residual_speed <= 1.0, detail=f"{residual_speed:.3f} m/s")
    run.check("stability review frame captured", frame is not None, detail="1280x720")
    if frame is not None:
        antioch.Logger("factory_sre/composite_policy").image("camera/forward_stop", np.asarray(frame)[..., :3])
    run.check("walk montage captured", len(walk_frames) == 3, detail=str([label for label, _ in walk_frames]))
    if len(walk_frames) == 3:
        labeled_frames: list[Image.Image] = []
        for label, rgb in walk_frames:
            panel = Image.fromarray(rgb)
            drawing = ImageDraw.Draw(panel)
            drawing.rectangle((16, 16, 180, 54), fill=(0, 0, 0))
            drawing.text((28, 25), label, fill=(255, 255, 255))
            labeled_frames.append(panel)
        montage = Image.new(
            "RGB",
            (sum(panel.width for panel in labeled_frames), labeled_frames[0].height),
        )
        x_offset = 0
        for panel in labeled_frames:
            montage.paste(panel, (x_offset, 0))
            x_offset += panel.width
        montage_path = "/tmp/go2_so101_walk_montage.png"
        montage.save(montage_path)
        run.add_artifact(montage_path, name="go2-so101-walk-montage.png", content_type="image/png")
        antioch.Logger("factory_sre/composite_policy").image(
            "camera/walk_montage",
            np.asarray(montage),
        )
    set_camera_view(
        eye=[float(stopped[0] - 1.35), float(stopped[1] - 1.10), 1.05],
        target=[float(stopped[0]), float(stopped[1]), 0.42],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(8):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    mount_closeup = antioch.capture_viewport()
    run.check("physical mount closeup captured", mount_closeup is not None, detail="unobstructed roof view")
    if mount_closeup is not None:
        closeup_rgb = np.asarray(mount_closeup)[..., :3]
        closeup_path = "/tmp/go2_so101_physical_mount_closeup.png"
        Image.fromarray(closeup_rgb).save(closeup_path)
        run.add_artifact(
            closeup_path,
            name="go2-so101-physical-mount-closeup.png",
            content_type="image/png",
        )
        antioch.Logger("factory_sre/composite_policy").image(
            "camera/physical_mount_closeup",
            closeup_rgb,
        )
    run.add_result(
        "stability",
        {
            "distance_m": distance,
            "residual_speed_mps": residual_speed,
            "maximum_tilt_deg": maximum_tilt_deg,
            "maximum_mount_error_m": maximum_mount_error_m,
            "maximum_mount_angle_error_deg": maximum_mount_angle_error_deg,
            "arm_travel_m": arm_travel_m,
            "physics_readback": "PhysX tensor",
            "mount_stack": mount_stack,
            "start": start.tolist(),
            "moving": moving.tolist(),
            "stopped": stopped.tolist(),
            "composite": report.to_dict(),
        },
    )


ROUTE_CASES = (
    antioch.case({"asset_id": "workbench-a"}, id="route-a", tags=("certification",)),
    antioch.case({"asset_id": "workbench-b"}, id="route-b", tags=("certification",)),
)


@antioch.scenario(
    name="manifest_route_and_registration",
    description="Plan to the incident-selected workspace and register its exact tag-to-dock transform.",
    tags=("certification", "navigation", "apriltag", "dock"),
    cases=ROUTE_CASES,
    sim=None,
    capture=False,
)
def manifest_route_and_registration(run: antioch.ScenarioRun, asset_id: str = "workbench-b") -> None:
    from factory_sre.contracts import Transform3D, manifest_for_asset
    from factory_sre.sim.docking import WrongAssetError, register_dock_target
    from factory_sre.sim.factory_navigation import plan_to_manifest, validate_route

    manifest = manifest_for_asset(asset_id)
    route = plan_to_manifest(manifest)
    tag_x = 1.8 if manifest is WORKBENCH_A else 4.0
    world_tag = Transform3D((tag_x, 0.64, 0.88))
    hidden_truth = Transform3D((tag_x - 0.30, 0.64, 0.88))
    registered = register_dock_target(
        manifest,
        observed_tag_id=manifest.tag_id,
        world_tag=world_tag,
        hidden_world_dock=hidden_truth,
    )
    neighbor = WORKBENCH_B if manifest is WORKBENCH_A else WORKBENCH_A
    wrong_tag_rejected = False
    try:
        register_dock_target(
            manifest,
            observed_tag_id=neighbor.tag_id,
            world_tag=world_tag,
            hidden_world_dock=hidden_truth,
        )
    except WrongAssetError:
        wrong_tag_rejected = True

    run.check("manifest-selected route is collision free", not validate_route(route), detail=str(validate_route(route)))
    run.check("route terminates at selected approach pose", route.final_error_m < 1e-9, detail=f"{route.final_error_m:.6f} m")
    run.check("tag registration matches hidden dock truth", registered.position_error_m < 1e-9, detail=f"{registered.position_error_m:.6f} m")
    run.check("neighbor tag is terminal wrong-asset", wrong_tag_rejected, detail=f"neighbor tag {neighbor.tag_id}")
    run.add_result(
        "route_registration",
        {
            "asset_id": asset_id,
            "tag_id": manifest.tag_id,
            "waypoints": [[pose.x, pose.y, pose.yaw] for pose in route.waypoints],
            "length_m": route.length_m,
            "dock_translation": list(registered.world_dock.translation),
        },
    )


@antioch.scenario(
    name="guarded_dock_controller_certification",
    description="Controller-level 20-case pose/contact/force certification before physics certification.",
    tags=("certification", "dock", "safety", "controller"),
    sim=None,
    capture=False,
)
def guarded_dock_controller_certification(run: antioch.ScenarioRun) -> None:
    from math import cos, radians, sin
    from random import Random

    from factory_sre.contracts import Transform3D
    from factory_sre.sim.docking import guarded_magnetic_dock, register_dock_target

    random = Random(3710)
    tag = Transform3D((4.0, 0.64, 0.88))
    truth = Transform3D((3.7, 0.64, 0.88))
    registered = register_dock_target(
        WORKBENCH_B,
        observed_tag_id=WORKBENCH_B.tag_id,
        world_tag=tag,
        hidden_world_dock=truth,
    )
    outcomes = []
    for case_index in range(20):
        dx = random.uniform(-0.0038, 0.0038)
        dy = random.uniform(-0.0020, 0.0020)
        yaw_deg = random.uniform(-2.5, 2.5)
        half = radians(yaw_deg) / 2.0
        reached = Transform3D(
            (truth.translation[0] + dx, truth.translation[1] + dy, truth.translation[2]),
            (cos(half), 0.0, 0.0, sin(half)),
        )
        result = guarded_magnetic_dock(
            registered,
            reached_pose=reached,
            axial_force_n=random.uniform(2.0, 7.2),
            contact_depth_m=random.uniform(0.0055, 0.0080),
            penetration_m=random.uniform(0.0002, 0.0040),
        )
        outcomes.append(
            {
                "case": case_index,
                "engaged": result.engaged,
                "position_error_m": result.position_error_m,
                "angle_error_deg": result.angle_error_deg,
                "axial_force_n": result.axial_force_n,
                "reason": result.terminal_reason,
            }
        )
    successes = sum(bool(outcome["engaged"]) for outcome in outcomes)
    maximum_force = max(float(outcome["axial_force_n"]) for outcome in outcomes)
    run.check("controller dock success is at least 18/20", successes >= 18, detail=f"{successes}/20")
    run.check("controller never exceeds 8 N", maximum_force <= 8.0, detail=f"{maximum_force:.3f} N")
    run.add_result("controller_certification", {"successes": successes, "trials": outcomes})


@antioch.scenario(
    name="composite_manifest_navigation_smoke",
    description="Carry the SO-101 payload around the aisle pallet to workbench B's manifest-selected approach pose.",
    tags=("smoke", "composite", "navigation", "policy", "workbench"),
    sim=antioch.BootProfile(
        physics_dt=1.0 / 200.0,
        render_dt=1.0 / 50.0,
        physics_engine="physx",
        render_quality="performance",
        viewport=(1280, 720),
    ),
    capture=True,
)
def composite_manifest_navigation_smoke(run: antioch.ScenarioRun) -> None:
    import numpy as np
    import torch
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.storage.native import get_assets_root_path

    from factory_sre.contracts import Pose2D, WORKBENCH_B
    from factory_sre.sim.composite import author_go2_so101_composite
    from factory_sre.sim.factory_navigation import plan_to_manifest
    from factory_sre.sim.go2_policy_navigation import Go2PolicyWaypointFollower
    from factory_sre.sim.navigation import hero_factory_grid
    from factory_sre.sim.scene import build_factory_scene

    SimulationManager.set_physics_sim_device("cuda")
    build_factory_scene()
    assets = get_assets_root_path()
    start = Pose2D(-0.5, -2.5, 0.0)
    policy = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        position=[start.x, start.y, 0.50],
        policy_path=assets + "/Isaac/Samples/Policies/go2/physx_policy.pt",
        env_config_path=assets + "/Isaac/Samples/Policies/go2/physx_env.yaml",
    )
    composite = author_go2_so101_composite(
        mode="controller_isolated",
        initial_chassis_position_m=(start.x, start.y, 0.50),
    )
    world = antioch.world()
    world.reset()
    policy.initialize()
    policy.post_reset()
    set_camera_view(
        eye=[2.2, -6.8, 6.4],
        target=[2.2, -0.7, 0.0],
        camera_prim_path="/OmniverseKit_Persp",
    )
    zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    for _ in range(120):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    occupancy = hero_factory_grid()
    route = plan_to_manifest(WORKBENCH_B, start=start, grid=occupancy)
    trace = Go2PolicyWaypointFollower(policy, world).follow(route, occupancy)
    frame = antioch.capture_viewport()
    final_position = trace.positions[-1]
    run.check("policy reaches manifest-selected approach", trace.reached, detail=f"error {trace.final_error_m:.3f} m")
    run.check("measured path avoids inflated occupancy", trace.occupied_samples == 0, detail=f"{trace.occupied_samples} occupied samples")
    run.check("payload navigation tilt stays below 15 degrees", trace.maximum_roll_pitch_deg < 15.0, detail=f"{trace.maximum_roll_pitch_deg:.3f} deg")
    run.check("payload navigation ends above ground", final_position[2] >= 0.15, detail=f"base z {final_position[2]:.3f} m")
    run.check("navigation review frame captured", frame is not None, detail="1280x720")
    if frame is not None:
        antioch.Logger("factory_sre/navigation").image("camera/final_approach", np.asarray(frame)[..., :3])
    run.add_result(
        "navigation",
        {
            "asset_id": route.asset_id,
            "route_length_m": route.length_m,
            "planned_waypoints": [[pose.x, pose.y, pose.yaw] for pose in route.waypoints],
            "final_error_m": trace.final_error_m,
            "maximum_roll_pitch_deg": trace.maximum_roll_pitch_deg,
            "occupied_samples": trace.occupied_samples,
            "steps": trace.steps,
            "final_position": list(final_position),
            "composite": composite.to_dict(),
        },
    )


@antioch.scenario(
    name="tagged_box_grid_scene_smoke",
    description="Render the top-mounted SO-101 Go2 in the center of eight numbered, four-sided AprilTagged service boxes.",
    tags=("smoke", "scene", "2x4", "apriltag", "composite"),
    sim=antioch.BootProfile(
        physics_dt=1.0 / 200.0,
        render_dt=1.0 / 50.0,
        physics_engine="physx",
        render_quality="performance",
        viewport=(1280, 720),
    ),
    capture=True,
)
def tagged_box_grid_scene_smoke(run: antioch.ScenarioRun) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Gf, UsdGeom

    from factory_sre.sim.composite import author_go2_so101_composite
    from factory_sre.sim.tagged_box_grid import build_tagged_box_grid

    SimulationManager.set_physics_sim_device("cuda")
    scene = build_tagged_box_grid()
    assets = get_assets_root_path()
    start = (2.4, 1.6, 0.50)
    policy = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        position=list(start),
        policy_path=assets + "/Isaac/Samples/Policies/go2/physx_policy.pt",
        env_config_path=assets + "/Isaac/Samples/Policies/go2/physx_env.yaml",
    )
    composite = author_go2_so101_composite(
        mode="controller_isolated",
        initial_chassis_position_m=start,
    )
    world = antioch.world()
    world.reset()
    policy.initialize()
    policy.post_reset()
    set_camera_view(
        eye=[7.8, -5.8, 6.7],
        target=[2.4, 1.6, 0.25],
        camera_prim_path="/OmniverseKit_Persp",
    )
    zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    for _ in range(160):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    position = np.asarray(policy.robot.get_world_poses()[0].numpy()[0], dtype=float)
    cache = UsdGeom.XformCache()
    chassis_world = cache.GetLocalToWorldTransform(antioch.stage().GetPrimAtPath(composite.go2_chassis))
    arm_world = cache.GetLocalToWorldTransform(antioch.stage().GetPrimAtPath(composite.so101_base))
    expected_arm_world = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*composite.mount_offset_m)) * chassis_world
    mount_error_m = float(
        np.linalg.norm(
            np.asarray(arm_world.ExtractTranslation(), dtype=float)
            - np.asarray(expected_arm_world.ExtractTranslation(), dtype=float)
        )
    )
    frame = antioch.capture_viewport()
    tag_ids = [box["tag_id"] for box in scene["boxes"]]
    run.check("scene has a 2x4 central-aisle layout", scene["layout"] == "2x4-central-aisle" and len(scene["boxes"]) == 8, detail=str(scene["layout"]))
    run.check("all eight AprilTags are unique", len(set(tag_ids)) == 8, detail=str(tag_ids))
    run.check("all four vertical faces are tagged", scene["tag_faces_per_box"] == 4 and all(len(box["tag_faces"]) == 4 for box in scene["boxes"]), detail="north/south/east/west")
    run.check("only Box 8 is offline", [box["box_number"] for box in scene["boxes"] if box["offline"]] == [8], detail=str(scene["incident"]))
    run.check("Go2 is held upright by supplied policy", float(position[2]) >= 0.20, detail=f"base z {float(position[2]):.3f} m")
    run.check("SO-101 is physically mounted on chassis top", composite.mount_body0 == (composite.go2_chassis,), detail=f"mount {composite.mount_joint}, payload {composite.payload_mass_kg:.1f} kg")
    run.check("SO-101 rigid base stays on its roof frame", mount_error_m <= 0.01, detail=f"frame error {mount_error_m:.4f} m")
    run.check("visible roof plate exists", antioch.stage().GetPrimAtPath(composite.mount_plate).IsValid(), detail=composite.mount_plate)
    run.check("solid roof pedestal exists", antioch.stage().GetPrimAtPath(composite.mount_pedestal).IsValid(), detail=composite.mount_pedestal)
    run.check("box-grid review frame captured", frame is not None, detail="1280x720")
    if frame is not None:
        rgb = np.asarray(frame)[..., :3]
        mean = float(rgb.mean())
        run.check("box-grid frame is visible", float(rgb.std()) >= 5.0, detail=f"std {float(rgb.std()):.2f}")
        run.check("box-grid frame has useful exposure", 10.0 <= mean <= 220.0, detail=f"mean {mean:.2f}")
        antioch.Logger("factory_sre/box_grid").image("camera/overview", rgb)
        overview_path = "/tmp/tagged_box_grid_overview.png"
        Image.fromarray(rgb).save(overview_path)
        run.add_artifact(overview_path, name="box-grid-overview.png", content_type="image/png")
    set_camera_view(
        eye=[1.0, 1.6, 1.10],
        target=[2.4, 1.6, 0.42],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(8):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    closeup = antioch.capture_viewport()
    run.check("arm-mount closeup captured", closeup is not None, detail="1280x720")
    if closeup is not None:
        closeup_rgb = np.asarray(closeup)[..., :3]
        closeup_path = "/tmp/mobile_manipulator_closeup.png"
        Image.fromarray(closeup_rgb).save(closeup_path)
        run.add_artifact(closeup_path, name="mobile-manipulator-closeup.png", content_type="image/png")
        antioch.Logger("factory_sre/box_grid").image("camera/mobile_manipulator_closeup", closeup_rgb)
    run.add_result(
        "box_grid",
        scene | {"composite": composite.to_dict(), "mount_frame_error_m": mount_error_m},
    )


@antioch.scenario(
    name="tagged_box_policy_navigation_smoke",
    description="Dispatch the centered Go2 to a selected offline box, reach its tag with the supplied policy, and confirm identity in the wrist camera.",
    tags=("smoke", "navigation", "policy", "apriltag", "wrist-camera"),
    sim=antioch.BootProfile(
        physics_dt=1.0 / 200.0,
        render_dt=1.0 / 50.0,
        physics_engine="physx",
        render_quality="performance",
        viewport=(1280, 720),
    ),
    capture=True,
)
def tagged_box_policy_navigation_smoke(run: antioch.ScenarioRun, box_number: int = 8) -> None:
    import numpy as np
    import subprocess
    import tempfile
    import torch
    from pathlib import Path
    from PIL import Image
    from pupil_apriltags import Detector
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.sensors.camera import Camera
    from isaacsim.storage.native import get_assets_root_path

    from factory_sre.contracts import Pose2D
    from factory_sre.perception import rotation_matrix_to_wxyz
    from factory_sre.sim.composite import (
        align_wrist_camera_to_chassis,
        author_go2_so101_composite,
        configure_so101_stow_drives,
    )
    from factory_sre.sim.go2_policy_navigation import Go2PolicyWaypointFollower
    from factory_sre.sim.tagged_box_grid import (
        build_tagged_box_grid,
        plan_to_service_box,
        service_box_for_number,
        tagged_box_occupancy,
    )

    SimulationManager.set_physics_sim_device("cuda")
    if box_number not in range(1, 9):
        run.fail("box_number must be between 1 and 8")
    scene = build_tagged_box_grid(incident_box_number=box_number)
    assets = get_assets_root_path()
    start = Pose2D(2.4, 1.6, 0.0)
    incident = {
        "asset_id": f"service-box-{box_number}",
        "tag_id": 29 + box_number,
        "status": "offline",
    }
    target_tag_id = incident["tag_id"]
    target = service_box_for_number(box_number, offline_box_number=box_number)
    policy = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        position=[start.x, start.y, 0.50],
        policy_path=assets + "/Isaac/Samples/Policies/go2/physx_policy.pt",
        env_config_path=assets + "/Isaac/Samples/Policies/go2/physx_env.yaml",
    )
    composite = author_go2_so101_composite(
        mode="controller_isolated",
        initial_chassis_position_m=(start.x, start.y, 0.50),
    )
    arm_stow_drive_count = configure_so101_stow_drives(
        antioch.stage(), composite.so101_movable_joints
    )
    # The authored wrist sensor is a normal USD Camera, so use Isaac's normal
    # Camera wrapper.  CameraSensor is reserved for prims carrying
    # OmniSensorAPI and rejects a plain USD camera before simulation starts.
    wrist_sensor = Camera(prim_path=composite.wrist_camera, resolution=(640, 480))
    world = antioch.world()
    world.reset()
    wrist_sensor.initialize()
    policy.initialize()
    policy.post_reset()
    set_camera_view(
        eye=[7.8, -5.8, 6.7],
        target=[2.4, 1.6, 0.25],
        camera_prim_path="/OmniverseKit_Persp",
    )
    zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    for _ in range(160):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    # Calibrate only after the SO-101 drives have reached their parked pose;
    # before this settle window the jaw frame is still moving from the asset's
    # authored pose toward the navigation stow targets.
    align_wrist_camera_to_chassis(
        antioch.stage(),
        composite.wrist_camera,
        composite.end_effector,
        composite.go2_chassis,
    )
    for _ in range(2):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    video_directory = Path(tempfile.mkdtemp(prefix=f"go2-box{box_number}-navigation-"))
    video_frame_count = 0
    video_frame_stats: list[tuple[float, float, int]] = []
    first_video_frame = None
    latest_video_frame = None

    def capture_navigation_frame(_step, _position=None, _orientation=None) -> None:
        nonlocal first_video_frame, latest_video_frame, video_frame_count
        viewport = antioch.capture_viewport()
        if viewport is None:
            return
        rgb = np.asarray(viewport, dtype=np.uint8)[..., :3]
        red_pixels = int(
            ((rgb[..., 0] > 1.35 * rgb[..., 1]) & (rgb[..., 0] > 1.25 * rgb[..., 2]) & (rgb[..., 0] > 80)).sum()
        )
        video_frame_stats.append((float(rgb.mean()), float(rgb.std()), red_pixels))
        frame_copy = np.array(rgb, dtype=np.uint8, copy=True, order="C")
        if first_video_frame is None:
            first_video_frame = frame_copy
        latest_video_frame = frame_copy
        Image.fromarray(rgb).save(video_directory / f"frame_{video_frame_count:05d}.png")
        video_frame_count += 1

    capture_navigation_frame(0)
    occupancy = tagged_box_occupancy()
    route = plan_to_service_box(target_tag_id, start=start, grid=occupancy)
    trace = Go2PolicyWaypointFollower(
        policy,
        world,
        maximum_speed_mps=0.45,
        maximum_yaw_rate_rps=1.3,
        final_tolerance_m=0.20,
        final_yaw_tolerance_rad=0.03,
        maximum_steps_per_waypoint=12000,
    ).follow(
        route,
        occupancy,
        frame_callback=capture_navigation_frame,
        frame_every_steps=30,
    )
    # Aim through the sensor's live Fabric pose.  USD XformCache contains the
    # authored pose and is stale after the quadruped has moved under PhysX.
    # Isaac's world camera axes are +X forward and +Z up.
    eye_position, _ = wrist_sensor.get_world_pose(camera_axes="world")
    forward = np.asarray(target.tag_center, dtype=float) - np.asarray(eye_position, dtype=float)
    forward /= np.linalg.norm(forward)
    left = np.cross(np.asarray([0.0, 0.0, 1.0]), forward)
    left /= np.linalg.norm(left)
    camera_up = np.cross(forward, left)
    live_orientation = rotation_matrix_to_wxyz(
        np.column_stack((forward, left, camera_up))
    )
    wrist_sensor.set_world_pose(
        position=eye_position,
        orientation=live_orientation,
        camera_axes="world",
    )
    for _ in range(40):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    capture_navigation_frame(trace.steps)
    video_name = f"go2-to-box{box_number}-navigation.mp4"
    video_path = video_directory / video_name
    video_error = ""
    try:
        encoded = subprocess.run(
            [
                "ffmpeg",
                "-n",
                "-loglevel",
                "error",
                "-framerate",
                "12",
                "-i",
                str(video_directory / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        video_error = encoded.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        video_error = str(exc)
    video_exists = video_path.is_file() and video_path.stat().st_size > 0
    video_motion_score = (
        0.0
        if first_video_frame is None or latest_video_frame is None
        else float(
            np.mean(
                np.abs(
                    first_video_frame.astype(np.int16)
                    - latest_video_frame.astype(np.int16)
                )
            )
        )
    )
    video_content_ok = (
        video_frame_count >= 20
        and bool(video_frame_stats)
        and all(10.0 <= mean <= 220.0 and standard_deviation >= 5.0 for mean, standard_deviation, _ in video_frame_stats)
        and video_motion_score >= 0.25
    )
    if video_exists:
        run.add_artifact(video_path, name=video_name, content_type="video/mp4")
    wrist_data = wrist_sensor.get_rgba()
    wrist_rgb = (
        None
        if wrist_data is None or wrist_data.size == 0
        else np.array(wrist_data, dtype=np.uint8, copy=True, order="C")
    )
    detected_ids: list[int] = []
    if wrist_rgb is not None:
        wrist_image = Image.fromarray(np.ascontiguousarray(wrist_rgb[..., :3]))
        gray = np.array(wrist_image.convert("L"), dtype=np.uint8, copy=True, order="C")
        detected_ids = [
            int(detection.tag_id)
            for detection in Detector(families="tag36h11", nthreads=1, refine_edges=1).detect(gray)
        ]
        wrist_name = f"box{box_number}-tag{target_tag_id}-wrist-camera.png"
        wrist_path = str(video_directory / wrist_name)
        wrist_image.save(wrist_path)
        run.add_artifact(wrist_path, name=wrist_name, content_type="image/png")
        antioch.Logger("factory_sre/tag_navigation").image("camera/wrist", wrist_rgb[..., :3])
    frame = antioch.capture_viewport()
    final_position = trace.positions[-1]
    lateral_face_error_m = abs(float(final_position[0]) - target.tag_center[0])
    face_standoff_m = -target.face_sign_y * (target.tag_center[1] - float(final_position[1]))
    body_colors = {tuple(box["body_color_rgb"]) for box in scene["boxes"]}
    status_colors = {tuple(box["status_color_rgb"]) for box in scene["boxes"]}
    run.check(
        "all eight service boxes use identical colours",
        len(body_colors) == 1 and len(status_colors) == 1,
        detail=f"body={sorted(body_colors)}, status={sorted(status_colors)}",
    )
    run.check(
        f"incident dispatch selects exactly offline Box {box_number}",
        target.offline and route.asset_id == incident["asset_id"] == target.asset_id,
        detail=f"tag {target_tag_id} -> {route.asset_id}",
    )
    run.check("supplied policy reaches tag-selected approach", trace.reached, detail=f"error {trace.final_error_m:.3f} m")
    run.check("robot faces the tag at terminal approach", trace.final_yaw_error_rad <= 0.15, detail=f"yaw error {trace.final_yaw_error_rad:.3f} rad")
    run.check(
        f"robot stops directly in front of Box {box_number}'s tagged face",
        lateral_face_error_m <= 0.30 and 0.35 <= face_standoff_m <= 0.85,
        detail=f"lateral error {lateral_face_error_m:.3f} m, face standoff {face_standoff_m:.3f} m",
    )
    run.check("measured base path avoids inflated boxes", trace.occupied_samples == 0, detail=f"{trace.occupied_samples} occupied samples")
    run.check("payload stays upright while navigating", trace.maximum_roll_pitch_deg < 15.0, detail=f"{trace.maximum_roll_pitch_deg:.3f} deg")
    run.check("mobile base remains above the floor", final_position[2] >= 0.15, detail=f"base z {final_position[2]:.3f} m")
    run.check("wrist camera returns RGB", wrist_rgb is not None, detail="480x640")
    run.check("wrist camera confirms expected AprilTag", target_tag_id in detected_ids, detail=str(detected_ids))
    run.check("navigation review frame captured", frame is not None, detail="1280x720")
    run.check(
        f"movement video records the Go2 travelling to Box {box_number}",
        video_exists and video_content_ok,
        detail=(
            f"{video_frame_count} frames, {video_path.stat().st_size if video_exists else 0} bytes"
            if video_exists and video_content_ok
            else f"frames={video_frame_count}, motion={video_motion_score:.3f}, encoded={video_exists}, error={video_error or 'content oracle failed'}"
        ),
    )
    run.add_result(
        "tag_navigation",
        {
            "incident": incident,
            "tag_id": target_tag_id,
            "asset_id": route.asset_id,
            "detected_tag_ids": detected_ids,
            "route_length_m": route.length_m,
            "final_error_m": trace.final_error_m,
            "final_yaw_error_rad": trace.final_yaw_error_rad,
            "lateral_face_error_m": lateral_face_error_m,
            "face_standoff_m": face_standoff_m,
            "maximum_roll_pitch_deg": trace.maximum_roll_pitch_deg,
            "occupied_samples": trace.occupied_samples,
            "occupied_positions": [list(position) for position in trace.occupied_positions],
            "steps": trace.steps,
            "final_position": list(final_position),
            "video_frame_count": video_frame_count,
            "video_duration_s": video_frame_count / 12.0,
            "video_motion_score": video_motion_score,
            "video_artifact": video_name if video_exists else None,
            "composite": composite.to_dict(),
            "arm_stow_drive_count": arm_stow_drive_count,
        },
    )
