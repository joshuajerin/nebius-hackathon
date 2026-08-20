"""Dedicated first-person video evidence for the original 2x4 box world.

This intentionally does not use the warehouse scene.  The recorded pixels are
read directly from the SO-101's authored wrist camera while the supplied Go2
flat-terrain policy drives to the selected service box.
"""

from __future__ import annotations

import antioch


FPV_PROFILE = antioch.BootProfile(
    physics_dt=1.0 / 200.0,
    render_dt=1.0 / 50.0,
    physics_engine="physx",
    render_quality="performance",
    viewport=(1280, 720),
)


@antioch.scenario(
    name="simple_boxes_wrist_fpv_video",
    description="Record the SO-101 wrist camera while Go2 travels through the original 2x4 tagged-box world to Box 8.",
    tags=("demo", "video", "fpv", "wrist-camera", "navigation", "simple-boxes"),
    sim=FPV_PROFILE,
    capture=False,
)
def simple_boxes_wrist_fpv_video(run: antioch.ScenarioRun) -> None:
    """Export an H.264 FPV artifact from actual sensor readback."""
    import subprocess
    import tempfile
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.sensors.camera import Camera
    from isaacsim.storage.native import get_assets_root_path

    from factory_sre.contracts import Pose2D
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

    box_number = 8
    SimulationManager.set_physics_sim_device("cuda")
    scene = build_tagged_box_grid(incident_box_number=box_number)
    start = Pose2D(2.4, 1.6, 0.0)
    target = service_box_for_number(box_number, offline_box_number=box_number)
    assets = get_assets_root_path()
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
    stowed_joint_count = configure_so101_stow_drives(
        antioch.stage(), composite.so101_movable_joints
    )
    wrist_sensor = Camera(prim_path=composite.wrist_camera, resolution=(640, 480))
    world = antioch.world()
    world.reset()
    wrist_sensor.initialize()
    policy.initialize()
    policy.post_reset()
    zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    for _ in range(160):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
    align_wrist_camera_to_chassis(
        antioch.stage(), composite.wrist_camera, composite.end_effector, composite.go2_chassis
    )
    for _ in range(4):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)

    frame_directory = Path(tempfile.mkdtemp(prefix="simple-boxes-wrist-fpv-"))
    frames: list[np.ndarray] = []
    frame_means: list[float] = []

    def capture_wrist_frame(_step: int, *_args: object) -> None:
        wrist_data = wrist_sensor.get_rgba()
        if wrist_data is None or wrist_data.size == 0:
            return
        # Isaac can return the render product on CUDA for a GPU physics
        # pipeline.  Video encoding and PIL are host-only, so explicitly
        # read it back instead of relying on NumPy's implicit conversion.
        if hasattr(wrist_data, "detach"):
            wrist_data = wrist_data.detach().cpu().numpy()
        rgb = np.ascontiguousarray(np.asarray(wrist_data, dtype=np.uint8)[..., :3])
        frame_index = len(frames)
        Image.fromarray(rgb).save(frame_directory / f"frame_{frame_index:05d}.png")
        frames.append(rgb.copy())
        frame_means.append(float(rgb.mean()))
        antioch.Logger("factory_sre/simple_boxes_fpv").image("camera/wrist", rgb)

    # Include the genuine arm-mounted perspective before the base begins to
    # move, then one rendered sensor readback every 20 physics ticks.
    capture_wrist_frame(0)
    route = plan_to_service_box(target.tag_id, start=start, grid=tagged_box_occupancy())
    trace = Go2PolicyWaypointFollower(
        policy,
        world,
        maximum_speed_mps=0.45,
        maximum_yaw_rate_rps=1.3,
        final_tolerance_m=0.20,
        final_yaw_tolerance_rad=0.10,
        maximum_steps_per_waypoint=12000,
    ).follow(
        route,
        tagged_box_occupancy(),
        frame_callback=capture_wrist_frame,
        frame_every_steps=20,
    )

    # Keep the camera attached to the terminal arm link for the final planted
    # hold.  Reading and reauthoring its world pose is unnecessary for the
    # FPV and differs across Isaac GPU backends.
    for step in range(60):
        policy.forward(1.0 / 200.0, zero)
        world.step(render=True)
        if step % 4 == 0:
            capture_wrist_frame(trace.steps + step)

    video_path = frame_directory / "simple-boxes-box8-wrist-fpv.mp4"
    encoding_error = ""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "12",
                "-i",
                str(frame_directory / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        encoding_error = result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        encoding_error = str(exc)

    final_frame = frames[-1] if frames else None
    if final_frame is not None:
        final_path = frame_directory / "simple-boxes-box8-wrist-fpv-final.png"
        Image.fromarray(final_frame).save(final_path)
        run.add_artifact(final_path, name=final_path.name, content_type="image/png")
    video_exists = video_path.is_file() and video_path.stat().st_size > 0
    if video_exists:
        run.add_artifact(video_path, name=video_path.name, content_type="video/mp4")
    motion_score = (
        float(np.mean(np.abs(frames[0].astype(np.int16) - frames[-1].astype(np.int16))))
        if len(frames) >= 2
        else 0.0
    )
    run.check("SO-101 wrist camera produced FPV frames", len(frames) >= 30, detail=f"{len(frames)} frames")
    run.check(
        "wrist FPV H.264 artifact encoded",
        video_exists,
        detail=(f"{video_path.stat().st_size} bytes" if video_exists else encoding_error or "no video"),
    )
    run.check(
        "FPV changes during approach",
        motion_score >= 0.25,
        detail=f"mean absolute pixel change {motion_score:.3f}",
    )
    run.check(
        "FPV frames are visibly exposed",
        bool(frame_means) and all(5.0 <= mean <= 245.0 for mean in frame_means),
        detail=f"range {min(frame_means, default=0.0):.1f}-{max(frame_means, default=0.0):.1f}",
    )
    run.add_result(
        "simple_boxes_wrist_fpv",
        {
            "world": scene["layout"],
            "target": target.asset_id,
            "target_tag_id": target.tag_id,
            "camera_prim": composite.wrist_camera,
            "stowed_arm_joints": stowed_joint_count,
            "frame_count": len(frames),
            "motion_score": motion_score,
            "navigation_reached": trace.reached,
            "navigation_final_error_m": trace.final_error_m,
            "navigation_steps": trace.steps,
        },
    )
