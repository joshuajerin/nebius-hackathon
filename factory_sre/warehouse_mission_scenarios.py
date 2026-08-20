"""Demo mission in the latest wide industrial warehouse scene."""

from __future__ import annotations

import antioch


PROFILE = antioch.BootProfile(
    physics_dt=1.0 / 200.0,
    render_dt=1.0 / 50.0,
    physics_engine="physx",
    render_quality="performance",
    viewport=(1280, 720),
)


@antioch.scenario(
    name="warehouse_single_repair_demo",
    description=(
        "Use the supplied Go2 policy in the latest high-bay warehouse, hold the "
        "selected service dock for five seconds, restore its beacon, and record MP4 evidence."
    ),
    tags=("demo", "warehouse", "go2", "policy", "repair", "video"),
    sim=PROFILE,
    capture=True,
)
def warehouse_single_repair_demo(
    run: antioch.ScenarioRun,
    box_number: int = 4,
) -> None:
    import subprocess
    import tempfile
    from math import pi
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from pxr import Gf, Sdf, UsdShade

    from factory_sre.contracts import Pose2D
    from factory_sre.sim.factory_navigation import RoutePlan
    from factory_sre.sim.go2_policy_navigation import Go2PolicyWaypointFollower
    from factory_sre.sim.industrial_cells_v1 import (
        IndustrialCellConfig,
        build_industrial_cell_grid,
    )
    from factory_sre.sim.industrial_cells_v1.runtime import GO2_START_POSITION_M
    from factory_sre.sim.navigation import GridMap

    if box_number not in range(1, 5):
        run.fail("warehouse demo currently targets the front row: box_number 1..4")

    SimulationManager.set_physics_sim_device("cuda")
    scene = build_industrial_cell_grid(IndustrialCellConfig())
    world = antioch.world()
    scene.prepare_runtime()
    world.reset()
    scene.initialize_runtime()

    target_x = 2.6 * (box_number - 1)
    start = Pose2D(GO2_START_POSITION_M[0], GO2_START_POSITION_M[1], 0.0)
    goal = Pose2D(target_x, -1.12, pi / 2.0)
    middle = Pose2D(target_x, GO2_START_POSITION_M[1], pi / 2.0)
    route = RoutePlan(
        asset_id=f"service-box-{box_number}",
        start=start,
        goal=goal,
        waypoints=(start, middle, goal),
        length_m=abs(target_x - start.x) + abs(goal.y - start.y),
    )
    occupancy = GridMap(
        width=120,
        height=80,
        resolution_m=0.10,
        origin_xy=(-2.0, -3.0),
        occupied=frozenset(),
    )

    stage = antioch.stage()
    material_root = "/World/IndustrialCellsV1/Materials"

    def make_status_material(name: str, color: tuple[float, float, float]) -> str:
        path = f"{material_root}/{name}"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Preview")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return path

    offline_material = make_status_material("DemoOffline", (0.90, 0.04, 0.03))
    online_material = make_status_material("DemoOnline", (0.03, 0.80, 0.18))
    beacon_path = (
        f"/World/IndustrialCellsV1/ServiceBoxes/service_box_{box_number}/StatusBeacon"
    )
    beacon = stage.GetPrimAtPath(beacon_path)
    UsdShade.MaterialBindingAPI.Apply(beacon).Bind(UsdShade.Material.Get(stage, offline_material))

    set_camera_view(
        eye=[10.6, -7.8, 4.7],
        target=[3.9, 0.45, 1.05],
        camera_prim_path="/OmniverseKit_Persp",
    )
    zero = torch.zeros(3, dtype=torch.float32, device="cuda")
    for _ in range(240):
        scene.go2_policy.forward(1.0 / 200.0, zero)
        scene.step_workcells(1.0 / 200.0)
        world.step(render=True)

    frame_directory = Path(tempfile.mkdtemp(prefix="warehouse-repair-demo-"))
    frame_count = 0
    phase = "DISPATCH · BOX 4 OFFLINE"
    hold_progress = 0.0

    def capture_frame(_step: int, _position=None, _orientation=None) -> None:
        nonlocal frame_count
        viewport = antioch.capture_viewport()
        if viewport is None:
            return
        image = Image.fromarray(np.asarray(viewport, dtype=np.uint8)[..., :3])
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((26, 24, 560, 112), radius=16, fill=(6, 12, 18, 220))
        draw.text((48, 42), "FACTORY SRE · WAREHOUSE DEMO", fill=(236, 244, 249))
        draw.text((48, 72), phase, fill=(255, 188, 48))
        if hold_progress > 0.0:
            draw.rectangle((48, 96, 48 + int(470 * hold_progress), 103), fill=(47, 214, 118))
        image.save(frame_directory / f"frame_{frame_count:05d}.png")
        frame_count += 1

    capture_frame(0)
    trace = Go2PolicyWaypointFollower(
        scene.go2_policy,
        world,
        maximum_speed_mps=0.55,
        maximum_yaw_rate_rps=1.3,
        final_tolerance_m=0.20,
        final_yaw_tolerance_rad=0.06,
        maximum_steps_per_waypoint=12_000,
    ).follow(
        route,
        occupancy,
        frame_callback=capture_frame,
        frame_every_steps=30,
        simulation_step_callback=scene.step_workcells,
    )

    phase = f"MAGNETIC SERVICE HOLD · TAG {29 + box_number}"
    hold_steps = 1_000
    for step in range(hold_steps):
        scene.go2_policy.forward(1.0 / 200.0, zero)
        scene.step_workcells(1.0 / 200.0)
        world.step(render=step % 10 == 0)
        if step % 25 == 0:
            hold_progress = min(1.0, step / hold_steps)
            capture_frame(trace.steps + step)

    UsdShade.MaterialBindingAPI.Apply(beacon).Bind(UsdShade.Material.Get(stage, online_material))
    phase = f"RECOVERED · BOX {box_number} ONLINE"
    hold_progress = 1.0
    for step in range(240):
        scene.go2_policy.forward(1.0 / 200.0, zero)
        scene.step_workcells(1.0 / 200.0)
        world.step(render=step % 10 == 0)
        if step % 30 == 0:
            capture_frame(trace.steps + hold_steps + step)

    video_path = frame_directory / f"warehouse-box{box_number}-repair.mp4"
    encoded = subprocess.run(
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
        timeout=180,
        check=False,
    )
    video_ok = video_path.is_file() and video_path.stat().st_size > 0
    if video_ok:
        run.add_artifact(
            video_path,
            name=video_path.name,
            content_type="video/mp4",
        )
    run.check(
        "supplied Go2 policy reaches the failed warehouse cell",
        trace.reached,
        detail=f"error {trace.final_error_m:.3f} m, yaw {trace.final_yaw_error_rad:.3f} rad",
    )
    run.check(
        "five simulated seconds of service hold complete",
        hold_steps * (1.0 / 200.0) >= 5.0,
        detail="5.000 s",
    )
    run.check(
        "warehouse repair MP4 is encoded",
        video_ok and frame_count >= 20 and encoded.returncode == 0,
        detail=f"{frame_count} frames, {video_path.stat().st_size if video_ok else 0} bytes; {encoded.stderr}",
    )
    run.add_result(
        "warehouse_repair",
        {
            "asset_id": route.asset_id,
            "tag_id": 29 + box_number,
            "status": "online",
            "hold_seconds": 5.0,
            "navigation_steps": trace.steps,
            "final_error_m": trace.final_error_m,
            "maximum_roll_pitch_deg": trace.maximum_roll_pitch_deg,
            "video_frames": frame_count,
        },
    )
