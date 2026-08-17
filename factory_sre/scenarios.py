"""Antioch evidence scenarios for Factory SRE."""

from __future__ import annotations

import antioch

from factory_sre.contracts import AssetPins


@antioch.scenario(
    tags=["smoke", "factory-sre", "assets"],
    capture=False,
    sim=antioch.BootProfile(
        physics_dt=1 / 200,
        render_dt=1 / 50,
        render_quality="performance",
        viewport=(960, 540),
    ),
)
def asset_compatibility_smoke(run: antioch.ScenarioRun) -> None:
    """Prove exact robot assets, joint trees, camera, tags, and dock geometry."""

    import numpy as np
    import rerun as rr
    from isaacsim.core.deprecation_manager import import_module
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera

    from factory_sre.sim.assets import inspect_robot_stage, load_robot_assets
    from factory_sre.sim.scene import add_smoke_fixture

    pins = AssetPins()
    world = antioch.world()
    world.scene.add_ground_plane()
    create_prim("/World/dome_light", "DomeLight", attributes={"inputs:intensity": 250.0})
    create_prim("/World/key_light", "DistantLight", attributes={"inputs:intensity": 500.0})
    loaded = load_robot_assets(pins=pins)
    go2 = loaded["go2_controller"]
    fixture = add_smoke_fixture(world)

    camera = Camera(
        prim_path="/World/WristCamera",
        name="wrist_camera",
        resolution=(640, 480),
    )
    world.reset()
    camera.initialize()
    set_camera_view(
        eye=[-1.8, -3.8, 2.2],
        target=[1.8, 0.0, 0.8],
        camera_prim_path="/World/WristCamera",
    )

    # The policy controller needs live physics tensors. Isaac's own Go2
    # example initializes it after the first physics step.
    world.step(render=True)
    go2.initialize()
    go2.post_reset()
    torch = import_module("torch")
    zero_command = torch.zeros(3, device=str(go2.robot._device))

    camera_samples: list[float] = []
    rgba = np.empty((0, 0, 4), dtype=np.uint8)
    for _ in range(60):
        go2.forward(1 / 200, zero_command)
        world.step(render=True)
        frame = camera.get_rgba()
        rgba = np.asarray(frame) if frame is not None else np.empty((0, 0, 4), dtype=np.uint8)
        if rgba.ndim == 3 and rgba.shape[-1] >= 3 and rgba.size:
            camera_samples.append(float(rgba[:, :, :3].mean()))

    rgb = rgba[:, :, :3] if rgba.ndim == 3 and rgba.shape[-1] >= 3 else np.empty((0, 0, 3), dtype=np.uint8)
    if rgb.size:
        antioch.Logger("factory_sre").image("wrist/rgb", rgb)

    layout = inspect_robot_stage()
    go2_roots = [path for path in layout["articulation_roots"] if path.startswith("/World/Go2")]
    so101_roots = [path for path in layout["articulation_roots"] if path.startswith("/World/SO101")]
    valid_tags = [path for path in fixture["tags"] if is_prim_path_valid(path)]
    valid_docks = [path for path in fixture["docks"] if is_prim_path_valid(path)]

    run.add_result("so101_asset", f"{pins.so101_name}:{pins.so101_version}")
    run.add_result("go2_usd", loaded["go2_usd"])
    run.add_result("articulation_roots", layout["articulation_roots"])
    run.add_result("go2_joint_count", len(layout["go2_joints"]))
    run.add_result("so101_joint_count", len(layout["so101_joints"]))
    run.add_result("go2_joints", layout["go2_joints"])
    run.add_result("so101_joints", layout["so101_joints"])
    run.add_result("camera_shape", list(rgb.shape))
    run.add_result("camera_mean", round(float(rgb.mean()), 3) if rgb.size else None)
    run.add_result("camera_mean_range", [round(min(camera_samples), 3), round(max(camera_samples), 3)] if camera_samples else None)
    run.add_result("tag_paths", valid_tags)
    run.add_result("dock_paths", valid_docks)

    run.check("the policy-backed Go2 asset loaded", is_prim_path_valid("/World/Go2"), detail=loaded["go2_usd"])
    run.check("the pinned SO-101 asset loaded", bool(loaded["so101"].IsValid()), detail=f"{pins.so101_name}:{pins.so101_version}")
    run.check("Go2 exposes an articulation root", bool(go2_roots), detail=", ".join(go2_roots) or "none")
    run.check("SO-101 exposes an articulation root", bool(so101_roots), detail=", ".join(so101_roots) or "none")
    run.check("Go2 exposes controllable joints", len(layout["go2_joints"]) >= 12, detail=str(len(layout["go2_joints"])))
    run.check("SO-101 exposes controllable joints", len(layout["so101_joints"]) >= 6, detail=str(len(layout["so101_joints"])))
    run.check("both workbench tags exist", len(valid_tags) == 2, detail=", ".join(valid_tags))
    run.check("both magnetic dock targets exist", len(valid_docks) == 2, detail=", ".join(valid_docks))
    run.check("the wrist camera returns RGB", rgb.shape == (480, 640, 3), detail=str(rgb.shape))
    run.check("the wrist image is not blank", bool(rgb.size and 2.0 < float(rgb.mean()) < 250.0), detail=f"mean={float(rgb.mean()) if rgb.size else 'none'}")

    # Minimal 3D evidence remains useful even if the camera check fails.
    logger = antioch.Logger("factory_sre")
    logger.value("layout/docks", rr.Points3D([[2.40, -1.50, 0.85], [2.40, 0.90, 0.85]], radii=0.08, colors=[[25, 166, 242], [25, 166, 242]]))
    logger.value("layout/tags", rr.Points3D([[2.39, -1.20, 1.10], [2.39, 1.20, 1.10]], radii=0.07, colors=[[245, 245, 245], [245, 245, 245]]))
