"""Persistent Antioch/Isaac process for dashboard-driven Factory SRE missions."""

from __future__ import annotations

import os
import signal
import time
import traceback
from math import atan2, hypot
from pathlib import Path
from threading import Event
from typing import Any

from factory_sre.contracts import Pose2D
from factory_sre.live.client import FleetClient
from factory_sre.live.planner import (
    REST_POSE,
    docking_pose_for_asset,
    plan_between,
    plan_failure_route,
)
from factory_sre.sim.tagged_box_grid import service_box_for_number


PHYSICS_DT = 1.0 / 200.0
CONTACT_TOLERANCE_M = 0.025
CONTACT_HOLD_SECONDS = 5.0


def _box_number(asset_id: str) -> int:
    return int(asset_id.removeprefix("service-box-"))


def _yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class PersistentFactorySim:
    """Own the live world and execute every queued failure in planned order."""

    def __init__(self) -> None:
        import antioch
        import numpy as np
        import torch
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
        from isaacsim.storage.native import get_assets_root_path

        from factory_sre.sim.composite import (
            author_go2_so101_composite,
            configure_so101_stow_drives,
        )
        from factory_sre.sim.go2_policy_navigation import Go2PolicyWaypointFollower
        from factory_sre.sim.magnetic_arm import MagneticArmController
        from factory_sre.sim.tagged_box_grid import build_tagged_box_grid, tagged_box_occupancy

        self.antioch = antioch
        self.np = np
        self.torch = torch
        self.stop = Event()
        self.client = FleetClient(
            os.environ.get("FACTORY_SRE_CONTROL_URL", "http://127.0.0.1:8765"),
            timeout_seconds=2.0,
        )
        self.frame_path = Path(
            os.environ.get("FACTORY_SRE_FRAME_PATH", "/workspace/output/live/latest.jpg")
        )
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_frame_wall_time = 0.0
        self._last_status: dict[str, Any] | None = None
        self._active_queue: tuple[str, ...] = ()
        self._phase = "starting"
        self._target: str | None = None
        self._detail = "Booting persistent Antioch simulation"

        SimulationManager.set_physics_sim_device("cuda")
        build_tagged_box_grid(incident_box_number=8)
        assets = get_assets_root_path()
        self.policy = Go2FlatTerrainPolicy(
            prim_path="/World/Go2",
            position=[REST_POSE.x, REST_POSE.y, 0.50],
            orientation=[1.0, 0.0, 0.0, 0.0],
            policy_path=assets + "/Isaac/Samples/Policies/go2/physx_policy.pt",
            env_config_path=assets + "/Isaac/Samples/Policies/go2/physx_env.yaml",
        )
        self.composite = author_go2_so101_composite(
            mode="controller_isolated",
            initial_chassis_position_m=(REST_POSE.x, REST_POSE.y, 0.50),
        )
        driven_joints = configure_so101_stow_drives(
            antioch.stage(), self.composite.so101_movable_joints
        )
        if driven_joints != 6:
            raise RuntimeError(f"expected six SO-101 drives, found {driven_joints}")
        self.arm = MagneticArmController(antioch.stage(), self.composite)
        self.world = antioch.world()
        self.world.reset()
        self.policy.initialize()
        self.policy.post_reset()
        if not self.arm.ready():
            raise RuntimeError("SO-101 magnetic IK tensor views did not initialize")
        self.zero_command = torch.zeros(3, dtype=torch.float32, device="cuda")
        self.occupancy = tagged_box_occupancy()
        self.follower = Go2PolicyWaypointFollower(
            self.policy,
            self.world,
            maximum_speed_mps=0.55,
            maximum_yaw_rate_rps=1.3,
            final_tolerance_m=0.14,
            final_yaw_tolerance_rad=0.04,
            maximum_steps_per_waypoint=12000,
        )
        set_camera_view(
            eye=[7.7, -5.5, 6.8],
            target=[2.25, 1.55, 0.30],
            camera_prim_path="/OmniverseKit_Persp",
        )
        for _ in range(240):
            self._step(render=True)
        self.arm.stow()
        self._phase = "resting"
        self._detail = "At rest; monitoring all eight workstations"
        self._publish(force_frame=True)

    def _pose(self) -> tuple[Any, Any]:
        positions, orientations = self.policy.robot.get_world_poses()
        return positions.numpy()[0], orientations.numpy()[0]

    def _pose_2d(self) -> Pose2D:
        position, quaternion = self._pose()
        return Pose2D(
            float(position[0]),
            float(position[1]),
            _yaw_from_wxyz(quaternion),
        )

    def _sim_time(self) -> float:
        return float(self.world.current_time)

    def _step(self, *, render: bool = False) -> None:
        self.policy.forward(PHYSICS_DT, self.zero_command)
        self.world.step(render=render)

    def _set_beacons(self, snapshot: dict[str, Any]) -> None:
        from pxr import Gf, UsdGeom

        stage = self.antioch.stage()
        for station in snapshot["workstations"]:
            path = (
                "/World/TaggedBoxes/"
                + station["asset_id"].replace("-", "_")
                + "/StatusBeacon"
            )
            cube = UsdGeom.Cube.Get(stage, path)
            if cube:
                color = (0.92, 0.10, 0.08) if station["status"] == "offline" else (0.03, 0.72, 0.18)
                cube.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    def _write_frame(self) -> None:
        from PIL import Image

        frame = self.antioch.capture_viewport()
        if frame is None:
            return
        rgb = self.np.asarray(frame, dtype=self.np.uint8)[..., :3]
        temporary = self.frame_path.with_name(self.frame_path.name + ".tmp")
        Image.fromarray(rgb).save(temporary, format="JPEG", quality=86)
        os.replace(temporary, self.frame_path)
        self._last_frame_wall_time = time.monotonic()

    def _publish(self, *, force_frame: bool = False) -> dict[str, Any]:
        position, quaternion = self._pose()
        payload = {
            "phase": self._phase,
            "position": [float(value) for value in position],
            "yaw": _yaw_from_wxyz(quaternion),
            "current_target": self._target,
            "planned_queue": list(self._active_queue),
            "simulator_ready": True,
            "detail": self._detail,
        }
        self._last_status = self.client.robot(payload)
        self._set_beacons(self._last_status)
        if force_frame or time.monotonic() - self._last_frame_wall_time >= 0.35:
            self._write_frame()
        return self._last_status

    def _navigation_frame(self, _step: int, _position=None, _orientation=None) -> None:
        if self.stop.is_set():
            raise InterruptedError("persistent simulator stopping")
        self._publish()

    def _follow(self, route, *, phase: str, detail: str) -> bool:
        self._phase = phase
        self._detail = detail
        self._publish(force_frame=True)
        trace = self.follower.follow(
            route,
            self.occupancy,
            frame_callback=self._navigation_frame,
            frame_every_steps=100,
        )
        self._publish(force_frame=True)
        return bool(
            trace.reached
            and trace.final_error_m <= self.follower.final_tolerance_m
            and trace.final_yaw_error_rad <= self.follower.final_yaw_tolerance_rad
        )

    def _move_to_target(self, asset_id: str, coarse_route) -> bool:
        if not self._follow(
            coarse_route,
            phase="navigating",
            detail=f"Following occupancy route to {_box_number(asset_id)}",
        ):
            self._phase = "navigation_failed"
            self._detail = f"Could not reach the coarse approach for {_box_number(asset_id)}"
            self._publish(force_frame=True)
            return False
        close_pose = docking_pose_for_asset(asset_id)
        close_route = plan_between(
            self._pose_2d(),
            close_pose,
            asset_id=asset_id,
            grid=self.occupancy,
        )
        return self._follow(
            close_route,
            phase="aligning",
            detail=f"Planting at AprilTag {29 + _box_number(asset_id)} docking stance",
        )

    def _repair(self, asset_id: str) -> bool:
        target = service_box_for_number(_box_number(asset_id))
        self._phase = "visual_servo"
        self._detail = f"Solving SO-101 magnetic pose for AprilTag {target.tag_id}"
        self._publish(force_frame=True)
        solution = self.arm.solve(target.tag_center)
        if not solution.success:
            self._phase = "ik_failed"
            self._detail = f"Existing Isaac Robot Poser could not reach AprilTag {target.tag_id}"
            self._publish(force_frame=True)
            return False

        acquired_at: float | None = None
        deadline = self._sim_time() + 15.0
        last_contact_update = -1.0
        while self._sim_time() < deadline and not self.stop.is_set():
            self.arm.apply_solution()
            self._step(render=True)
            contact_position, _ = self.arm.contact_world_pose()
            error_m = float(
                self.np.linalg.norm(
                    self.np.asarray(target.tag_center, dtype=float) - contact_position
                )
            )
            now = self._sim_time()
            attached = error_m <= CONTACT_TOLERANCE_M
            if attached and acquired_at is None:
                acquired_at = now
                self._phase = "magnetic_contact"
                self._detail = f"Magnet locked to Tag {target.tag_id}; holding for five simulated seconds"
                self.client.contact(asset_id=asset_id, attached=True, sim_time_s=now)
                last_contact_update = now
                self._publish(force_frame=True)
            elif not attached and acquired_at is not None:
                self.client.contact(asset_id=asset_id, attached=False, sim_time_s=now)
                acquired_at = None
                self._phase = "visual_servo"
                self._detail = f"Contact shifted by {error_m * 1000.0:.1f} mm; reacquiring"
                self._publish(force_frame=True)
            elif attached and now - last_contact_update >= 0.20:
                status = self.client.contact(asset_id=asset_id, attached=True, sim_time_s=now)
                last_contact_update = now
                self._last_status = status
                if asset_id not in status["pending_failures"]:
                    self._phase = "recovered"
                    self._detail = f"Box {_box_number(asset_id)} is back online"
                    self._publish(force_frame=True)
                    return True
            if int(now / 0.25) != int((now - PHYSICS_DT) / 0.25):
                self._publish()

        self.client.contact(asset_id=asset_id, attached=False, sim_time_s=self._sim_time())
        self._phase = "contact_failed"
        self._detail = f"Could not maintain physical contact with AprilTag {target.tag_id} for five seconds"
        self._publish(force_frame=True)
        return False

    def _stow(self) -> None:
        self._phase = "disconnecting"
        self._detail = "Releasing magnetic contact and stowing SO-101"
        if self._target is not None:
            self.client.contact(
                asset_id=self._target,
                attached=False,
                sim_time_s=self._sim_time(),
            )
        for step in range(360):
            self.arm.stow()
            self._step(render=step % 10 == 0)
            if step % 100 == 0:
                self._publish()
        self._publish(force_frame=True)

    def _return_to_rest(self) -> None:
        current = self._pose_2d()
        if hypot(current.x - REST_POSE.x, current.y - REST_POSE.y) > 0.25:
            route = plan_between(current, REST_POSE, asset_id="rest", grid=self.occupancy)
            self._target = None
            self._active_queue = ()
            self._follow(route, phase="returning", detail="All repairs complete; returning to rest")
        self._target = None
        self._active_queue = ()
        self._phase = "resting"
        self._detail = "At rest; monitoring all eight workstations"
        self._publish(force_frame=True)

    def run(self) -> None:
        retry_after: dict[str, float] = {}
        while not self.stop.is_set():
            try:
                snapshot = self.client.get()
                pending = tuple(
                    asset_id
                    for asset_id in snapshot["pending_failures"]
                    if retry_after.get(asset_id, 0.0) <= time.monotonic()
                )
                if not pending:
                    if self._phase != "resting":
                        self._return_to_rest()
                    for step in range(100):
                        if self.stop.is_set():
                            return
                        self._step(render=step % 20 == 0)
                    self._publish()
                    continue

                planned = plan_failure_route(pending, start=self._pose_2d())
                self._active_queue = planned.asset_ids
                self._target = planned.asset_ids[0]
                self._phase = "dispatching"
                self._detail = f"Planning {len(planned.asset_ids)} queued repair mission(s)"
                self._publish(force_frame=True)
                reached = self._move_to_target(self._target, planned.legs[0])
                repaired = reached and self._repair(self._target)
                self._stow()
                if not repaired:
                    retry_after[self._target] = time.monotonic() + 10.0
                else:
                    retry_after.pop(self._target, None)
                # Always re-read the fleet after a repair so failures injected
                # mid-mission are included in the next nearest-first plan.
            except InterruptedError:
                return
            except Exception:
                self._phase = "sim_error"
                self._detail = traceback.format_exc().splitlines()[-1]
                try:
                    self._publish(force_frame=True)
                except Exception:
                    print(traceback.format_exc(), flush=True)
                time.sleep(1.0)


def _serve_degraded(detail: str, stop_event: Event) -> None:
    from PIL import Image, ImageDraw

    frame_path = Path(
        os.environ.get("FACTORY_SRE_FRAME_PATH", "/workspace/output/live/latest.jpg")
    )
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    client = FleetClient(
        os.environ.get("FACTORY_SRE_CONTROL_URL", "http://127.0.0.1:8765"),
        timeout_seconds=2.0,
    )
    while not stop_event.is_set():
        image = Image.new("RGB", (1280, 720), (24, 31, 39))
        draw = ImageDraw.Draw(image)
        draw.text((56, 56), "FACTORY SRE SIM UNAVAILABLE", fill=(238, 242, 246))
        draw.text((56, 102), detail[:150], fill=(240, 168, 84))
        temporary = frame_path.with_name(frame_path.name + ".tmp")
        image.save(temporary, format="JPEG", quality=86)
        os.replace(temporary, frame_path)
        try:
            client.robot(
                {
                    "phase": "sim_error",
                    "position": [0.0, 0.0, 0.0],
                    "yaw": 0.0,
                    "current_target": None,
                    "planned_queue": [],
                    "simulator_ready": False,
                    "detail": detail,
                }
            )
        except Exception:
            pass
        stop_event.wait(5.0)


def main() -> None:
    stop_event = Event()

    def stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if os.environ.get("FACTORY_SRE_LIVE_SIM_ENABLED", "0") != "1":
        _serve_degraded("Live simulator disabled while IK startup is under repair", stop_event)
        return

    import antioch

    antioch.boot(
        physics_engine="physx",
        physics_dt=PHYSICS_DT,
        render_dt=1.0 / 50.0,
        render_quality="performance",
        viewport=(1280, 720),
    )
    try:
        simulator = PersistentFactorySim()
    except Exception as exc:
        # Keep the managed sim service available for independent scenario
        # execution when the optional live dashboard runtime cannot initialize.
        # The frame and fleet state are explicitly degraded; this is not a
        # readiness bypass and scenario checks still run in their own process.
        print(traceback.format_exc(), flush=True)
        detail = f"Live simulator unavailable: {type(exc).__name__}: {exc}"
        _serve_degraded(detail, stop_event)
        return

    simulator.stop = stop_event
    print("factory-sre persistent Antioch simulator ready", flush=True)
    simulator.run()


if __name__ == "__main__":
    main()
