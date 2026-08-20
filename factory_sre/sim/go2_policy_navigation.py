"""Closed-loop waypoint following around Isaac Sim's supplied Go2 policy."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

from .factory_navigation import RoutePlan
from .navigation import GridMap


def _wrap(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def _yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(frozen=True, slots=True)
class NavigationTrace:
    reached: bool
    positions: tuple[tuple[float, float, float], ...]
    final_error_m: float
    final_yaw_error_rad: float
    maximum_roll_pitch_deg: float
    occupied_samples: int
    occupied_positions: tuple[tuple[float, float, float], ...]
    steps: int


class Go2PolicyWaypointFollower:
    """Translate world-frame route error into Go2 body velocity commands."""

    def __init__(
        self,
        policy,
        world,
        *,
        physics_dt: float = 1.0 / 200.0,
        maximum_speed_mps: float = 0.55,
        maximum_yaw_rate_rps: float = 0.8,
        waypoint_tolerance_m: float = 0.24,
        final_tolerance_m: float = 0.30,
        final_yaw_tolerance_rad: float = 0.15,
        maximum_steps_per_waypoint: int = 3600,
    ) -> None:
        self.policy = policy
        self.world = world
        self.physics_dt = physics_dt
        self.maximum_speed_mps = maximum_speed_mps
        self.maximum_yaw_rate_rps = maximum_yaw_rate_rps
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.final_tolerance_m = final_tolerance_m
        self.final_yaw_tolerance_rad = final_yaw_tolerance_rad
        self.maximum_steps_per_waypoint = maximum_steps_per_waypoint

    def _pose(self):
        import numpy as np

        positions, orientations = self.policy.robot.get_world_poses()
        return (
            np.asarray(positions.numpy()[0], dtype=float),
            np.asarray(orientations.numpy()[0], dtype=float),
        )

    @staticmethod
    def _roll_pitch_degrees(quaternion) -> float:
        from math import asin, degrees

        w, x, y, z = (float(value) for value in quaternion)
        roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        return max(abs(degrees(roll)), abs(degrees(pitch)))

    def follow(
        self,
        route: RoutePlan,
        occupancy: GridMap,
        *,
        render_every_step: bool = False,
        frame_callback=None,
        frame_every_steps: int = 40,
        simulation_step_callback=None,
    ) -> NavigationTrace:
        import torch

        if frame_every_steps <= 0:
            raise ValueError("frame_every_steps must be positive")
        device = "cuda"
        samples: list[tuple[float, float, float]] = []
        occupied_samples = 0
        occupied_positions: list[tuple[float, float, float]] = []
        maximum_tilt = 0.0
        steps = 0
        for waypoint_index, waypoint in enumerate(route.waypoints[1:], start=1):
            final = waypoint_index == len(route.waypoints) - 1
            tolerance = self.final_tolerance_m if final else self.waypoint_tolerance_m
            for _ in range(self.maximum_steps_per_waypoint):
                position, quaternion = self._pose()
                yaw = _yaw_from_wxyz(quaternion)
                dx, dy = waypoint.x - float(position[0]), waypoint.y - float(position[1])
                distance = hypot(dx, dy)
                final_yaw_error = abs(_wrap(waypoint.yaw - yaw))
                if distance <= tolerance and (not final or final_yaw_error <= self.final_yaw_tolerance_rad):
                    break
                body_x = cos(yaw) * dx + sin(yaw) * dy
                body_y = -sin(yaw) * dx + cos(yaw) * dy
                # The route gives the final waypoint a short face-normal lead
                # in, so hold the service-face yaw for the complete terminal
                # segment.  Keep correcting x/y while turning: a learned
                # quadruped gait drifts during a nominal in-place rotation,
                # and zeroing translation here lets it walk out of the camera
                # standoff envelope before its yaw settles.
                desired_yaw = waypoint.yaw if final else atan2(dy, dx)
                linear_scale = 0.9
                yaw_error = _wrap(desired_yaw - yaw)
                yaw_rate = 2.5 * yaw_error
                if final and abs(yaw_error) > self.final_yaw_tolerance_rad:
                    # The roof payload creates a measurable steady-state yaw
                    # bias in the supplied gait.  Maintain enough authority to
                    # cross that bias, then let the proportional term settle.
                    minimum_terminal_yaw_rate = 0.9
                    yaw_rate = (1.0 if yaw_error > 0.0 else -1.0) * max(
                        abs(yaw_rate), minimum_terminal_yaw_rate
                    )
                command = torch.tensor(
                    [
                        max(-self.maximum_speed_mps, min(self.maximum_speed_mps, linear_scale * body_x)),
                        max(-0.35, min(0.35, linear_scale * body_y)),
                        max(-self.maximum_yaw_rate_rps, min(self.maximum_yaw_rate_rps, yaw_rate)),
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                self.policy.forward(self.physics_dt, command)
                if simulation_step_callback is not None:
                    simulation_step_callback(self.physics_dt)
                capture_frame = frame_callback is not None and steps % frame_every_steps == 0
                self.world.step(render=render_every_step or steps % 10 == 0 or capture_frame)
                steps += 1
                if capture_frame:
                    frame_callback(steps, position, quaternion)
                if steps % 10 == 0:
                    sample = (float(position[0]), float(position[1]), float(position[2]))
                    samples.append(sample)
                    maximum_tilt = max(maximum_tilt, self._roll_pitch_degrees(quaternion))
                    if not occupancy.free(occupancy.world_to_cell(sample[0], sample[1])):
                        occupied_samples += 1
                        occupied_positions.append(sample)
            else:
                break

        # Actively hold the planted service pose.  A zero velocity command
        # alone allows residual yaw momentum from the roof payload to carry
        # the body and wrist camera away from the tag after arrival.
        for _ in range(240):
            position, quaternion = self._pose()
            yaw = _yaw_from_wxyz(quaternion)
            dx, dy = route.goal.x - float(position[0]), route.goal.y - float(position[1])
            body_x = cos(yaw) * dx + sin(yaw) * dy
            body_y = -sin(yaw) * dx + cos(yaw) * dy
            yaw_error = _wrap(route.goal.yaw - yaw)
            yaw_rate = 2.5 * yaw_error
            if abs(yaw_error) > self.final_yaw_tolerance_rad:
                yaw_rate = (1.0 if yaw_error > 0.0 else -1.0) * max(abs(yaw_rate), 0.9)
            hold = torch.tensor(
                [
                    max(-0.25, min(0.25, 0.9 * body_x)),
                    max(-0.25, min(0.25, 0.9 * body_y)),
                    max(-self.maximum_yaw_rate_rps, min(self.maximum_yaw_rate_rps, yaw_rate)),
                ],
                dtype=torch.float32,
                device=device,
            )
            self.policy.forward(self.physics_dt, hold)
            if simulation_step_callback is not None:
                simulation_step_callback(self.physics_dt)
            capture_frame = frame_callback is not None and steps % frame_every_steps == 0
            self.world.step(render=render_every_step or steps % 10 == 0 or capture_frame)
            steps += 1
            if capture_frame:
                frame_callback(steps, position, quaternion)
        final_position, final_orientation = self._pose()
        samples.append(tuple(float(value) for value in final_position))
        maximum_tilt = max(maximum_tilt, self._roll_pitch_degrees(final_orientation))
        final_error = hypot(route.goal.x - float(final_position[0]), route.goal.y - float(final_position[1]))
        final_yaw_error = abs(_wrap(route.goal.yaw - _yaw_from_wxyz(final_orientation)))
        return NavigationTrace(
            reached=final_error <= self.final_tolerance_m,
            positions=tuple(samples),
            final_error_m=final_error,
            final_yaw_error_rad=final_yaw_error,
            maximum_roll_pitch_deg=maximum_tilt,
            occupied_samples=occupied_samples,
            occupied_positions=tuple(occupied_positions),
            steps=steps,
        )
