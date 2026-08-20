# ROS 2 and DIMOS hardware mapping

[DIMOS](https://github.com/dimensionalOS/dimos) is useful for the later physical Go2 layer, not as a replacement for the Antioch/Isaac Sim certification stack.

## Where DIMOS helps

- Unitree Go2 hardware connectivity and teleoperation.
- Physical navigation, SLAM, costmaps, and A*-style routing.
- Agent/MCP integration for dispatch and human takeover.
- A practical bridge from `BaseAdapter` to a real robot runtime.

## Where it does not close this project

- The current DIMOS Go2 support does not certify the Go2 + SO-101 composite in Isaac Sim.
- Its documented manipulation paths target other arms; this repository still needs an SO-101-specific adapter.
- It does not provide this project's AprilTag-to-dock calibration, 8 N guarded magnetic docking, transient link epochs, or operation ledger.
- It cannot turn a failed simulation/payload gate into a physical-readiness claim.

## Adapter map

| Factory SRE boundary | ROS 2 / DIMOS implementation target |
| --- | --- |
| `BaseAdapter.navigate` | DIMOS Go2 navigation or ROS 2 Nav2 action client |
| `BaseAdapter.plant/stable` | Go2 stance command plus IMU/odometry stability gate |
| `BaseAdapter.emergency_stop` | Independent Unitree stop path; never only a planner cancel |
| `ArmAdapter.align/approach` | SO-101 ROS 2 control + MoveIt Servo or a measured custom controller |
| `ArmAdapter.guarded_dock` | Force-sensor loop with pose, penetration, and 8 N gates |
| `CameraAdapter.detect_tags` | ROS image transport + `pupil-apriltags` with calibrated intrinsics |
| `DiagnosticTransport` | HTTP over the dock-created USB/Ethernet link with mission/epoch headers |

## Recommended integration sequence

1. Keep Antioch as the simulation and evidence authority.
2. Implement a DIMOS-backed Go2 adapter in no-motor mode and pass `run_no_motor_dry_run`.
3. Implement the SO-101 ROS 2 adapter separately with measured frames and independent stop.
4. Replay the same manifest, identity, authorization, and diagnostic-target tests against hardware-in-the-loop.
5. Enable supervised motion only after the physical-readiness report is entirely green.
