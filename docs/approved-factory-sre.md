# Factory SRE: approved two-workbench implementation

This is the source of truth for the hackathon demo. It supersedes the preserved one-workcell prototype without deleting that work.

## Demo contract

The simulated Go2 carries `so101_antioch@1.3.2` and a wrist camera. An incident names either `workbench-a` or `workbench-b`. The robot plans through an inflated occupancy grid to that manifest's approach pose, accepts only the expected `tag36h11` identity, computes the service target as `T_world_tag × T_tag_dock`, docks under pose/contact/force limits, opens a transient diagnostic listener, restarts one crashed vision service, verifies its heartbeat, disconnects, and reports.

| Workspace | AprilTag | Printed size | Dock transform | Approach pose |
| --- | ---: | ---: | --- | --- |
| `workbench-a` | 10 | 120 mm | 0.30 m left of tag center | `(1.5, -0.9, π/2)` |
| `workbench-b` | 20 | 120 mm | 0.30 m left of tag center | `(3.7, -0.9, π/2)` |

The adjacent workspace is never a fallback destination. Observing its tag produces `WRONG_ASSET` before any dock attempt or mutation.

## Runtime architecture

`antioch.yaml` defines exactly two services:

1. `sim`: scene, Go2/SO-101 composition, wrist camera, route planner, perception, docking controller, mission state machine, and telemetry.
2. `diagnostic-target`: always-on control endpoint, contact-gated diagnostic endpoint, leases, epochs, command sequencing, and SQLite ledger.

The hero mutation allowlist is exactly `{restart_vision_service}`. Network restoration and camera-driver reload remain diagnosable but are not autonomously authorized.

## Implemented boundaries

- `factory_sre/contracts.py`: manifests, typed frames, outcomes, and adapter protocols.
- `factory_sre/sim/composite.py`: strict single-articulation composite and controller-isolated physical-mount fallback.
- `factory_sre/sim/scene.py`: adjacent guarded workspaces, deterministic tabletop motion, tags, docks, and aisle obstacle.
- `factory_sre/sim/factory_navigation.py`: incident-selected occupancy-grid A* route.
- `factory_sre/sim/go2_policy_navigation.py`: body-frame waypoint follower around Isaac Sim's supplied Go2 policy.
- `factory_sre/sim/docking.py`: identity registration and 5 mm / 3° / 8 N magnetic engagement gates.
- `factory_sre/target/`: listener lifecycle, epochs, leases, and crash-reconcilable operation ledger.
- `factory_sre/readiness.py`: fail-closed physical-readiness and no-motor adapter checks.
- `factory_sre/evidence.py`: exact version/config hashes and 20/10/5 Hz evidence schedule.

## Verified state

Local verification currently passes 53 tests. Preserved historical tests that import the abandoned root-level `REGISTRY`/`VISION_CELL` API are explicitly non-collected; maintained direct-USB tests still run.

Confirmed Antioch runs:

- `asset_compatibility_smoke`: `e9f997100694496d88aa51483d55b7ec` — passed.
- `two_workbench_scene_smoke`: `4c58359ca8ec46f882b435824d8dbf63` — passed.
- `hero_diagnostic_recovery` target B: `5893b66c6de942f2886023cebbcb501c` — passed.
- `go2_forward_stop_smoke`: `f98474ed52e4462cb748f1bc48a31c52` — passed with 2.372 m motion and 0.114 m/s residual speed.
- `composite_mount_smoke`: `b8bd9286644e4ff294ffa4984b50c88c` — structural checks passed: one articulation, 12 Go2 joints, 6 SO-101 joints, real mount relationship, wrist camera on `/World/SO101/jaw`.

Do not interpret structural composition or controller-level docking math as physical payload-navigation/docking certification. Those remain separate physics gates.

## Run it

```bash
cd /Users/joshuajerin/Desktop/jarvis/hackathon-nebius
uv sync --extra dev
uv run pytest -q
uv run antioch scenario collect --json
uv run antioch scenario run --scenario composite_mount_smoke --queue --json
uv run antioch scenario run --scenario manifest_route_and_registration --queue --json
uv run antioch scenario run --scenario guarded_dock_controller_certification --queue --json
uv run antioch scenario run --scenario hero_diagnostic_recovery --case target-b --queue --json
```

Inspect any result with:

```bash
uv run antioch scenario show RUN_ID --json
uv run antioch scenario logs RUN_ID
uv run antioch scenario download RUN_ID
```

## Definition of ready

Simulation readiness requires the held-out physics/navigation/perception/docking runs to meet the release gates. Physical readiness additionally requires measured mount/tool CAD, approved calibration, a wrist force sensor, independently verified emergency stop, real Go2 and SO-101 adapters, and supervised low-speed trials. The repository deliberately fails closed until those inputs exist.

See [`calibration-and-safety.md`](calibration-and-safety.md) and [`ros2-dimos-mapping.md`](ros2-dimos-mapping.md).
