# Factory SRE — Simulation

Isaac Sim environment for a mobile robotic first responder to factory downtime.
A Unitree Go2 carries an SO-101 arm and wrist camera, identifies a failed cell,
approaches its service dock, and provides physical out-of-band diagnostic access.

This branch owns the simulation boundary only: robot assets and control, factory
scene composition, perception, navigation, docking, and scenario evidence. The
diagnostic target service and product-facing components can integrate through
typed contracts without sharing simulator internals.

## Current simulation gate

The `asset_compatibility_smoke` scenario composes:

- Isaac Sim 6.0.1's policy-backed Go2 asset;
- pinned Antioch asset `so101_antioch@1.3.2`;
- two factory workbenches with Tag 10/Tag 20 and dock targets;
- an illuminated 640×480 wrist-camera view.

The latest completed evidence established a Go2 articulation with 12 joints,
an SO-101 articulation with 7 joints, and a nonblank camera frame. Do not rerun
this gate unless its assets, runtime, or acceptance criteria change.

## Local setup

Requires Python 3.12 and `uv`.

```bash
uv sync
source .venv/bin/activate
antioch auth login
antioch scenario collect
```

To submit a genuinely new simulation check without selecting or interfering
with a shared machine:

```bash
antioch scenario run \
  --scenario asset_compatibility_smoke \
  --queue --no-stream --json
```

## Layout

```text
factory_sre/
├── contracts.py       # asset pins and workbench geometry contracts
├── scenarios.py       # thin Antioch evidence scenarios
└── sim/
    ├── assets.py      # policy-backed Go2 and pinned SO-101 loading
    └── scene.py       # deterministic two-workbench fixture
```

## Next simulation work

1. Mount the SO-101 and camera to the validated Go2 frame.
2. Add a short travel-stop-plant locomotion scenario.
3. Replace tag placeholders with detectable AprilTag textures.
4. Add local tag registration and visual servoing.
5. Model passive funnel alignment, magnetic capture, and contact/force limits.

Learned diffusion or world-model components should propose or rank trajectories;
deterministic identity, force, authorization, and recovery gates remain outside
the learned policy.
