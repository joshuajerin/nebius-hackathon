# Factory SRE

> **Current approved implementation:** the two-adjacent-workbench, 120 mm AprilTag, magnetic service-dock design is documented in [`docs/approved-factory-sre.md`](docs/approved-factory-sre.md). It uses exactly two Antioch services and permits only `restart_vision_service` autonomously. The older one-workcell/direct-USB notes below are preserved for the parallel prototype; they are not the certification target.

A mobile out-of-band first responder for factory automation failures. The v1 scene is one Unitree Go2 carrying an Antioch SO-101 arm, approaching an UR5e vision cell, localizing from two AprilTags, inserting a direct USB-C diagnostic connection, and recovering one of three whitelisted faults.

## What is implemented

- Exact Isaac Sim 6.0.1 asset manifest for Go2, UR5e, warehouse, D455, SICK InspectorP61x, SICK nanoScan3, and SICK Inspector83x.
- Antioch `so101_antioch` pin at `1.3.2` and remote composition code that removes the arm articulation root and mounts it to the Go2 with a fixed joint.
- One workcell with a 150 mm coarse tag, 40 mm fine tag, calibrated port-to-tag transform, and direct USB-C collision geometry.
- Metric AprilTag pose estimation using `pupil-apriltags` and WXYZ transform composition.
- Occupancy-grid A* with obstacle inflation and diagonal corner-cut prevention.
- Transport-independent mission controller covering `DISPATCH → NAVIGATE → STABILIZE → LOCALIZE → APPROACH → INSERT → ENUMERATE → DIAGNOSE → RECOVER → VERIFY → DISCONNECT → REPORT`.
- Dock-gated diagnostic target with a real TCP listener, lease/epoch protection, crash-reconcilable SQLite ledger, idempotent operations, and three recoveries:
  - restart crashed vision service;
  - restore golden network configuration;
  - reload a failed camera driver.
- Fail-closed escalation for safety faults, excessive insertion force, missing tags, stale links, unknown operations, and failed verification.

## Asset source of truth

Stock USDs are referenced from the remote Isaac asset root and are not copied into this repository because they carry relative mesh and material dependencies. The custom mount, USB-C tool and panel must come from measured physical CAD before physical insertion trials; the current scene authors explicit collision proxies and labels that status in scenario results.

The canonical manifest is `factory_sre/contracts.py`. Remote composition and articulation validation are in `factory_sre/sim/assets.py`.

## Local verification

```bash
cd /Users/joshuajerin/Desktop/jarvis/hackathon-nebius
uv sync --extra dev
uv run pytest -q
uv run antioch scenario collect --json
```

The test suite exercises all three recovery missions, a real two-port target process, AprilTag detection, transform math, path planning, operation idempotency, link epochs, and safety aborts.

## Antioch runs

After Antioch enables the organization tenant:

```bash
uv run antioch auth whoami
uv run antioch assets show so101_antioch --json
uv run antioch scenario run --scenario asset_compatibility_smoke --no-stream
uv run antioch scenario run --scenario factory_workcell_scene_smoke
uv run antioch scenario run --scenario diagnostic_recovery_control_plane --no-stream
uv run antioch suite run smoke --no-stream
uv run antioch suite run hero --no-stream
```

For every remote run, inspect the recorded checks and artifacts with `antioch scenario show <run-id> --json`, `antioch scenario logs <run-id>`, and `antioch scenario download <run-id>`.

## Physical deployment gates

This repository deliberately does not claim the robot is ready for unsupervised hardware insertion yet. Before commanding motors:

1. Replace the collision proxies with measured mount, compliant USB-C tool, cable and panel CAD.
2. Duplicate `config/calibration.example.json`, measure every transform and P61x intrinsic, and set `calibrated=true` only after independent verification.
3. Configure the target as a USB **device**, not a second USB host; see `docs/direct-usb-c.md`.
4. Implement the Unitree and real SO-101 adapters behind the existing protocols and prove emergency stop independently.
5. Validate the stock Go2 policy with the mounted payload, then fine-tune it in Isaac Lab if stability or tracking gates fail.
6. Pass 20 payload navigation runs and 27/30 guarded insertion trials before enabling autonomous recovery.

## Current platform blocker

Local discovery and tests work, but Antioch currently responds that the configured organization tenant is not enabled. Therefore the stock USD references, SO-101 catalog object, articulation count, live Go2 policy and rendered scene have not yet been validated on the remote Isaac Sim 6.0.1 machine. The smoke scenarios are the fail-closed compatibility gate once access is restored.
