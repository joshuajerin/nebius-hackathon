# Calibration and safety runbook

This procedure is required before any physical motor-enabled run. `config/factory_sre_calibration.template.json` remains `calibrated=false` until every measurement below is complete and independently reviewed.

## Required hardware

- Measured Go2-to-SO-101 mount and service-tool CAD.
- Wrist or tool-axis force measurement capable of enforcing the calibrated 8 N ceiling.
- Independent hardware emergency stop that removes actuator command authority.
- Guarded magnetic service dock. Direct free-space USB-C or RJ45 insertion is outside this release.
- Printed `tag36h11` IDs 10 and 20 at 120 mm, rigidly attached to their panels.

## Coordinate-frame calibration

1. Lock the Go2 in a powered-off service fixture and measure `body → arm_base` from mount datums. Record WXYZ quaternions.
2. Calibrate wrist-camera intrinsics from at least 20 views spanning the field of view. Reject a calibration with reprojection error above the camera team's agreed threshold.
3. Solve `wrist_camera → service_tip` with a fixed target and at least 15 arm poses. Independently check the result with held-out poses.
4. Survey each panel's `tag → magnetic_dock` transform. The approved nominal translation is exactly `(-0.30, 0, 0)` metres. Do not compensate a misplaced dock in code; repair the panel or issue a new signed calibration.
5. Recompute the complete chain and measure the service-tip error against the dock at ten held-out body poses. The simulation certification gate is 5 mm and 3°; physical deployment should use a tighter internal target to preserve margin.
6. Copy the template to a site-specific file, enter measured values, review it, then set `calibrated=true`.

## Force-limit calibration

Run all three cases at the exact physics/control rate and with the production tool compliance:

1. Nominal centered docking: measure peak and steady axial force.
2. Maximum allowed lateral/angular misalignment: verify graceful rejection or engagement below the ceiling.
3. Hard stop: verify the independent stop path fires before force exceeds the calibrated 8 N limit or penetration exceeds 10 mm.

The 8 N value is not a universal USB-C or robot safety rating. It is the approved software ceiling for this demo and must be validated against the actual magnetic tool, sensor accuracy, latency, and workbench compliance.

## Pre-run checklist

- Clear people and loose objects from the test volume.
- Confirm guards isolate moving tabletop machinery from the service-panel volume.
- Verify expected incident asset, tag ID, and approach pose.
- Verify calibration file hash and adapter versions in the evidence manifest.
- Test base and arm emergency-stop paths separately.
- Confirm the diagnostic listener is closed before contact.
- Start with human approval required for every recovery.

## Immediate aborts

Abort and leave the diagnostic listener closed for wrong identity, tag loss, unstable base, excess roll/pitch, force above 8 N, penetration above 10 mm, stale link epoch, unavailable/corrupt ledger, unknown mutation outcome, safety-circuit fault, or failed health verification. Never automatically bypass or reset a safety circuit.
