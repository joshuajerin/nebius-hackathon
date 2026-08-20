"""Fail-closed hardware handoff and no-motor dry-run verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .calibration import load_calibration
from .contracts import ArmAdapter, BaseAdapter, CameraAdapter, Transform3D, WorkbenchManifest


@dataclass(frozen=True, slots=True)
class HardwareReadinessInputs:
    calibration_path: str
    measured_mount_cad_path: str
    force_sensor_present: bool
    emergency_stop_verified: bool
    guarded_motion_verified: bool
    real_base_adapter_present: bool
    real_arm_adapter_present: bool


@dataclass(frozen=True, slots=True)
class HardwareReadinessReport:
    ready: bool
    checks: tuple[tuple[str, bool, str], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DryRunReport:
    passed: bool
    actions: tuple[str, ...]
    expected_tag_id: int
    computed_dock: Transform3D | None
    motors_enabled: bool


def assess_hardware_readiness(inputs: HardwareReadinessInputs) -> HardwareReadinessReport:
    checks: list[tuple[str, bool, str]] = []
    try:
        load_calibration(inputs.calibration_path)
        checks.append(("approved calibration loads", True, inputs.calibration_path))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        checks.append(("approved calibration loads", False, f"{type(exc).__name__}: {exc}"))
    cad = Path(inputs.measured_mount_cad_path)
    checks.extend(
        (
            ("measured mount CAD exists", cad.is_file(), str(cad)),
            ("force sensor is installed", inputs.force_sensor_present, "required for the 8 N limit"),
            ("independent emergency stop is verified", inputs.emergency_stop_verified, "hardware checklist sign-off"),
            ("guarded low-speed motion is verified", inputs.guarded_motion_verified, "no autonomous insertion before this"),
            ("real Go2 adapter is present", inputs.real_base_adapter_present, "simulation adapter does not qualify"),
            ("real SO-101 adapter is present", inputs.real_arm_adapter_present, "simulation adapter does not qualify"),
        )
    )
    return HardwareReadinessReport(all(passed for _, passed, _ in checks), tuple(checks))


def run_no_motor_dry_run(
    manifest: WorkbenchManifest,
    base: BaseAdapter,
    arm: ArmAdapter,
    camera: CameraAdapter,
    *,
    motors_enabled: bool = False,
) -> DryRunReport:
    """Exercise adapter contracts while refusing any motor-enabled invocation."""
    if motors_enabled:
        raise RuntimeError("hardware dry-run refuses motors_enabled=true")
    actions: list[str] = []
    actions.append("adapter-contracts-loaded")
    detections = camera.detect_tags(manifest.tag_family, manifest.tag_size_m)
    actions.append("camera-tag-query")
    tag = next((pose for tag_id, pose in detections if tag_id == manifest.tag_id), None)
    if tag is None:
        return DryRunReport(False, tuple(actions + ["expected-tag-missing"]), manifest.tag_id, None, False)
    dock = tag.compose(manifest.tag_to_dock)
    actions.extend(("expected-tag-confirmed", "dock-transform-computed"))
    # E-stop methods are required contract probes. Dry-run adapters only log;
    # physical implementations must be configured in no-motor mode upstream.
    arm.emergency_stop()
    base.emergency_stop()
    actions.append("emergency-stop-contract-invoked")
    return DryRunReport(True, tuple(actions), manifest.tag_id, dock, False)
