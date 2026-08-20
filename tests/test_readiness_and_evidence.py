from __future__ import annotations

from factory_sre.contracts import Transform3D, WORKBENCH_A
from factory_sre.evidence import TelemetrySchedule, build_evidence_manifest
from factory_sre.hardware import DryRunArmAdapter, DryRunBaseAdapter, DryRunCameraAdapter
from factory_sre.readiness import HardwareReadinessInputs, assess_hardware_readiness, run_no_motor_dry_run


def test_template_fails_closed_for_physical_readiness() -> None:
    report = assess_hardware_readiness(
        HardwareReadinessInputs(
            calibration_path="config/factory_sre_calibration.template.json",
            measured_mount_cad_path="assets/custom/missing-mount.step",
            force_sensor_present=False,
            emergency_stop_verified=False,
            guarded_motion_verified=False,
            real_base_adapter_present=False,
            real_arm_adapter_present=False,
        )
    )
    assert not report.ready
    assert all(not passed for _, passed, _ in report.checks)


def test_no_motor_dry_run_computes_the_300_mm_left_dock() -> None:
    tag = Transform3D((1.8, 0.64, 0.88))
    report = run_no_motor_dry_run(
        WORKBENCH_A,
        DryRunBaseAdapter(),
        DryRunArmAdapter(asset_id=WORKBENCH_A.asset_id),
        DryRunCameraAdapter({0.120: ((WORKBENCH_A.tag_id, tag),)}),
    )
    assert report.passed
    assert not report.motors_enabled
    assert report.computed_dock is not None
    assert report.computed_dock.translation == (1.5, 0.64, 0.88)


def test_evidence_manifest_freezes_exact_versions_and_rates() -> None:
    manifest = build_evidence_manifest(".", "config/factory_sre_calibration.template.json")
    assert manifest.antioch_version == "0.3.63"
    assert manifest.isaac_sim_version == "6.0.1"
    assert manifest.so101_asset == "so101_antioch@1.3.2"
    assert manifest.schedule == TelemetrySchedule(20, 10, 5)
    assert len(manifest.project_file_sha256) == 64
    assert len(manifest.calibration_sha256 or "") == 64
