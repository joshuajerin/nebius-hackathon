import pytest

from factory_sre.contracts import FaultCode, Transform3D
from factory_sre.direct_usb.adapters import DryArm, DryCamera, DryLocomotion
from factory_sre.direct_usb.contracts import MissionState, TerminalReason, VISION_CELL
from factory_sre.direct_usb.mission import DirectUsbMissionController, FLOW
from factory_sre.direct_usb.runtime import DirectUsbTargetRuntime
from factory_sre.direct_usb.transport import InProcessDirectUsbTransport
from factory_sre.target.ledger import OperationLedger


def camera_with_both_tags() -> DryCamera:
    return DryCamera(
        {
            0.150: ((VISION_CELL.coarse_tag_id, Transform3D((2.0, 0.8, 0.9))),),
            0.040: ((VISION_CELL.fine_tag_id, Transform3D((0.45, 0.0, 0.25))),),
        }
    )


@pytest.mark.parametrize(
    "fault",
    (
        FaultCode.VISION_SERVICE_CRASHED,
        FaultCode.NETWORK_INTERFACE_MISCONFIGURED,
        FaultCode.CAMERA_UNREACHABLE,
    ),
)
def test_direct_usb_mission_recovers_all_three_faults(tmp_path, fault) -> None:
    ledger = OperationLedger(tmp_path / f"{fault.value}.sqlite3")
    ledger.open()
    try:
        runtime = DirectUsbTargetRuntime(ledger)
        runtime.inject_fault(fault)
        arm = DryArm(VISION_CELL.asset_id)
        result = DirectUsbMissionController().run(
            f"mission-{fault.value}",
            VISION_CELL,
            DryLocomotion(),
            arm,
            camera_with_both_tags(),
            InProcessDirectUsbTransport(runtime),
        )
        assert result.success
        assert result.terminal_state == MissionState.REPORT
        assert result.terminal_reason == TerminalReason.RECOVERED
        assert result.insertion_attempts == 1
        assert [event.state for event in result.events] == list(FLOW)
        assert arm.last_target is not None
        assert arm.last_target.translation == pytest.approx((0.39, 0.0, 0.25))
        assert arm.retreated
        assert not runtime.diagnostic_open
    finally:
        ledger.close()


def test_direct_usb_mission_aborts_on_tag_loss_without_insertion(tmp_path) -> None:
    ledger = OperationLedger(tmp_path / "tag-loss.sqlite3")
    ledger.open()
    try:
        runtime = DirectUsbTargetRuntime(ledger)
        runtime.inject_fault(FaultCode.VISION_SERVICE_CRASHED)
        arm = DryArm(VISION_CELL.asset_id)
        result = DirectUsbMissionController().run(
            "mission-tag-loss",
            VISION_CELL,
            DryLocomotion(),
            arm,
            DryCamera({0.150: ((VISION_CELL.coarse_tag_id, Transform3D((2.0, 0.8, 0.9))),)}),
            InProcessDirectUsbTransport(runtime),
        )
        assert not result.success
        assert result.terminal_reason == TerminalReason.PERCEPTION_FAILED
        assert result.insertion_attempts == 0
        assert ledger.count() == 0
        assert not runtime.diagnostic_open
    finally:
        ledger.close()


def test_direct_usb_mission_emergency_stops_on_excess_force(tmp_path) -> None:
    ledger = OperationLedger(tmp_path / "force.sqlite3")
    ledger.open()
    try:
        runtime = DirectUsbTargetRuntime(ledger)
        runtime.inject_fault(FaultCode.VISION_SERVICE_CRASHED)
        base = DryLocomotion()
        arm = DryArm(VISION_CELL.asset_id, axial_force_n=9.0)
        result = DirectUsbMissionController(maximum_force_n=8.0).run(
            "mission-force",
            VISION_CELL,
            base,
            arm,
            camera_with_both_tags(),
            InProcessDirectUsbTransport(runtime),
        )
        assert result.terminal_state == MissionState.EMERGENCY_STOP
        assert result.terminal_reason == TerminalReason.FORCE_LIMIT
        assert base.stopped and arm.stopped
        assert ledger.count() == 0
    finally:
        ledger.close()


def test_direct_usb_mission_escalates_safety_fault_without_mutation(tmp_path) -> None:
    ledger = OperationLedger(tmp_path / "safety.sqlite3")
    ledger.open()
    try:
        runtime = DirectUsbTargetRuntime(ledger)
        runtime.inject_fault(FaultCode.SAFETY_CIRCUIT_OPEN)
        result = DirectUsbMissionController().run(
            "mission-safety",
            VISION_CELL,
            DryLocomotion(),
            DryArm(VISION_CELL.asset_id),
            camera_with_both_tags(),
            InProcessDirectUsbTransport(runtime),
        )
        assert not result.success
        assert result.terminal_reason == TerminalReason.UNSUPPORTED_DIAGNOSIS
        assert ledger.count() == 0
        assert not runtime.diagnostic_open
    finally:
        ledger.close()
