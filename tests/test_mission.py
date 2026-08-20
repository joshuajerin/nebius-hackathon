import pytest

from factory_sre.contracts import FaultCode, Incident, MissionState, TerminalReason, Transform3D, VISION_CELL
from factory_sre.hardware import DryRunArmAdapter, DryRunCameraAdapter, DryRunLocomotionAdapter
from factory_sre.mission import MissionController
from factory_sre.target.ledger import OperationLedger
from factory_sre.target.runtime import DiagnosticTargetRuntime
from factory_sre.transport import RuntimeDiagnosticTransport


def camera():
    return DryRunCameraAdapter({
        0.150: ((10, Transform3D((0, 1, 0.9))),),
        0.040: ((20, Transform3D((0, 0.25, 0))),),
    })


@pytest.fixture
def runtime(tmp_path):
    ledger = OperationLedger(tmp_path / "ledger.sqlite3")
    ledger.open()
    yield DiagnosticTargetRuntime(ledger)
    ledger.close()


@pytest.mark.parametrize("fault", (
    FaultCode.VISION_SERVICE_CRASHED,
    FaultCode.NETWORK_INTERFACE_MISCONFIGURED,
    FaultCode.CAMERA_UNREACHABLE,
))
def test_mission_recovers_three_faults(runtime, fault) -> None:
    transport = RuntimeDiagnosticTransport(runtime)
    transport.inject_fault(VISION_CELL.asset_id, fault)
    result = MissionController().run(
        Incident(f"m-{fault}", VISION_CELL.asset_id, fault.value), VISION_CELL,
        DryRunLocomotionAdapter(), DryRunArmAdapter(), camera(), transport,
    )
    assert result.success
    assert result.insertion_attempts == 1
    assert result.terminal_reason == TerminalReason.RECOVERED
    states = [event.state for event in result.events]
    assert all(state in states for state in MissionState if state not in {MissionState.ESCALATE, MissionState.EMERGENCY_STOP})
    assert not runtime.diagnostic_open


def test_force_limit_emergency_stops(runtime) -> None:
    transport = RuntimeDiagnosticTransport(runtime)
    transport.inject_fault(VISION_CELL.asset_id, FaultCode.VISION_SERVICE_CRASHED)
    arm, base = DryRunArmAdapter(measured_force_n=8.01), DryRunLocomotionAdapter()
    result = MissionController().run(
        Incident("force", VISION_CELL.asset_id, "fault"), VISION_CELL, base, arm, camera(), transport,
    )
    assert result.terminal_reason == TerminalReason.FORCE_LIMIT
    assert "emergency_stop" in arm.log and "emergency_stop" in base.log


def test_missing_fine_tag_never_inserts(runtime) -> None:
    transport = RuntimeDiagnosticTransport(runtime)
    transport.inject_fault(VISION_CELL.asset_id, FaultCode.VISION_SERVICE_CRASHED)
    result = MissionController().run(
        Incident("tag", VISION_CELL.asset_id, "fault"), VISION_CELL,
        DryRunLocomotionAdapter(), DryRunArmAdapter(),
        DryRunCameraAdapter({0.150: ((10, Transform3D((0, 1, 0.9))),)}), transport,
    )
    assert result.terminal_reason == TerminalReason.PERCEPTION_FAILED
    assert result.insertion_attempts == 0
