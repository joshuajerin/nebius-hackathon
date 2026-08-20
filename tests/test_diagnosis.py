import pytest

from factory_sre.contracts import DiagnosticSnapshot, FaultCode
from factory_sre.diagnosis import diagnose


def snapshot(**overrides) -> DiagnosticSnapshot:
    values = dict(
        asset_id="vision-cell-37", vision_service_running=True, vision_heartbeat=True,
        camera_reachable=True, network_configuration_ok=True, safety_circuit_closed=True,
    )
    values.update(overrides)
    return DiagnosticSnapshot(**values)


@pytest.mark.parametrize(("overrides", "fault", "action"), (
    ({"vision_service_running": False, "vision_heartbeat": False}, FaultCode.VISION_SERVICE_CRASHED, "restart_vision_service"),
    ({"network_configuration_ok": False, "vision_heartbeat": False}, FaultCode.NETWORK_INTERFACE_MISCONFIGURED, "restore_network_configuration"),
    ({"camera_reachable": False, "vision_heartbeat": False}, FaultCode.CAMERA_UNREACHABLE, "reload_camera_driver"),
))
def test_supported_faults_are_autonomous(overrides, fault, action) -> None:
    result = diagnose(snapshot(**overrides))
    assert (result.fault, result.recommended_action, result.autonomous) == (fault, action, True)


def test_safety_fault_escalates() -> None:
    result = diagnose(snapshot(safety_circuit_closed=False))
    assert result.fault == FaultCode.SAFETY_CIRCUIT_OPEN
    assert not result.autonomous
