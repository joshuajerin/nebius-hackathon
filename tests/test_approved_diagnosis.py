from __future__ import annotations

import pytest

from factory_sre.contracts import DiagnosticSnapshot, FaultCode
from factory_sre.diagnosis import diagnose


def snapshot(**overrides):
    values = {
        "asset_id": "workbench-b",
        "vision_service_running": True,
        "vision_heartbeat": True,
        "camera_reachable": True,
        "network_configuration_ok": True,
        "usb_peripheral_present": True,
        "safety_circuit_closed": True,
    }
    values.update(overrides)
    return DiagnosticSnapshot(**values)


@pytest.mark.parametrize(
    ("overrides", "fault", "action", "autonomous"),
    (
        ({"safety_circuit_closed": False}, FaultCode.SAFETY_CIRCUIT_OPEN, None, False),
        ({"usb_peripheral_present": False}, FaultCode.USB_PERIPHERAL_MISSING, None, False),
        ({"vision_service_running": False, "vision_heartbeat": False}, FaultCode.VISION_SERVICE_CRASHED, "restart_vision_service", True),
        ({"network_configuration_ok": False, "vision_heartbeat": False}, FaultCode.NETWORK_INTERFACE_MISCONFIGURED, "restore_network_configuration", False),
        ({"camera_reachable": False, "vision_heartbeat": False}, FaultCode.CAMERA_UNREACHABLE, "reload_camera_driver", False),
        ({"vision_heartbeat": False}, FaultCode.VISION_HEALTHCHECK_FAILED, None, False),
    ),
)
def test_six_deterministic_rules_and_hero_only_authorization(overrides, fault, action, autonomous) -> None:
    result = diagnose(snapshot(**overrides))
    assert result.fault == fault
    assert result.recommended_action == action
    assert result.autonomous is autonomous


def test_healthy_snapshot_requires_no_action() -> None:
    result = diagnose(snapshot())
    assert result.fault == FaultCode.NONE
    assert result.recommended_action is None
    assert not result.autonomous
