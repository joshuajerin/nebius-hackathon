"""Deterministic diagnosis rules. This module never executes a mutation."""

from __future__ import annotations

from .contracts import Diagnosis, DiagnosticSnapshot, FaultCode


def diagnose(snapshot: DiagnosticSnapshot) -> Diagnosis:
    if not snapshot.safety_circuit_closed:
        return Diagnosis(FaultCode.SAFETY_CIRCUIT_OPEN, 1.0, ("safety circuit is open",), None, False)
    if not snapshot.usb_peripheral_present:
        return Diagnosis(FaultCode.USB_PERIPHERAL_MISSING, 0.98, ("required USB peripheral is absent",), None, False)
    if not snapshot.vision_service_running and snapshot.camera_reachable and snapshot.network_configuration_ok:
        return Diagnosis(
            FaultCode.VISION_SERVICE_CRASHED,
            0.99,
            ("vision process absent", "camera reachable", "network configuration valid"),
            "restart_vision_service",
            True,
        )
    if not snapshot.network_configuration_ok:
        return Diagnosis(
            FaultCode.NETWORK_INTERFACE_MISCONFIGURED,
            0.95,
            ("network configuration differs from golden manifest",),
            "restore_network_configuration",
            False,
        )
    if not snapshot.camera_reachable:
        return Diagnosis(
            FaultCode.CAMERA_UNREACHABLE,
            0.92,
            ("camera probe failed", "camera driver is reloadable for this asset"),
            "reload_camera_driver",
            False,
        )
    if snapshot.vision_service_running and not snapshot.vision_heartbeat:
        return Diagnosis(
            FaultCode.VISION_HEALTHCHECK_FAILED,
            0.90,
            ("vision process is running", "vision heartbeat is absent"),
            None,
            False,
        )
    if snapshot.vision_service_running and snapshot.vision_heartbeat:
        return Diagnosis(FaultCode.NONE, 1.0, ("all supported probes healthy",), None, False)
    return Diagnosis(FaultCode.UNKNOWN, 0.0, ("no deterministic rule matched",), None, False)
