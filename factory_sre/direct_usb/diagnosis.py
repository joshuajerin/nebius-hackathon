from factory_sre.contracts import FaultCode
from .contracts import Diagnosis, Snapshot


def diagnose(snapshot: Snapshot) -> Diagnosis:
    if not snapshot.safety_circuit_closed:
        return Diagnosis(FaultCode.SAFETY_CIRCUIT_OPEN, None, False)
    if not snapshot.vision_service_running:
        return Diagnosis(FaultCode.VISION_SERVICE_CRASHED, "restart_vision_service", True)
    if not snapshot.network_configuration_ok:
        return Diagnosis(FaultCode.NETWORK_INTERFACE_MISCONFIGURED, "restore_network_configuration", True)
    if not snapshot.camera_reachable:
        return Diagnosis(FaultCode.CAMERA_UNREACHABLE, "reload_camera_driver", True)
    if snapshot.vision_heartbeat:
        return Diagnosis(FaultCode.NONE, None, False)
    return Diagnosis(FaultCode.UNKNOWN, None, False)
