from datetime import UTC, datetime

import pytest

from factory_sre.contracts import DiagnosticSession, FaultCode, OperationStatus
from factory_sre.direct_usb.contracts import VISION_CELL
from factory_sre.direct_usb.runtime import DirectUsbRejected, DirectUsbTargetRuntime
from factory_sre.target.ledger import OperationLedger


@pytest.fixture
def runtime(tmp_path):
    ledger = OperationLedger(tmp_path / "direct-usb.sqlite3")
    ledger.open()
    target = DirectUsbTargetRuntime(ledger)
    try:
        yield target, ledger
    finally:
        ledger.close()


@pytest.mark.parametrize(
    ("fault", "action"),
    (
        (FaultCode.VISION_SERVICE_CRASHED, "restart_vision_service"),
        (FaultCode.NETWORK_INTERFACE_MISCONFIGURED, "restore_network_configuration"),
        (FaultCode.CAMERA_UNREACHABLE, "reload_camera_driver"),
    ),
)
def test_direct_usb_runtime_recovers_each_whitelisted_fault(runtime, fault, action) -> None:
    target, ledger = runtime
    target.inject_fault(fault)
    session = target.enumerate("mission-1", VISION_CELL.asset_id)
    assert target.execute(session, action) == OperationStatus.SUCCEEDED
    assert target.snapshot(session).vision_heartbeat
    assert ledger.count() == 1
    target.disconnect(session)
    assert not target.diagnostic_open


def test_direct_usb_runtime_rejects_wrong_identity_and_stale_epoch(runtime) -> None:
    target, _ = runtime
    with pytest.raises(DirectUsbRejected, match="identity mismatch"):
        target.enumerate("mission-1", "another-cell")

    session = target.enumerate("mission-1", VISION_CELL.asset_id)
    stale = DiagnosticSession(
        mission_id=session.mission_id,
        asset_id=session.asset_id,
        link_epoch=session.link_epoch - 1,
        diagnostic_endpoint=session.diagnostic_endpoint,
        lease_expiry=datetime.now(UTC),
        next_sequence=1,
    )
    with pytest.raises(DirectUsbRejected, match="stale"):
        target.snapshot(stale)


def test_direct_usb_runtime_rejects_non_whitelisted_mutation(runtime) -> None:
    target, _ = runtime
    session = target.enumerate("mission-1", VISION_CELL.asset_id)
    with pytest.raises(DirectUsbRejected, match="not authorized"):
        target.execute(session, "bypass_safety_circuit")

