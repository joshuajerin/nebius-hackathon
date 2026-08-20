import pytest

from factory_sre.contracts import FaultCode, OperationStatus, VISION_CELL
from factory_sre.target.ledger import OperationLedger
from factory_sre.target.runtime import DiagnosticTargetRuntime, TargetRejected


def runtime(tmp_path):
    ledger = OperationLedger(tmp_path / "ledger.sqlite3")
    ledger.open()
    return DiagnosticTargetRuntime(ledger), ledger


def test_stale_epoch_rejected(tmp_path) -> None:
    target, ledger = runtime(tmp_path)
    target.inject_fault(VISION_CELL.asset_id, FaultCode.VISION_SERVICE_CRASHED)
    old = target.dock("m1", VISION_CELL.asset_id, "usb://target")
    target.disconnect()
    target.dock("m2", VISION_CELL.asset_id, "usb://target")
    with pytest.raises(TargetRejected):
        target.snapshot(mission_id=old.mission_id, link_epoch=old.link_epoch)
    ledger.close()


def test_operation_idempotent(tmp_path) -> None:
    target, ledger = runtime(tmp_path)
    target.inject_fault(VISION_CELL.asset_id, FaultCode.VISION_SERVICE_CRASHED)
    session = target.dock("m1", VISION_CELL.asset_id, "usb://target")
    kwargs = dict(
        mission_id="m1", asset_id=VISION_CELL.asset_id, link_epoch=session.link_epoch,
        sequence=1, idempotency_key="m1:restart", command="restart_vision_service",
    )
    assert target.execute(**kwargs).status == OperationStatus.SUCCEEDED
    assert target.execute(**kwargs).status == OperationStatus.SUCCEEDED
    assert ledger.count("restart_vision_service") == 1
    ledger.close()
