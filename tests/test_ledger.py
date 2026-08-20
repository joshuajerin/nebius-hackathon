import sqlite3

from factory_sre.contracts import OperationStatus
from factory_sre.target.ledger import OperationLedger


def test_pending_operations_reconcile_to_unknown_and_are_not_replayed(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    ledger = OperationLedger(path)
    ledger.open()
    ledger.begin(idempotency_key="key", mission_id="m", asset_id="vision-cell-37", link_epoch=1, sequence=1, command="restart_vision_service")
    ledger.close()
    reopened = OperationLedger(path)
    reopened.open()
    assert reopened.get("key").status == OperationStatus.UNKNOWN
    reopened.close()


def test_integrity_check_fails_closed_on_corrupt_database(tmp_path) -> None:
    path = tmp_path / "ledger.sqlite3"
    path.write_bytes(b"not a sqlite database")
    ledger = OperationLedger(path)
    try:
        ledger.open()
    except Exception:
        pass
    assert not ledger.available
