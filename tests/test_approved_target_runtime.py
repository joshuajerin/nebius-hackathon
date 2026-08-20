from __future__ import annotations

from time import sleep

import pytest

from factory_sre.contracts import FaultCode, OperationStatus, WORKBENCH_A
from factory_sre.target.ledger import LedgerUnavailable, OperationLedger
from factory_sre.target.runtime import DiagnosticTargetRuntime, TargetRejected


@pytest.fixture
def target(tmp_path):
    ledger = OperationLedger(tmp_path / "approved-ledger.sqlite3")
    ledger.open()
    runtime = DiagnosticTargetRuntime(ledger, lease_seconds=0.03)
    runtime.inject_fault(WORKBENCH_A.asset_id, FaultCode.VISION_SERVICE_CRASHED)
    yield runtime, ledger
    ledger.close()


def test_link_is_closed_before_contact_after_disconnect_and_after_lease(target) -> None:
    runtime, _ = target
    assert not runtime.diagnostic_open
    session = runtime.dock("mission-a", WORKBENCH_A.asset_id, "http://127.0.0.1:8766")
    assert runtime.diagnostic_open
    runtime.disconnect(mission_id=session.mission_id, link_epoch=session.link_epoch)
    assert not runtime.diagnostic_open
    runtime.dock("mission-b", WORKBENCH_A.asset_id, "http://127.0.0.1:8766")
    sleep(0.04)
    assert not runtime.diagnostic_open


def test_new_contact_increments_epoch_and_rejects_delayed_request(target) -> None:
    runtime, _ = target
    old = runtime.dock("old", WORKBENCH_A.asset_id, "dock://target")
    runtime.disconnect(mission_id=old.mission_id, link_epoch=old.link_epoch)
    new = runtime.dock("new", WORKBENCH_A.asset_id, "dock://target")
    assert new.link_epoch == old.link_epoch + 1
    with pytest.raises(TargetRejected, match="mission identity|stale"):
        runtime.snapshot(mission_id=old.mission_id, link_epoch=old.link_epoch)


def test_only_hero_restart_is_authorized_and_duplicate_is_idempotent(target) -> None:
    runtime, ledger = target
    session = runtime.dock("mission", WORKBENCH_A.asset_id, "dock://target")
    kwargs = {
        "mission_id": session.mission_id,
        "asset_id": session.asset_id,
        "link_epoch": session.link_epoch,
        "sequence": 1,
        "idempotency_key": "mission:1:restart",
        "command": "restart_vision_service",
    }
    assert runtime.execute(**kwargs).status == OperationStatus.SUCCEEDED
    assert runtime.execute(**kwargs).status == OperationStatus.SUCCEEDED
    assert ledger.count("restart_vision_service") == 1
    with pytest.raises(TargetRejected, match="not authorized"):
        runtime.execute(**(kwargs | {"sequence": 2, "idempotency_key": "bad", "command": "restore_network_configuration"}))
    assert ledger.count() == 1


def test_unavailable_ledger_fails_closed_without_session(tmp_path) -> None:
    ledger = OperationLedger(tmp_path / "never-opened.sqlite3")
    runtime = DiagnosticTargetRuntime(ledger)
    runtime.inject_fault(WORKBENCH_A.asset_id, FaultCode.VISION_SERVICE_CRASHED)
    with pytest.raises(LedgerUnavailable):
        runtime.dock("mission", WORKBENCH_A.asset_id, "dock://target")
    assert not runtime.diagnostic_open
