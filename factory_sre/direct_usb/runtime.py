"""In-process emulator for the dedicated USB device-mode diagnostic endpoint."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

from factory_sre.contracts import DiagnosticSession, FaultCode, OperationStatus
from factory_sre.target.ledger import LedgerUnavailable, OperationLedger

from .contracts import Snapshot, VISION_CELL


class DirectUsbRejected(RuntimeError):
    """The target rejected a stale, unsafe, or unauthorized operation."""


class DirectUsbTargetRuntime:
    """Models enumeration, leases, logs/config state, and whitelisted repairs."""

    DIAGNOSTIC_ENDPOINT = "usb://factory-sre/cdc-ncm+serial"

    def __init__(self, ledger: OperationLedger, *, lease_seconds: float = 10.0) -> None:
        self.ledger = ledger
        self.lease_seconds = lease_seconds
        self._lock = threading.RLock()
        self._link_epoch = 0
        self._next_sequence = 1
        self._session: DiagnosticSession | None = None
        self._snapshot = self._healthy_snapshot()

    @staticmethod
    def _healthy_snapshot() -> Snapshot:
        return Snapshot(
            asset_id=VISION_CELL.asset_id,
            vision_service_running=True,
            vision_heartbeat=True,
            camera_reachable=True,
            network_configuration_ok=True,
            safety_circuit_closed=True,
        )

    @property
    def diagnostic_open(self) -> bool:
        with self._lock:
            return (
                self._session is not None
                and datetime.now(UTC) < self._session.lease_expiry
                and self.ledger.available
            )

    @property
    def link_epoch(self) -> int:
        return self._link_epoch

    def inject_fault(self, fault: FaultCode) -> None:
        with self._lock:
            values = {
                "asset_id": VISION_CELL.asset_id,
                "vision_service_running": True,
                "vision_heartbeat": True,
                "camera_reachable": True,
                "network_configuration_ok": True,
                "safety_circuit_closed": True,
                "error_code": fault.value if fault != FaultCode.NONE else None,
            }
            if fault == FaultCode.VISION_SERVICE_CRASHED:
                values.update(vision_service_running=False, vision_heartbeat=False)
            elif fault == FaultCode.NETWORK_INTERFACE_MISCONFIGURED:
                values.update(network_configuration_ok=False, vision_heartbeat=False)
            elif fault == FaultCode.CAMERA_UNREACHABLE:
                values.update(camera_reachable=False, vision_heartbeat=False)
            elif fault == FaultCode.SAFETY_CIRCUIT_OPEN:
                values.update(safety_circuit_closed=False, vision_heartbeat=False)
            elif fault != FaultCode.NONE:
                raise ValueError(f"unsupported injected fault: {fault}")
            self._snapshot = Snapshot(**values)

    def enumerate(self, mission_id: str, asset_id: str) -> DiagnosticSession:
        with self._lock:
            if not self.ledger.available:
                raise LedgerUnavailable("USB enumeration denied while the command ledger is unavailable")
            if asset_id != self._snapshot.asset_id:
                raise DirectUsbRejected(
                    f"USB identity mismatch: expected {self._snapshot.asset_id}, got {asset_id}"
                )
            self._link_epoch += 1
            self._next_sequence = 1
            self._session = DiagnosticSession(
                mission_id=mission_id,
                asset_id=asset_id,
                link_epoch=self._link_epoch,
                diagnostic_endpoint=self.DIAGNOSTIC_ENDPOINT,
                lease_expiry=datetime.now(UTC) + timedelta(seconds=self.lease_seconds),
                next_sequence=1,
            )
            return self._session

    def snapshot(self, session: DiagnosticSession) -> Snapshot:
        with self._lock:
            self._require_session(session)
            return self._snapshot

    def execute(self, session: DiagnosticSession, action: str) -> OperationStatus:
        with self._lock:
            active = self._require_session(session)
            if action not in VISION_CELL.permitted_actions:
                raise DirectUsbRejected(f"action is not authorized: {action}")

            sequence = self._next_sequence
            key = f"{active.mission_id}:{active.link_epoch}:{sequence}:{action}"
            existing = self.ledger.get(key)
            if existing is not None:
                if existing.status == OperationStatus.UNKNOWN:
                    raise DirectUsbRejected("an operation with an uncertain outcome cannot be retried automatically")
                return existing.status

            self.ledger.begin(
                idempotency_key=key,
                mission_id=active.mission_id,
                asset_id=active.asset_id,
                link_epoch=active.link_epoch,
                sequence=sequence,
                command=action,
            )
            try:
                values = {
                    "asset_id": self._snapshot.asset_id,
                    "vision_service_running": self._snapshot.vision_service_running,
                    "vision_heartbeat": self._snapshot.vision_heartbeat,
                    "camera_reachable": self._snapshot.camera_reachable,
                    "network_configuration_ok": self._snapshot.network_configuration_ok,
                    "safety_circuit_closed": self._snapshot.safety_circuit_closed,
                    "error_code": None,
                }
                if action == "restart_vision_service":
                    values["vision_service_running"] = True
                elif action == "restore_network_configuration":
                    values["network_configuration_ok"] = True
                elif action == "reload_camera_driver":
                    values["camera_reachable"] = True
                values["vision_heartbeat"] = bool(
                    values["vision_service_running"]
                    and values["camera_reachable"]
                    and values["network_configuration_ok"]
                    and values["safety_circuit_closed"]
                )
                self._snapshot = Snapshot(**values)
                recovered = self._snapshot.vision_heartbeat
                status = OperationStatus.SUCCEEDED if recovered else OperationStatus.FAILED
                self.ledger.finish(key, status, {"vision_healthy": recovered})
                self._next_sequence += 1
                return status
            except BaseException:
                try:
                    self.ledger.finish(key, OperationStatus.UNKNOWN, {"error": "mutation result uncertain"})
                finally:
                    self._session = None
                raise

    def disconnect(self, session: DiagnosticSession) -> None:
        with self._lock:
            self._require_session(session)
            self._session = None

    def _require_session(self, session: DiagnosticSession) -> DiagnosticSession:
        if not self.diagnostic_open or self._session is None:
            raise DirectUsbRejected("USB diagnostic link is closed or its lease expired")
        active = self._session
        if session.mission_id != active.mission_id:
            raise DirectUsbRejected("mission identity mismatch")
        if session.asset_id != active.asset_id:
            raise DirectUsbRejected("asset identity mismatch")
        if session.link_epoch != active.link_epoch:
            raise DirectUsbRejected("stale USB link epoch")
        return active

