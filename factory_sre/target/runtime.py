"""Diagnostic target state, dock leases, and authorized mutations."""

from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from ..contracts import DiagnosticSession, DiagnosticSnapshot, FaultCode, OperationStatus
from .ledger import LedgerUnavailable, Operation, OperationLedger


class TargetRejected(RuntimeError):
    pass


class DiagnosticTargetRuntime:
    # The implementation below retains deferred recovery handlers for later
    # releases, but this hero release authorizes exactly one mutation.
    HERO_ALLOWED_MUTATIONS = frozenset({"restart_vision_service"})
    ALLOWED_MUTATIONS = frozenset(
        {"restart_vision_service", "restore_network_configuration", "reload_camera_driver"}
    )

    def __init__(self, ledger: OperationLedger, *, lease_seconds: float = 10.0) -> None:
        self.ledger = ledger
        self.lease_seconds = lease_seconds
        self._lock = threading.RLock()
        self._session: DiagnosticSession | None = None
        self._link_epoch = 0
        self._next_sequence = 1
        self._snapshot = DiagnosticSnapshot(
            asset_id="workbench-b",
            vision_service_running=True,
            vision_heartbeat=True,
            camera_reachable=True,
            network_configuration_ok=True,
            usb_peripheral_present=True,
            safety_circuit_closed=True,
        )

    @property
    def healthy(self) -> bool:
        return self.ledger.available

    @property
    def diagnostic_open(self) -> bool:
        with self._lock:
            return self._session is not None and datetime.now(UTC) < self._session.lease_expiry and self.ledger.available

    @property
    def link_epoch(self) -> int:
        return self._link_epoch

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "healthy": self.healthy,
                "diagnostic_open": self.diagnostic_open,
                "link_epoch": self._link_epoch,
                "session": asdict(self._session) if self._session else None,
                "snapshot": asdict(self._snapshot),
            }

    def inject_crashed_vision_service(self, asset_id: str) -> None:
        self.inject_fault(asset_id, FaultCode.VISION_SERVICE_CRASHED)

    def inject_fault(self, asset_id: str, fault: FaultCode) -> None:
        with self._lock:
            healthy = {
                "asset_id": asset_id,
                "vision_service_running": True,
                "vision_heartbeat": True,
                "camera_reachable": True,
                "network_configuration_ok": True,
                "usb_peripheral_present": True,
                "safety_circuit_closed": True,
            }
            if fault == FaultCode.VISION_SERVICE_CRASHED:
                healthy.update(vision_service_running=False, vision_heartbeat=False)
            elif fault == FaultCode.NETWORK_INTERFACE_MISCONFIGURED:
                healthy.update(network_configuration_ok=False, vision_heartbeat=False)
            elif fault == FaultCode.CAMERA_UNREACHABLE:
                healthy.update(camera_reachable=False, vision_heartbeat=False)
            elif fault == FaultCode.SAFETY_CIRCUIT_OPEN:
                healthy.update(safety_circuit_closed=False, vision_heartbeat=False)
            elif fault != FaultCode.NONE:
                raise ValueError(f"unsupported injected fault: {fault}")
            self._snapshot = DiagnosticSnapshot(**healthy, error_code=fault.value if fault != FaultCode.NONE else None)

    def dock(self, mission_id: str, asset_id: str, diagnostic_endpoint: str) -> DiagnosticSession:
        with self._lock:
            if not self.ledger.available:
                raise LedgerUnavailable("cannot dock while ledger is unavailable")
            if asset_id != self._snapshot.asset_id:
                raise TargetRejected(f"target identity mismatch: expected {self._snapshot.asset_id}, got {asset_id}")
            self._link_epoch += 1
            self._next_sequence = 1
            self._session = DiagnosticSession(
                mission_id=mission_id,
                asset_id=asset_id,
                link_epoch=self._link_epoch,
                diagnostic_endpoint=diagnostic_endpoint,
                lease_expiry=datetime.now(UTC) + timedelta(seconds=self.lease_seconds),
                next_sequence=1,
            )
            return self._session

    def renew(self, *, mission_id: str, link_epoch: int) -> DiagnosticSession:
        with self._lock:
            session = self._require_session(mission_id, link_epoch)
            self._session = DiagnosticSession(
                mission_id=session.mission_id,
                asset_id=session.asset_id,
                link_epoch=session.link_epoch,
                diagnostic_endpoint=session.diagnostic_endpoint,
                lease_expiry=datetime.now(UTC) + timedelta(seconds=self.lease_seconds),
                next_sequence=self._next_sequence,
            )
            return self._session

    def disconnect(self, *, mission_id: str | None = None, link_epoch: int | None = None) -> None:
        with self._lock:
            if self._session is None:
                return
            if mission_id is not None and mission_id != self._session.mission_id:
                raise TargetRejected("mission identity mismatch")
            if link_epoch is not None and link_epoch != self._session.link_epoch:
                raise TargetRejected("stale link epoch")
            self._session = None

    def expire(self) -> bool:
        with self._lock:
            if self._session is not None and datetime.now(UTC) >= self._session.lease_expiry:
                self._session = None
                return True
            return False

    def snapshot(self, *, mission_id: str, link_epoch: int) -> DiagnosticSnapshot:
        with self._lock:
            self._require_session(mission_id, link_epoch)
            return self._snapshot

    def execute(self, *, mission_id: str, asset_id: str, link_epoch: int, sequence: int, idempotency_key: str, command: str) -> Operation:
        with self._lock:
            session = self._require_session(mission_id, link_epoch)
            if asset_id != session.asset_id:
                raise TargetRejected("asset identity mismatch")
            existing = self.ledger.get(idempotency_key)
            if existing is not None:
                if existing.status == OperationStatus.UNKNOWN:
                    raise TargetRejected("UNKNOWN operation cannot be repeated automatically")
                return existing
            if command not in self.HERO_ALLOWED_MUTATIONS:
                raise TargetRejected(f"command is not authorized: {command}")
            if sequence != self._next_sequence:
                raise TargetRejected(f"expected sequence {self._next_sequence}, got {sequence}")
            operation = self.ledger.begin(
                idempotency_key=idempotency_key,
                mission_id=mission_id,
                asset_id=asset_id,
                link_epoch=link_epoch,
                sequence=sequence,
                command=command,
            )
            try:
                if command == "restart_vision_service":
                    self._snapshot = DiagnosticSnapshot(
                        asset_id=asset_id,
                        vision_service_running=True,
                        vision_heartbeat=True,
                        camera_reachable=self._snapshot.camera_reachable,
                        network_configuration_ok=self._snapshot.network_configuration_ok,
                        usb_peripheral_present=self._snapshot.usb_peripheral_present,
                        safety_circuit_closed=self._snapshot.safety_circuit_closed,
                    )
                elif command == "restore_network_configuration":
                    self._snapshot = DiagnosticSnapshot(
                        asset_id=asset_id,
                        vision_service_running=self._snapshot.vision_service_running,
                        vision_heartbeat=self._snapshot.vision_service_running and self._snapshot.camera_reachable,
                        camera_reachable=self._snapshot.camera_reachable,
                        network_configuration_ok=True,
                        usb_peripheral_present=self._snapshot.usb_peripheral_present,
                        safety_circuit_closed=self._snapshot.safety_circuit_closed,
                    )
                elif command == "reload_camera_driver":
                    self._snapshot = DiagnosticSnapshot(
                        asset_id=asset_id,
                        vision_service_running=self._snapshot.vision_service_running,
                        vision_heartbeat=self._snapshot.vision_service_running and self._snapshot.network_configuration_ok,
                        camera_reachable=True,
                        network_configuration_ok=self._snapshot.network_configuration_ok,
                        usb_peripheral_present=self._snapshot.usb_peripheral_present,
                        safety_circuit_closed=self._snapshot.safety_circuit_closed,
                    )
                recovered = (
                    self._snapshot.vision_service_running
                    and self._snapshot.vision_heartbeat
                    and self._snapshot.camera_reachable
                    and self._snapshot.network_configuration_ok
                    and self._snapshot.safety_circuit_closed
                )
                result = self.ledger.finish(
                    idempotency_key,
                    OperationStatus.SUCCEEDED if recovered else OperationStatus.FAILED,
                    {"vision_healthy": recovered},
                )
                self._next_sequence += 1
                return result
            except BaseException:
                try:
                    self.ledger.finish(idempotency_key, OperationStatus.UNKNOWN, {"error": "mutation result uncertain"})
                finally:
                    self._session = None
                raise

    def _require_session(self, mission_id: str, link_epoch: int) -> DiagnosticSession:
        if not self.diagnostic_open or self._session is None:
            raise TargetRejected("diagnostic link is closed")
        if mission_id != self._session.mission_id:
            raise TargetRejected("mission identity mismatch")
        if link_epoch != self._session.link_epoch:
            raise TargetRejected("stale link epoch")
        return self._session
