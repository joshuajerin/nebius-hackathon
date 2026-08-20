"""Diagnostic transports for the target emulator and physical USB network gadget."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from .contracts import DiagnosticSession, DiagnosticSnapshot, FaultCode, OperationStatus
from .target.runtime import DiagnosticTargetRuntime


class TransportError(RuntimeError):
    pass


def _session(payload: dict[str, Any]) -> DiagnosticSession:
    return DiagnosticSession(
        mission_id=str(payload["mission_id"]),
        asset_id=str(payload["asset_id"]),
        link_epoch=int(payload["link_epoch"]),
        diagnostic_endpoint=str(payload["diagnostic_endpoint"]),
        lease_expiry=datetime.fromisoformat(str(payload["lease_expiry"])),
        next_sequence=int(payload["next_sequence"]),
    )


class HttpDiagnosticTransport:
    """HTTP-over-USB transport; production uses the USB gadget's fixed link-local address."""

    def __init__(self, control_url: str, diagnostic_url: str, *, timeout_s: float = 2.0) -> None:
        self.control_url = control_url.rstrip("/")
        self.diagnostic_url = diagnostic_url.rstrip("/")
        self.timeout_s = timeout_s
        self._next_sequence = 1

    def _request(
        self,
        method: str,
        url: str,
        payload: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
        if body is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise TransportError(f"{method} {url} failed: {exc}") from exc

    def inject_fault(self, asset_id: str, fault: FaultCode) -> None:
        self._request(
            "POST",
            self.control_url + "/v1/faults/inject",
            {"asset_id": asset_id, "fault": fault.value},
        )

    def enumerate(self, mission_id: str, asset_id: str) -> DiagnosticSession:
        payload = self._request(
            "POST",
            self.control_url + "/v1/dock",
            {"mission_id": mission_id, "asset_id": asset_id, "diagnostic_endpoint": self.diagnostic_url},
        )
        self._next_sequence = 1
        return _session(payload)

    def snapshot(self, session: DiagnosticSession) -> DiagnosticSnapshot:
        payload = self._request(
            "GET",
            self.diagnostic_url + "/v1/snapshot",
            headers={"X-Mission-Id": session.mission_id, "X-Link-Epoch": str(session.link_epoch)},
        )
        return DiagnosticSnapshot(**payload)

    def execute(self, session: DiagnosticSession, action: str) -> OperationStatus:
        sequence = self._next_sequence
        payload = self._request(
            "POST",
            self.diagnostic_url + "/v1/operations",
            {
                "mission_id": session.mission_id,
                "asset_id": session.asset_id,
                "link_epoch": session.link_epoch,
                "sequence": sequence,
                "idempotency_key": f"{session.mission_id}:{session.link_epoch}:{sequence}:{action}",
                "command": action,
            },
        )
        self._next_sequence += 1
        return OperationStatus(payload["status"])

    def disconnect(self, session: DiagnosticSession) -> None:
        self._request(
            "POST",
            self.control_url + "/v1/disconnect",
            {"mission_id": session.mission_id, "link_epoch": session.link_epoch},
        )


class RuntimeDiagnosticTransport:
    """In-process transport used by deterministic scenario and unit tests."""

    def __init__(self, runtime: DiagnosticTargetRuntime, diagnostic_url: str = "usb://factory-sre") -> None:
        self.runtime = runtime
        self.diagnostic_url = diagnostic_url
        self._next_sequence = 1

    def inject_fault(self, asset_id: str, fault: FaultCode) -> None:
        self.runtime.inject_fault(asset_id, fault)

    def enumerate(self, mission_id: str, asset_id: str) -> DiagnosticSession:
        self._next_sequence = 1
        return self.runtime.dock(mission_id, asset_id, self.diagnostic_url)

    def snapshot(self, session: DiagnosticSession) -> DiagnosticSnapshot:
        return self.runtime.snapshot(mission_id=session.mission_id, link_epoch=session.link_epoch)

    def execute(self, session: DiagnosticSession, action: str) -> OperationStatus:
        sequence = self._next_sequence
        operation = self.runtime.execute(
            mission_id=session.mission_id,
            asset_id=session.asset_id,
            link_epoch=session.link_epoch,
            sequence=sequence,
            idempotency_key=f"{session.mission_id}:{session.link_epoch}:{sequence}:{action}",
            command=action,
        )
        self._next_sequence += 1
        return operation.status

    def disconnect(self, session: DiagnosticSession) -> None:
        self.runtime.disconnect(mission_id=session.mission_id, link_epoch=session.link_epoch)
