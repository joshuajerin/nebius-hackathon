"""Transport adapters for the direct USB device-mode target."""

from __future__ import annotations

from factory_sre.contracts import DiagnosticSession, OperationStatus

from .contracts import Snapshot
from .runtime import DirectUsbTargetRuntime


class InProcessDirectUsbTransport:
    """Keeps mission logic identical while tests replace the physical USB link."""

    def __init__(self, runtime: DirectUsbTargetRuntime) -> None:
        self.runtime = runtime

    def enumerate(self, mission_id: str, asset_id: str) -> DiagnosticSession:
        return self.runtime.enumerate(mission_id, asset_id)

    def snapshot(self, session: DiagnosticSession) -> Snapshot:
        return self.runtime.snapshot(session)

    def execute(self, session: DiagnosticSession, action: str) -> OperationStatus:
        return self.runtime.execute(session, action)

    def disconnect(self, session: DiagnosticSession) -> None:
        self.runtime.disconnect(session)

