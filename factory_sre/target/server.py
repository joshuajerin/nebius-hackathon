"""Two-port HTTP service: always-on control and dock-gated diagnostics."""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from ..contracts import FaultCode
from ..live.fleet import FleetCoordinator
from .ledger import LedgerUnavailable, OperationLedger
from .runtime import DiagnosticTargetRuntime, TargetRejected


def _json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    return json.loads(handler.rfile.read(length) or b"{}")


def _send(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: object) -> None:
    body = json.dumps(payload, default=str, sort_keys=True).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_bytes(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    body: bytes,
    *,
    content_type: str,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_control_handler(
    runtime: DiagnosticTargetRuntime,
    fleet: FleetCoordinator,
    *,
    open_diagnostic: Callable[[], None],
    close_diagnostic: Callable[[], None],
    frame_path: Path,
) -> type[BaseHTTPRequestHandler]:
    dashboard_path = Path(__file__).resolve().parents[1] / "live" / "dashboard" / "index.html"

    class ControlHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in {"/", "/index.html"}:
                _send_bytes(self, HTTPStatus.OK, dashboard_path.read_bytes(), content_type="text/html; charset=utf-8")
            elif path == "/healthz":
                _send(self, HTTPStatus.OK if runtime.healthy else HTTPStatus.SERVICE_UNAVAILABLE, runtime.status())
            elif path == "/v1/status":
                _send(self, HTTPStatus.OK, runtime.status())
            elif path == "/v1/fleet":
                _send(self, HTTPStatus.OK, fleet.snapshot())
            elif path == "/v1/sim/frame.jpg":
                if frame_path.is_file():
                    _send_bytes(self, HTTPStatus.OK, frame_path.read_bytes(), content_type="image/jpeg")
                else:
                    _send(self, HTTPStatus.NOT_FOUND, {"error": "simulator frame is not available yet"})
            else:
                _send(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                path = urlsplit(self.path).path
                data = _json(self)
                if path == "/v1/faults/crash-vision":
                    runtime.inject_crashed_vision_service(str(data["asset_id"]))
                    _send(self, HTTPStatus.OK, runtime.status())
                elif path == "/v1/faults/inject":
                    runtime.inject_fault(str(data["asset_id"]), FaultCode(str(data["fault"])))
                    _send(self, HTTPStatus.OK, runtime.status())
                elif path == "/v1/dock":
                    session = runtime.dock(str(data["mission_id"]), str(data["asset_id"]), str(data["diagnostic_endpoint"]))
                    open_diagnostic()
                    _send(self, HTTPStatus.CREATED, asdict(session))
                elif path == "/v1/renew":
                    session = runtime.renew(mission_id=str(data["mission_id"]), link_epoch=int(data["link_epoch"]))
                    _send(self, HTTPStatus.OK, asdict(session))
                elif path == "/v1/disconnect":
                    runtime.disconnect(mission_id=data.get("mission_id"), link_epoch=data.get("link_epoch"))
                    close_diagnostic()
                    _send(self, HTTPStatus.OK, runtime.status())
                elif path == "/v1/fleet/reset":
                    _send(self, HTTPStatus.OK, fleet.reset())
                elif path.startswith("/v1/workstations/") and path.endswith("/fail"):
                    asset_id = path.removeprefix("/v1/workstations/").removesuffix("/fail").strip("/")
                    _send(self, HTTPStatus.OK, fleet.fail(asset_id))
                elif path == "/v1/sim/robot":
                    _send(self, HTTPStatus.OK, fleet.update_robot(data))
                elif path == "/v1/sim/contact":
                    _send(
                        self,
                        HTTPStatus.OK,
                        fleet.update_contact(
                            asset_id=str(data["asset_id"]),
                            attached=bool(data["attached"]),
                            sim_time_s=float(data["sim_time_s"]),
                        ),
                    )
                else:
                    _send(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (KeyError, ValueError, TargetRejected, LedgerUnavailable) as exc:
                _send(self, HTTPStatus.CONFLICT, {"error": str(exc)})

    return ControlHandler


def make_diagnostic_handler(runtime: DiagnosticTargetRuntime) -> type[BaseHTTPRequestHandler]:
    class DiagnosticHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _closed(self) -> bool:
            if runtime.diagnostic_open:
                return False
            _send(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": "diagnostic link is closed"})
            return True

        def do_GET(self) -> None:
            if self._closed():
                return
            try:
                mission_id = self.headers["X-Mission-Id"]
                link_epoch = int(self.headers["X-Link-Epoch"])
                if self.path == "/v1/snapshot":
                    _send(self, HTTPStatus.OK, asdict(runtime.snapshot(mission_id=mission_id, link_epoch=link_epoch)))
                else:
                    _send(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (KeyError, ValueError, TargetRejected) as exc:
                _send(self, HTTPStatus.CONFLICT, {"error": str(exc)})

        def do_POST(self) -> None:
            if self._closed():
                return
            if self.path != "/v1/operations":
                _send(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                data = _json(self)
                operation = runtime.execute(
                    mission_id=str(data["mission_id"]), asset_id=str(data["asset_id"]),
                    link_epoch=int(data["link_epoch"]), sequence=int(data["sequence"]),
                    idempotency_key=str(data["idempotency_key"]), command=str(data["command"]),
                )
                _send(self, HTTPStatus.OK, asdict(operation))
            except (KeyError, ValueError, TargetRejected, LedgerUnavailable) as exc:
                _send(self, HTTPStatus.CONFLICT, {"error": str(exc)})

    return DiagnosticHandler


class DiagnosticListener:
    """Opens the diagnostic TCP socket only while physical contact is valid."""

    def __init__(self, runtime: DiagnosticTargetRuntime, host: str, port: int) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self._lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def listening(self) -> bool:
        return self._server is not None

    def open(self) -> None:
        with self._lock:
            if self._server is not None:
                return
            server = ThreadingHTTPServer((self.host, self.port), make_diagnostic_handler(self.runtime))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self._server = server
            self._thread = thread

    def close(self) -> None:
        with self._lock:
            server = self._server
            self._server = None
            self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()


def serve() -> None:
    ledger = OperationLedger(os.environ.get("FACTORY_SRE_LEDGER", "/tmp/factory-sre.sqlite3"))
    ledger.open()
    runtime = DiagnosticTargetRuntime(ledger, lease_seconds=float(os.environ.get("FACTORY_SRE_LEASE_SECONDS", "10")))
    fleet = FleetCoordinator(
        contact_hold_seconds=float(os.environ.get("FACTORY_SRE_CONTACT_HOLD_SECONDS", "5"))
    )
    diagnostic = DiagnosticListener(
        runtime,
        os.environ.get("FACTORY_SRE_DIAGNOSTIC_HOST", "127.0.0.1"),
        int(os.environ.get("FACTORY_SRE_DIAGNOSTIC_PORT", "8766")),
    )
    control = ThreadingHTTPServer(
        (os.environ.get("FACTORY_SRE_CONTROL_HOST", "127.0.0.1"), int(os.environ.get("FACTORY_SRE_CONTROL_PORT", "8765"))),
        make_control_handler(
            runtime,
            fleet,
            open_diagnostic=diagnostic.open,
            close_diagnostic=diagnostic.close,
            frame_path=Path(
                os.environ.get("FACTORY_SRE_FRAME_PATH", "/workspace/output/live/latest.jpg")
            ),
        ),
    )
    stop = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=control.serve_forever, daemon=True).start()
    print("factory-sre diagnostic target ready", flush=True)
    try:
        while not stop.wait(0.1):
            if runtime.expire():
                diagnostic.close()
    finally:
        control.shutdown()
        control.server_close()
        diagnostic.close()
        ledger.close()


if __name__ == "__main__":
    serve()
