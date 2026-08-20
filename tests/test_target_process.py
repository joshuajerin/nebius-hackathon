from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(url: str, data: dict | None = None, headers: dict[str, str] | None = None):
    payload = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(request, timeout=1.0) as response:
        return response.status, json.loads(response.read())


def _port_closed(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.15)
        return sock.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.integration
def test_physical_link_controls_actual_diagnostic_listener(tmp_path) -> None:
    control, diagnostic = _free_port(), _free_port()
    env = os.environ | {
        "FACTORY_SRE_CONTROL_PORT": str(control),
        "FACTORY_SRE_DIAGNOSTIC_PORT": str(diagnostic),
        "FACTORY_SRE_LEDGER": str(tmp_path / "ledger.sqlite3"),
        "FACTORY_SRE_LEASE_SECONDS": "2",
    }
    process = subprocess.Popen([sys.executable, "-m", "factory_sre.target.server"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if _request(f"http://127.0.0.1:{control}/healthz")[0] == 200:
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        else:
            raise AssertionError("control listener did not become healthy")
        assert _port_closed(diagnostic)
        _request(f"http://127.0.0.1:{control}/v1/faults/crash-vision", {"asset_id": "vision-cell-37"})
        _, session = _request(
            f"http://127.0.0.1:{control}/v1/dock",
            {"mission_id": "m-1", "asset_id": "vision-cell-37", "diagnostic_endpoint": f"http://127.0.0.1:{diagnostic}"},
        )
        assert not _port_closed(diagnostic)
        headers = {"X-Mission-Id": "m-1", "X-Link-Epoch": str(session["link_epoch"])}
        assert _request(f"http://127.0.0.1:{diagnostic}/v1/snapshot", headers=headers)[1]["vision_service_running"] is False
        status, operation = _request(
            f"http://127.0.0.1:{diagnostic}/v1/operations",
            {"mission_id": "m-1", "asset_id": "vision-cell-37", "link_epoch": session["link_epoch"], "sequence": 1, "idempotency_key": "m-1:restart", "command": "restart_vision_service"},
        )
        assert status == 200 and operation["status"] == "SUCCEEDED"
        _request(f"http://127.0.0.1:{control}/v1/disconnect", {"mission_id": "m-1", "link_epoch": session["link_epoch"]})
        assert _port_closed(diagnostic)
    finally:
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
