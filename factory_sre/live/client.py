"""Small standard-library client used by the persistent Isaac process."""

from __future__ import annotations

import json
import urllib.request
from typing import Any


class FleetClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get(self) -> dict[str, Any]:
        return self._request("GET", "/v1/fleet")

    def robot(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/sim/robot", payload)

    def contact(self, *, asset_id: str, attached: bool, sim_time_s: float) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/sim/contact",
            {"asset_id": asset_id, "attached": attached, "sim_time_s": sim_time_s},
        )

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read())
