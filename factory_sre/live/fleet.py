"""Thread-safe fleet state shared by the dashboard and persistent simulator."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from factory_sre.sim.tagged_box_grid import tagged_service_boxes


CONTACT_HOLD_SECONDS = 5.0
MAX_EVENTS = 120


@dataclass(slots=True)
class WorkstationState:
    asset_id: str
    box_number: int
    tag_id: int
    center: tuple[float, float, float]
    status: str = "online"
    failure_sequence: int | None = None
    failed_at: float | None = None
    recovered_at: float | None = None
    recoveries: int = 0


@dataclass(slots=True)
class RobotState:
    phase: str = "resting"
    position: tuple[float, float, float] = (-0.55, 1.60, 0.50)
    yaw: float = 0.0
    current_target: str | None = None
    planned_queue: tuple[str, ...] = ()
    contact_asset: str | None = None
    contact_seconds: float = 0.0
    last_heartbeat: float | None = None
    simulator_ready: bool = False
    detail: str = "Waiting for persistent simulator"


@dataclass(frozen=True, slots=True)
class FleetEvent:
    event_id: int
    timestamp: float
    kind: str
    asset_id: str | None
    message: str


class FleetCoordinator:
    """Own workstation incidents, robot telemetry, and the five-second gate."""

    def __init__(self, *, contact_hold_seconds: float = CONTACT_HOLD_SECONDS) -> None:
        if contact_hold_seconds <= 0.0:
            raise ValueError("contact hold must be positive")
        self.contact_hold_seconds = float(contact_hold_seconds)
        self._lock = threading.RLock()
        self._failure_sequence = 0
        self._event_sequence = 0
        self._stations = {
            box.asset_id: WorkstationState(
                asset_id=box.asset_id,
                box_number=box.box_number,
                tag_id=box.tag_id,
                center=box.center,
            )
            for box in tagged_service_boxes()
        }
        self._robot = RobotState()
        self._events: deque[FleetEvent] = deque(maxlen=MAX_EVENTS)
        self._contact_started_sim_time: float | None = None
        self._record("system", None, "Fleet coordinator ready; all workstations online")

    def _record(self, kind: str, asset_id: str | None, message: str) -> None:
        self._event_sequence += 1
        self._events.appendleft(
            FleetEvent(
                event_id=self._event_sequence,
                timestamp=time.time(),
                kind=kind,
                asset_id=asset_id,
                message=message,
            )
        )

    def fail(self, asset_id: str) -> dict[str, Any]:
        with self._lock:
            station = self._station(asset_id)
            if station.status == "offline":
                return self.snapshot()
            self._failure_sequence += 1
            station.status = "offline"
            station.failure_sequence = self._failure_sequence
            station.failed_at = time.time()
            station.recovered_at = None
            self._record("failure", asset_id, f"Box {station.box_number} taken offline from dashboard")
            return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            for station in self._stations.values():
                station.status = "online"
                station.failure_sequence = None
                station.failed_at = None
                station.recovered_at = None
            self._robot.current_target = None
            self._robot.planned_queue = ()
            self._robot.contact_asset = None
            self._robot.contact_seconds = 0.0
            self._contact_started_sim_time = None
            self._record("reset", None, "Dashboard reset all workstation incidents")
            return self.snapshot()

    def update_robot(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            previous_phase = self._robot.phase
            previous_target = self._robot.current_target
            position = payload.get("position", self._robot.position)
            if len(position) != 3:
                raise ValueError("robot position must contain x, y, z")
            self._robot.phase = str(payload.get("phase", self._robot.phase))
            self._robot.position = tuple(float(value) for value in position)
            self._robot.yaw = float(payload.get("yaw", self._robot.yaw))
            target = payload.get("current_target", self._robot.current_target)
            if target is not None:
                self._station(str(target))
            self._robot.current_target = str(target) if target is not None else None
            queue = tuple(str(asset_id) for asset_id in payload.get("planned_queue", self._robot.planned_queue))
            for asset_id in queue:
                self._station(asset_id)
            self._robot.planned_queue = queue
            self._robot.simulator_ready = bool(payload.get("simulator_ready", True))
            self._robot.detail = str(payload.get("detail", self._robot.detail))
            self._robot.last_heartbeat = time.time()
            if self._robot.phase != previous_phase or self._robot.current_target != previous_target:
                self._record(
                    "robot",
                    self._robot.current_target,
                    f"Robot {self._robot.phase}: {self._robot.detail}",
                )
            return self.snapshot()

    def update_contact(self, *, asset_id: str, attached: bool, sim_time_s: float) -> dict[str, Any]:
        with self._lock:
            station = self._station(asset_id)
            sim_time_s = float(sim_time_s)
            if not attached:
                if self._robot.contact_asset is not None:
                    self._record("contact", self._robot.contact_asset, "Magnetic contact released")
                self._robot.contact_asset = None
                self._robot.contact_seconds = 0.0
                self._contact_started_sim_time = None
                return self.snapshot()
            if station.status != "offline":
                raise ValueError(f"cannot repair online workstation: {asset_id}")
            if self._robot.current_target != asset_id:
                raise ValueError("magnetic contact identity does not match the active target")
            if self._robot.contact_asset != asset_id or self._contact_started_sim_time is None:
                self._robot.contact_asset = asset_id
                self._contact_started_sim_time = sim_time_s
                self._robot.contact_seconds = 0.0
                self._record("contact", asset_id, "Magnetic contact established; five-second recovery timer started")
            elapsed = max(0.0, sim_time_s - self._contact_started_sim_time)
            self._robot.contact_seconds = min(elapsed, self.contact_hold_seconds)
            if elapsed >= self.contact_hold_seconds:
                station.status = "online"
                station.recovered_at = time.time()
                station.failure_sequence = None
                station.recoveries += 1
                self._record(
                    "recovery",
                    asset_id,
                    f"Box {station.box_number} restored after {elapsed:.1f}s magnetic contact",
                )
                self._robot.contact_asset = None
                self._robot.contact_seconds = 0.0
                self._contact_started_sim_time = None
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stations = sorted(self._stations.values(), key=lambda item: item.box_number)
            failures = sorted(
                (station for station in stations if station.status == "offline"),
                key=lambda item: item.failure_sequence or 0,
            )
            heartbeat_age = (
                None
                if self._robot.last_heartbeat is None
                else max(0.0, time.time() - self._robot.last_heartbeat)
            )
            return {
                "schema_version": 1,
                "contact_hold_seconds": self.contact_hold_seconds,
                "workstations": [asdict(station) for station in stations],
                "pending_failures": [station.asset_id for station in failures],
                "robot": asdict(self._robot) | {"heartbeat_age_seconds": heartbeat_age},
                "events": [asdict(event) for event in self._events],
                "summary": {
                    "online": sum(station.status == "online" for station in stations),
                    "offline": len(failures),
                    "recoveries": sum(station.recoveries for station in stations),
                },
            }

    def _station(self, asset_id: str) -> WorkstationState:
        try:
            return self._stations[asset_id]
        except KeyError as exc:
            raise ValueError(f"unknown workstation: {asset_id}") from exc
