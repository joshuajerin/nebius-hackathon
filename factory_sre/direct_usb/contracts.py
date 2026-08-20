"""Decision-complete contracts for the direct-USB-C Factory SRE workcell."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from factory_sre.contracts import (
    DiagnosticSession, DockObservation, FaultCode, OperationStatus, Pose2D, Transform3D,
)


class MissionState(StrEnum):
    DISPATCH="DISPATCH"; NAVIGATE="NAVIGATE"; STABILIZE="STABILIZE"; LOCALIZE="LOCALIZE"
    APPROACH="APPROACH"; INSERT="INSERT"; ENUMERATE="ENUMERATE"; DIAGNOSE="DIAGNOSE"
    RECOVER="RECOVER"; VERIFY="VERIFY"; DISCONNECT="DISCONNECT"; REPORT="REPORT"
    ESCALATE="ESCALATE"; EMERGENCY_STOP="EMERGENCY_STOP"


class TerminalReason(StrEnum):
    RECOVERED="RECOVERED"; WRONG_ASSET="WRONG_ASSET"; PERCEPTION_FAILED="PERCEPTION_FAILED"
    NAVIGATION_FAILED="NAVIGATION_FAILED"; BASE_UNSTABLE="BASE_UNSTABLE"; FORCE_LIMIT="FORCE_LIMIT"
    ENUMERATION_FAILED="ENUMERATION_FAILED"; UNSUPPORTED_DIAGNOSIS="UNSUPPORTED_DIAGNOSIS"
    UNAUTHORIZED="UNAUTHORIZED"; RECOVERY_FAILED="RECOVERY_FAILED"
    VERIFICATION_FAILED="VERIFICATION_FAILED"; INTERNAL_ERROR="INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class Snapshot:
    asset_id: str
    vision_service_running: bool
    vision_heartbeat: bool
    camera_reachable: bool
    network_configuration_ok: bool
    safety_circuit_closed: bool
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class Diagnosis:
    fault: FaultCode
    action: str | None
    autonomous: bool


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    path: str
    source_release: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class Workcell:
    asset_id: str
    coarse_tag_id: int
    coarse_tag_size_m: float
    fine_tag_id: int
    fine_tag_size_m: float
    approach_pose: Pose2D
    fine_tag_to_usb_port: Transform3D
    permitted_actions: frozenset[str]

    def __post_init__(self) -> None:
        if self.coarse_tag_size_m != 0.150 or self.fine_tag_size_m != 0.040:
            raise ValueError("direct USB workcell requires 150 mm coarse and 40 mm fine tags")
        if self.fine_tag_to_usb_port.translation[0] >= 0:
            raise ValueError("USB-C port must be left of the fine tag")


@dataclass(frozen=True, slots=True)
class Registry:
    assets: tuple[Asset, ...]
    workcells: tuple[Workcell, ...]


class Locomotion(Protocol):
    def navigate(self, pose: Pose2D) -> bool: ...
    def stabilize(self) -> bool: ...
    def stable(self) -> bool: ...
    def emergency_stop(self) -> None: ...


class Arm(Protocol):
    def move_to_standoff(self) -> bool: ...
    def visual_servo(self, pose: Transform3D) -> bool: ...
    def guarded_insert(self, pose: Transform3D) -> DockObservation: ...
    def retreat(self) -> bool: ...
    def emergency_stop(self) -> None: ...


class Camera(Protocol):
    def detect_tags(self, family: str, size_m: float) -> tuple[tuple[int, Transform3D], ...]: ...


class Transport(Protocol):
    def enumerate(self, mission_id: str, asset_id: str) -> DiagnosticSession: ...
    def snapshot(self, session: DiagnosticSession) -> Snapshot: ...
    def execute(self, session: DiagnosticSession, action: str) -> OperationStatus: ...
    def disconnect(self, session: DiagnosticSession) -> None: ...


ASSETS = (
    Asset("go2", "/Isaac/Robots/Unitree/Go2/go2.usd", "isaac-sim-6.0.1"),
    Asset("go2-policy", "/Isaac/Samples/Policies/go2/physx_policy.pt", "isaac-sim-6.0.1"),
    Asset("go2-policy-env", "/Isaac/Samples/Policies/go2/physx_env.yaml", "isaac-sim-6.0.1"),
    Asset("so101", "so101_antioch", "antioch", "1.3.2"),
    Asset("warehouse", "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd", "isaac-sim-6.0.1"),
    Asset("ur5e", "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd", "isaac-sim-6.0.1"),
    Asset("d455", "/Isaac/Sensors/Realsense/D455/rsd455.usd", "isaac-sim-6.0.1"),
    Asset("p61x", "/Isaac/Sensors/SICK/InspectorP61x/SICK_InspectorP61x.usd", "isaac-sim-6.0.1"),
    Asset("nanoscan3", "/Isaac/Sensors/SICK/nanoScan3/SICK_nanoScan3.usd", "isaac-sim-6.0.1"),
    Asset("inspector83x", "/Isaac/Sensors/SICK/Inspector83x/SICK_Inspector83x.usd", "isaac-sim-6.0.1"),
)
VISION_CELL = Workcell(
    "vision-cell-37", 10, 0.150, 20, 0.040, Pose2D(2.0, -0.9, 1.5707963268),
    Transform3D((-0.060, 0.0, 0.0)),
    frozenset({"restart_vision_service", "restore_network_configuration", "reload_camera_driver"}),
)
REGISTRY = Registry(ASSETS, (VISION_CELL,))
