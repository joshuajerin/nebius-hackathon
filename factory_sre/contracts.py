"""Typed boundaries for the Factory SRE simulator and target."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import sqrt
from typing import Protocol, runtime_checkable

class MissionState(StrEnum):
    DETECT="DETECT"; DISPATCH="DISPATCH"; NAVIGATE="NAVIGATE"; IDENTIFY="IDENTIFY"; INSPECT="INSPECT"
    ALIGN="ALIGN"; PLANT_BASE="PLANT_BASE"; APPROACH="APPROACH"; VISUAL_SERVO="VISUAL_SERVO"
    MAGNETIC_DOCK="MAGNETIC_DOCK"; LINK_VERIFY="LINK_VERIFY"; DIAGNOSE="DIAGNOSE"; AUTHORIZE="AUTHORIZE"
    RESTART="RESTART"; VERIFY="VERIFY"; DISCONNECT="DISCONNECT"; REPORT="REPORT"
    ESCALATE="ESCALATE"; EMERGENCY_STOP="EMERGENCY_STOP"

class TerminalReason(StrEnum):
    RECOVERED="RECOVERED"; WRONG_ASSET="WRONG_ASSET"; PERCEPTION_FAILED="PERCEPTION_FAILED"
    NAVIGATION_FAILED="NAVIGATION_FAILED"; BASE_UNSTABLE="BASE_UNSTABLE"; FORCE_LIMIT="FORCE_LIMIT"
    LINK_FAILED="LINK_FAILED"; UNSUPPORTED_DIAGNOSIS="UNSUPPORTED_DIAGNOSIS"; UNAUTHORIZED="UNAUTHORIZED"
    RECOVERY_FAILED="RECOVERY_FAILED"; TELEMETRY_INCOMPLETE="TELEMETRY_INCOMPLETE"; INTERNAL_ERROR="INTERNAL_ERROR"

class OperationStatus(StrEnum):
    PENDING="PENDING"; SUCCEEDED="SUCCEEDED"; FAILED="FAILED"; UNKNOWN="UNKNOWN"

class FaultCode(StrEnum):
    NONE="NONE"; VISION_SERVICE_CRASHED="VISION_SERVICE_CRASHED"; VISION_HEALTHCHECK_FAILED="VISION_HEALTHCHECK_FAILED"
    NETWORK_INTERFACE_MISCONFIGURED="NETWORK_INTERFACE_MISCONFIGURED"; CAMERA_UNREACHABLE="CAMERA_UNREACHABLE"
    USB_PERIPHERAL_MISSING="USB_PERIPHERAL_MISSING"; SAFETY_CIRCUIT_OPEN="SAFETY_CIRCUIT_OPEN"; UNKNOWN="UNKNOWN"

@dataclass(frozen=True, slots=True)
class Pose2D: x: float; y: float; yaw: float

@dataclass(frozen=True, slots=True)
class Transform3D:
    translation: tuple[float,float,float]
    quaternion_wxyz: tuple[float,float,float,float]=(1.0,0.0,0.0,0.0)
    def compose(self, child: "Transform3D") -> "Transform3D":
        w,x,y,z=self.quaternion_wxyz; n=sqrt(w*w+x*x+y*y+z*z)
        if n==0: raise ValueError("quaternion must be non-zero")
        w,x,y,z=(v/n for v in (w,x,y,z)); vx,vy,vz=child.translation
        tx,ty,tz=2*(y*vz-z*vy),2*(z*vx-x*vz),2*(x*vy-y*vx)
        r=(vx+w*tx+y*tz-z*ty,vy+w*ty+z*tx-x*tz,vz+w*tz+x*ty-y*tx)
        cw,cx,cy,cz=child.quaternion_wxyz
        q=(w*cw-x*cx-y*cy-z*cz,w*cx+x*cw+y*cz-z*cy,w*cy-x*cz+y*cw+z*cx,w*cz+x*cy-y*cx+z*cw)
        qn=sqrt(sum(v*v for v in q))
        return Transform3D(tuple(a+b for a,b in zip(self.translation,r,strict=True)),tuple(v/qn for v in q))

@dataclass(frozen=True, slots=True)
class Incident:
    incident_id:str; asset_id:str; fault_hint:str
    detected_at:datetime=field(default_factory=lambda:datetime.now(UTC))

@dataclass(frozen=True, slots=True)
class DiagnosticTarget: control_endpoint:str; diagnostic_endpoint:str

@dataclass(frozen=True, slots=True)
class WorkbenchManifest:
    asset_id:str; tag_family:str; tag_id:int; tag_size_m:float; approach_pose:Pose2D
    tag_to_dock:Transform3D; diagnostic_target:DiagnosticTarget
    permitted_recovery_actions:frozenset[str]=frozenset({"restart_vision_service"})
    def __post_init__(self):
        if self.tag_family!="tag36h11": raise ValueError("Factory SRE supports tag36h11 only")
        if self.tag_size_m!=0.120: raise ValueError("AprilTag printed size must be 120 mm")
        if self.tag_to_dock.translation!=(-0.30,0.0,0.0): raise ValueError("dock must be exactly 0.30 m left of tag center")
        if self.permitted_recovery_actions!=frozenset({"restart_vision_service"}): raise ValueError("hero mutation allowlist changed")

@dataclass(frozen=True, slots=True)
class DockObservation:
    asset_id:str; estimated_pose:Transform3D; covariance:tuple[float,...]; contact:bool; depth_m:float; axial_force_n:float; link_epoch:int

@dataclass(frozen=True, slots=True)
class DiagnosticSession:
    mission_id:str; asset_id:str; link_epoch:int; diagnostic_endpoint:str; lease_expiry:datetime; next_sequence:int

@dataclass(frozen=True, slots=True)
class DiagnosticSnapshot:
    asset_id:str; vision_service_running:bool; vision_heartbeat:bool; camera_reachable:bool
    network_configuration_ok:bool; safety_circuit_closed:bool; usb_peripheral_present:bool=True; error_code:str|None=None

@dataclass(frozen=True, slots=True)
class Diagnosis:
    fault:FaultCode; confidence:float; evidence:tuple[str,...]; recommended_action:str|None; autonomous:bool

@runtime_checkable
class BaseAdapter(Protocol):
    def navigate(self,pose:Pose2D)->bool: ...
    def plant(self)->bool: ...
    def stable(self)->bool: ...
    def emergency_stop(self)->None: ...
@runtime_checkable
class ArmAdapter(Protocol):
    def move_to_inspection(self)->bool: ...
    def align(self,target:Transform3D)->bool: ...
    def approach(self,target:Transform3D)->bool: ...
    def guarded_dock(self,target:Transform3D)->DockObservation: ...
    def retreat(self)->bool: ...
    def emergency_stop(self)->None: ...
@runtime_checkable
class CameraAdapter(Protocol):
    def detect_tags(self,family:str,size_m:float)->tuple[tuple[int,Transform3D],...]: ...
    def capture_rgb(self)->object: ...
@runtime_checkable
class DiagnosticTransport(Protocol):
    def enumerate(self,mission_id:str,asset_id:str)->DiagnosticSession: ...
    def snapshot(self,session:DiagnosticSession)->DiagnosticSnapshot: ...
    def execute(self,session:DiagnosticSession,action:str)->OperationStatus: ...
    def disconnect(self,session:DiagnosticSession)->None: ...

TARGET=DiagnosticTarget("http://127.0.0.1:8765","http://127.0.0.1:8766")
WORKBENCH_A=WorkbenchManifest("workbench-a","tag36h11",10,0.120,Pose2D(1.5,-0.9,1.5707963268),Transform3D((-0.30,0.0,0.0)),TARGET)
WORKBENCH_B=WorkbenchManifest("workbench-b","tag36h11",20,0.120,Pose2D(3.7,-0.9,1.5707963268),Transform3D((-0.30,0.0,0.0)),TARGET)
WORKBENCHES=(WORKBENCH_A,WORKBENCH_B)
def manifest_for_asset(asset_id:str)->WorkbenchManifest:
    for manifest in WORKBENCHES:
        if manifest.asset_id==asset_id:return manifest
    raise KeyError(asset_id)
