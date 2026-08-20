"""Fail-closed mission state machine."""
from __future__ import annotations
from dataclasses import dataclass,field
from time import monotonic
from .contracts import *
from .diagnosis import diagnose
FLOW=(MissionState.DETECT,MissionState.DISPATCH,MissionState.NAVIGATE,MissionState.IDENTIFY,MissionState.INSPECT,MissionState.ALIGN,MissionState.PLANT_BASE,MissionState.APPROACH,MissionState.VISUAL_SERVO,MissionState.MAGNETIC_DOCK,MissionState.LINK_VERIFY,MissionState.DIAGNOSE,MissionState.AUTHORIZE,MissionState.RESTART,MissionState.VERIFY,MissionState.DISCONNECT,MissionState.REPORT)
@dataclass(frozen=True,slots=True)
class MissionEvent: state:MissionState; elapsed_s:float; detail:str
@dataclass(frozen=True,slots=True)
class MissionResult:
    mission_id:str; asset_id:str; success:bool; terminal_state:MissionState; terminal_reason:TerminalReason; dock_attempts:int; events:tuple[MissionEvent,...]
@dataclass(slots=True)
class MissionMachine:
    mission_id:str; expected_asset_id:str; state:MissionState=MissionState.DETECT; max_force_n:float=8.0
    terminal_reason:TerminalReason|None=None; dock_attempts:int=0; events:list[MissionEvent]=field(default_factory=list); _start:float=field(default_factory=monotonic)
    def __post_init__(self):self.record("incident detected")
    def record(self,d):self.events.append(MissionEvent(self.state,monotonic()-self._start,d))
    def advance(self,d):self.state=FLOW[FLOW.index(self.state)+1];self.record(d)
    def escalate(self,r,d):self.state=MissionState.ESCALATE;self.terminal_reason=r;self.record(d)
    def stop(self,r,d):self.state=MissionState.EMERGENCY_STOP;self.terminal_reason=r;self.record(d)
    def result(self):
        r=self.terminal_reason or TerminalReason.INTERNAL_ERROR
        return MissionResult(self.mission_id,self.expected_asset_id,r==TerminalReason.RECOVERED,self.state,r,self.dock_attempts,tuple(self.events))
class MissionController:
    def __init__(self,max_force_n=8.0,minimum_depth_m=0.005):self.max_force_n=max_force_n;self.minimum_depth_m=minimum_depth_m
    def run(self,incident:Incident,manifest:WorkbenchManifest,base:BaseAdapter,arm:ArmAdapter,camera:CameraAdapter,transport:DiagnosticTransport)->MissionResult:
        m=MissionMachine(incident.incident_id,incident.asset_id,max_force_n=self.max_force_n);session=None
        try:
            if incident.asset_id!=manifest.asset_id:m.escalate(TerminalReason.WRONG_ASSET,"manifest mismatch");return m.result()
            m.advance("accepted");m.advance("dispatched")
            if not base.navigate(manifest.approach_pose):m.escalate(TerminalReason.NAVIGATION_FAILED,"unreachable");return m.result()
            m.advance("arrived");tags=camera.detect_tags(manifest.tag_family,manifest.tag_size_m)
            if tags and all(i!=manifest.tag_id for i,_ in tags):m.escalate(TerminalReason.WRONG_ASSET,"neighbor tag");return m.result()
            tag=next((p for i,p in tags if i==manifest.tag_id),None)
            if tag is None:m.escalate(TerminalReason.PERCEPTION_FAILED,"tag absent");return m.result()
            dock=tag.compose(manifest.tag_to_dock);m.advance("identified")
            if not arm.move_to_inspection():m.escalate(TerminalReason.PERCEPTION_FAILED,"inspect");return m.result()
            m.advance("inspected")
            if not arm.align(dock):m.escalate(TerminalReason.PERCEPTION_FAILED,"align");return m.result()
            m.advance("aligned")
            if not base.plant() or not base.stable():m.stop(TerminalReason.BASE_UNSTABLE,"unstable");return m.result()
            m.advance("planted")
            if not arm.approach(dock):m.escalate(TerminalReason.PERCEPTION_FAILED,"approach");return m.result()
            m.advance("standoff")
            if not arm.align(dock):m.escalate(TerminalReason.PERCEPTION_FAILED,"servo");return m.result()
            m.advance("servoed");m.dock_attempts+=1;obs=arm.guarded_dock(dock)
            if obs.axial_force_n>m.max_force_n:arm.emergency_stop();base.emergency_stop();m.stop(TerminalReason.FORCE_LIMIT,"force");return m.result()
            if obs.asset_id!=manifest.asset_id:m.escalate(TerminalReason.WRONG_ASSET,"dock identity");return m.result()
            if not obs.contact or obs.depth_m<self.minimum_depth_m:m.escalate(TerminalReason.LINK_FAILED,"dock incomplete");return m.result()
            m.advance("docked");session=transport.enumerate(m.mission_id,manifest.asset_id);m.advance("link verified")
            dx=diagnose(transport.snapshot(session))
            if not dx.autonomous or dx.recommended_action is None:m.escalate(TerminalReason.UNSUPPORTED_DIAGNOSIS,dx.fault.value);return m.result()
            m.advance("diagnosed")
            if dx.recommended_action not in manifest.permitted_recovery_actions:m.escalate(TerminalReason.UNAUTHORIZED,dx.recommended_action);return m.result()
            m.advance("authorized")
            if transport.execute(session,dx.recommended_action)!=OperationStatus.SUCCEEDED:m.escalate(TerminalReason.RECOVERY_FAILED,"restart");return m.result()
            m.advance("restarted")
            if diagnose(transport.snapshot(session)).fault!=FaultCode.NONE:m.escalate(TerminalReason.RECOVERY_FAILED,"verify");return m.result()
            m.advance("verified");transport.disconnect(session);session=None;arm.retreat();m.advance("disconnected");m.terminal_reason=TerminalReason.RECOVERED;m.record("reported");return m.result()
        except Exception as e:arm.emergency_stop();base.emergency_stop();m.stop(TerminalReason.INTERNAL_ERROR,f"{type(e).__name__}:{e}");return m.result()
        finally:
            if session is not None:
                try:transport.disconnect(session)
                except Exception:pass
