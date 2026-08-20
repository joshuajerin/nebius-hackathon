"""Fail-closed mission controller for calibrated AprilTag-to-USB insertion."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from factory_sre.contracts import DiagnosticSession, FaultCode, OperationStatus

from .contracts import (
    Arm,
    Camera,
    Locomotion,
    MissionState,
    TerminalReason,
    Transport,
    Workcell,
)
from .diagnosis import diagnose


FLOW = (
    MissionState.DISPATCH,
    MissionState.NAVIGATE,
    MissionState.STABILIZE,
    MissionState.LOCALIZE,
    MissionState.APPROACH,
    MissionState.INSERT,
    MissionState.ENUMERATE,
    MissionState.DIAGNOSE,
    MissionState.RECOVER,
    MissionState.VERIFY,
    MissionState.DISCONNECT,
    MissionState.REPORT,
)


@dataclass(frozen=True, slots=True)
class MissionEvent:
    state: MissionState
    elapsed_s: float
    detail: str


@dataclass(frozen=True, slots=True)
class MissionResult:
    mission_id: str
    asset_id: str
    success: bool
    terminal_state: MissionState
    terminal_reason: TerminalReason
    insertion_attempts: int
    events: tuple[MissionEvent, ...]


@dataclass(slots=True)
class _Trace:
    mission_id: str
    asset_id: str
    state: MissionState = MissionState.DISPATCH
    reason: TerminalReason | None = None
    insertion_attempts: int = 0
    events: list[MissionEvent] = field(default_factory=list)
    started: float = field(default_factory=monotonic)

    def record(self, detail: str) -> None:
        self.events.append(MissionEvent(self.state, monotonic() - self.started, detail))

    def enter(self, state: MissionState, detail: str) -> None:
        self.state = state
        self.record(detail)

    def fail(self, reason: TerminalReason, detail: str, *, emergency: bool = False) -> None:
        self.reason = reason
        self.enter(MissionState.EMERGENCY_STOP if emergency else MissionState.ESCALATE, detail)

    def result(self) -> MissionResult:
        reason = self.reason or TerminalReason.INTERNAL_ERROR
        return MissionResult(
            mission_id=self.mission_id,
            asset_id=self.asset_id,
            success=reason == TerminalReason.RECOVERED,
            terminal_state=self.state,
            terminal_reason=reason,
            insertion_attempts=self.insertion_attempts,
            events=tuple(self.events),
        )


class DirectUsbMissionController:
    def __init__(self, *, maximum_force_n: float = 8.0, minimum_depth_m: float = 0.005) -> None:
        self.maximum_force_n = maximum_force_n
        self.minimum_depth_m = minimum_depth_m

    @staticmethod
    def _tag_pose(
        camera: Camera, tag_id: int, size_m: float
    ) -> tuple[object | None, bool]:
        detections = camera.detect_tags("tag36h11", size_m)
        matching = [pose for observed_id, pose in detections if observed_id == tag_id]
        wrong_identity = bool(detections) and not matching
        return (matching[0] if matching else None), wrong_identity

    def run(
        self,
        mission_id: str,
        workcell: Workcell,
        locomotion: Locomotion,
        arm: Arm,
        camera: Camera,
        transport: Transport,
    ) -> MissionResult:
        trace = _Trace(mission_id, workcell.asset_id)
        trace.record("incident accepted")
        session: DiagnosticSession | None = None
        inserted = False
        retreated = False
        try:
            trace.enter(MissionState.NAVIGATE, "traveling to calibrated staging pose")
            if not locomotion.navigate(workcell.approach_pose):
                trace.fail(TerminalReason.NAVIGATION_FAILED, "staging pose unreachable")
                return trace.result()

            trace.enter(MissionState.STABILIZE, "holding stationary manipulation stance")
            if not locomotion.stabilize() or not locomotion.stable():
                locomotion.emergency_stop()
                arm.emergency_stop()
                trace.fail(TerminalReason.BASE_UNSTABLE, "base failed stability gate", emergency=True)
                return trace.result()

            trace.enter(MissionState.LOCALIZE, "acquiring 150 mm coarse AprilTag")
            coarse_pose, wrong_coarse = self._tag_pose(
                camera, workcell.coarse_tag_id, workcell.coarse_tag_size_m
            )
            if wrong_coarse:
                trace.fail(TerminalReason.WRONG_ASSET, "coarse AprilTag identity mismatch")
                return trace.result()
            if coarse_pose is None:
                trace.fail(TerminalReason.PERCEPTION_FAILED, "coarse AprilTag not visible")
                return trace.result()

            trace.enter(MissionState.APPROACH, "moving SO-101 to fine-localization standoff")
            if not arm.move_to_standoff():
                trace.fail(TerminalReason.PERCEPTION_FAILED, "arm could not reach standoff")
                return trace.result()

            fine_pose, wrong_fine = self._tag_pose(camera, workcell.fine_tag_id, workcell.fine_tag_size_m)
            if wrong_fine:
                trace.fail(TerminalReason.WRONG_ASSET, "fine AprilTag identity mismatch")
                return trace.result()
            if fine_pose is None:
                trace.fail(TerminalReason.PERCEPTION_FAILED, "fine AprilTag lost")
                return trace.result()
            usb_port_pose = fine_pose.compose(workcell.fine_tag_to_usb_port)
            if not arm.visual_servo(usb_port_pose):
                trace.fail(TerminalReason.PERCEPTION_FAILED, "visual servo failed")
                return trace.result()

            trace.enter(MissionState.INSERT, "performing compliant guarded insertion")
            trace.insertion_attempts += 1
            observation = arm.guarded_insert(usb_port_pose)
            if observation.axial_force_n > self.maximum_force_n:
                locomotion.emergency_stop()
                arm.emergency_stop()
                trace.fail(TerminalReason.FORCE_LIMIT, "insertion force limit exceeded", emergency=True)
                return trace.result()
            if observation.asset_id != workcell.asset_id:
                trace.fail(TerminalReason.WRONG_ASSET, "connector endpoint identity mismatch")
                return trace.result()
            if not observation.contact or observation.depth_m < self.minimum_depth_m:
                trace.fail(TerminalReason.ENUMERATION_FAILED, "connector seating criteria not met")
                return trace.result()
            inserted = True

            trace.enter(MissionState.ENUMERATE, "waiting for CDC-NCM/serial USB enumeration")
            session = transport.enumerate(mission_id, workcell.asset_id)
            if session.asset_id != workcell.asset_id:
                trace.fail(TerminalReason.WRONG_ASSET, "enumerated USB identity mismatch")
                return trace.result()

            trace.enter(MissionState.DIAGNOSE, "combining target status and permitted procedures")
            diagnosis = diagnose(transport.snapshot(session))
            if not diagnosis.autonomous or diagnosis.action is None:
                trace.fail(TerminalReason.UNSUPPORTED_DIAGNOSIS, diagnosis.fault.value)
                return trace.result()
            if diagnosis.action not in workcell.permitted_actions:
                trace.fail(TerminalReason.UNAUTHORIZED, diagnosis.action)
                return trace.result()

            trace.enter(MissionState.RECOVER, diagnosis.action)
            if transport.execute(session, diagnosis.action) != OperationStatus.SUCCEEDED:
                trace.fail(TerminalReason.RECOVERY_FAILED, f"{diagnosis.action} did not succeed")
                return trace.result()

            trace.enter(MissionState.VERIFY, "checking heartbeat and full target health")
            if diagnose(transport.snapshot(session)).fault != FaultCode.NONE:
                trace.fail(TerminalReason.VERIFICATION_FAILED, "post-recovery health check failed")
                return trace.result()

            trace.enter(MissionState.DISCONNECT, "closing diagnostic lease and withdrawing connector")
            transport.disconnect(session)
            session = None
            if not arm.retreat():
                trace.fail(TerminalReason.INTERNAL_ERROR, "connector retreat failed")
                return trace.result()
            retreated = True
            inserted = False

            trace.reason = TerminalReason.RECOVERED
            trace.enter(MissionState.REPORT, "recovery verified and ticket ready to close")
            return trace.result()
        except Exception as exc:
            locomotion.emergency_stop()
            arm.emergency_stop()
            trace.fail(TerminalReason.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}", emergency=True)
            return trace.result()
        finally:
            if session is not None:
                try:
                    transport.disconnect(session)
                except Exception:
                    pass
            if inserted and not retreated:
                try:
                    arm.retreat()
                except Exception:
                    pass

