from __future__ import annotations

import pytest

from factory_sre.contracts import (
    FaultCode,
    Incident,
    TerminalReason,
    Transform3D,
    WORKBENCH_A,
    WORKBENCH_B,
)
from factory_sre.hardware import DryRunArmAdapter, DryRunBaseAdapter, DryRunCameraAdapter
from factory_sre.mission import FLOW, MissionController
from factory_sre.target.ledger import OperationLedger
from factory_sre.target.runtime import DiagnosticTargetRuntime
from factory_sre.transport import RuntimeDiagnosticTransport


def run_mission(tmp_path, manifest, *, observed_tag=None, force=2.0, fault=FaultCode.VISION_SERVICE_CRASHED):
    ledger = OperationLedger(tmp_path / f"{manifest.asset_id}.sqlite3")
    ledger.open()
    runtime = DiagnosticTargetRuntime(ledger)
    transport = RuntimeDiagnosticTransport(runtime)
    transport.inject_fault(manifest.asset_id, fault)
    tag_id = manifest.tag_id if observed_tag is None else observed_tag
    result = MissionController().run(
        Incident(f"mission-{manifest.asset_id}", manifest.asset_id, fault.value),
        manifest,
        DryRunBaseAdapter(),
        DryRunArmAdapter(asset_id=manifest.asset_id, measured_force_n=force),
        DryRunCameraAdapter({0.120: ((tag_id, Transform3D((2.0, 0.8, 0.9))),)}),
        transport,
    )
    count = ledger.count()
    listener_open = runtime.diagnostic_open
    ledger.close()
    return result, count, listener_open


@pytest.mark.parametrize("manifest", (WORKBENCH_A, WORKBENCH_B))
def test_both_incident_targets_complete_the_full_whitelisted_flow(tmp_path, manifest) -> None:
    result, mutation_count, listener_open = run_mission(tmp_path, manifest)
    states = [event.state for event in result.events]
    assert result.success
    assert result.terminal_reason == TerminalReason.RECOVERED
    assert result.dock_attempts == 1
    assert all(state in states for state in FLOW)
    assert mutation_count == 1
    assert not listener_open


def test_neighbor_tag_never_redirects_or_docks(tmp_path) -> None:
    result, mutation_count, listener_open = run_mission(tmp_path, WORKBENCH_A, observed_tag=WORKBENCH_B.tag_id)
    assert result.terminal_reason == TerminalReason.WRONG_ASSET
    assert result.dock_attempts == 0
    assert mutation_count == 0
    assert not listener_open


def test_excess_force_emergency_stops_before_mutation(tmp_path) -> None:
    result, mutation_count, listener_open = run_mission(tmp_path, WORKBENCH_B, force=8.01)
    assert result.terminal_reason == TerminalReason.FORCE_LIMIT
    assert mutation_count == 0
    assert not listener_open


def test_safety_fault_escalates_without_mutation(tmp_path) -> None:
    result, mutation_count, listener_open = run_mission(tmp_path, WORKBENCH_B, fault=FaultCode.SAFETY_CIRCUIT_OPEN)
    assert result.terminal_reason == TerminalReason.UNSUPPORTED_DIAGNOSIS
    assert mutation_count == 0
    assert not listener_open
