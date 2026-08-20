"""Self-contained Antioch scenarios for the isolated direct-USB implementation."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import antioch


FAULT_CASES = tuple(
    antioch.case({"fault": fault}, id=fault.lower().replace("_", "-"), tags=("hero", "direct-usb"))
    for fault in (
        "VISION_SERVICE_CRASHED",
        "NETWORK_INTERFACE_MISCONFIGURED",
        "CAMERA_UNREACHABLE",
    )
)


@antioch.scenario(
    name="direct_usb_three_fault_recovery",
    description="Recover all three whitelisted failures through a device-mode USB-C endpoint.",
    tags=("hero", "direct-usb", "recovery"),
    cases=FAULT_CASES,
    sim=None,
    capture=False,
)
def direct_usb_three_fault_recovery(run: antioch.ScenarioRun, fault: str) -> None:
    from factory_sre.contracts import FaultCode, Transform3D
    from factory_sre.target.ledger import OperationLedger

    from .adapters import DryArm, DryCamera, DryLocomotion
    from .contracts import VISION_CELL
    from .mission import DirectUsbMissionController, FLOW
    from .runtime import DirectUsbTargetRuntime
    from .transport import InProcessDirectUsbTransport

    with TemporaryDirectory(prefix="factory-sre-direct-usb-") as directory:
        ledger = OperationLedger(f"{directory}/operations.sqlite3")
        ledger.open()
        try:
            runtime = DirectUsbTargetRuntime(ledger)
            runtime.inject_fault(FaultCode(fault))
            camera = DryCamera(
                {
                    0.150: ((VISION_CELL.coarse_tag_id, Transform3D((2.0, 0.8, 0.9))),),
                    0.040: ((VISION_CELL.fine_tag_id, Transform3D((0.45, 0.0, 0.25))),),
                }
            )
            result = DirectUsbMissionController().run(
                f"direct-usb-{fault.lower()}",
                VISION_CELL,
                DryLocomotion(),
                DryArm(VISION_CELL.asset_id),
                camera,
                InProcessDirectUsbTransport(runtime),
            )
            states = [event.state.value for event in result.events]
            run.check("mission recovered", result.success, detail=result.terminal_reason.value)
            run.check("one guarded insertion", result.insertion_attempts == 1, detail=str(result.insertion_attempts))
            run.check(
                "all mission states recorded",
                all(state.value in states for state in FLOW),
                detail=str(states),
            )
            run.check("USB lease closed", not runtime.diagnostic_open, detail=f"epoch {runtime.link_epoch}")
            run.check("one recovery mutation", ledger.count() == 1, detail=f"operations {ledger.count()}")
            run.add_result(
                "direct_usb_mission",
                {
                    "fault": fault,
                    "terminal_reason": result.terminal_reason.value,
                    "states": states,
                    "link_epoch": runtime.link_epoch,
                },
            )
        finally:
            ledger.close()

