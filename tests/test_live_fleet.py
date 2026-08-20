from __future__ import annotations

import pytest

from factory_sre.live.fleet import FleetCoordinator


def test_three_failures_queue_and_recover_after_five_seconds_each() -> None:
    fleet = FleetCoordinator(contact_hold_seconds=5.0)
    for asset_id in ("service-box-2", "service-box-5", "service-box-7"):
        fleet.fail(asset_id)

    assert fleet.snapshot()["pending_failures"] == [
        "service-box-2",
        "service-box-5",
        "service-box-7",
    ]

    sim_time = 10.0
    for asset_id in ("service-box-2", "service-box-5", "service-box-7"):
        fleet.update_robot(
            {
                "phase": "magnetic_contact",
                "current_target": asset_id,
                "planned_queue": [asset_id],
                "simulator_ready": True,
                "position": [0.0, 1.6, 0.5],
            }
        )
        fleet.update_contact(asset_id=asset_id, attached=True, sim_time_s=sim_time)
        before = fleet.update_contact(asset_id=asset_id, attached=True, sim_time_s=sim_time + 4.99)
        assert asset_id in before["pending_failures"]
        recovered = fleet.update_contact(asset_id=asset_id, attached=True, sim_time_s=sim_time + 5.0)
        assert asset_id not in recovered["pending_failures"]
        sim_time += 7.0

    assert fleet.snapshot()["summary"] == {"online": 8, "offline": 0, "recoveries": 3}


def test_contact_identity_must_match_active_target() -> None:
    fleet = FleetCoordinator()
    fleet.fail("service-box-3")
    fleet.update_robot({"phase": "magnetic_contact", "current_target": "service-box-2"})

    with pytest.raises(ValueError, match="online workstation"):
        fleet.update_contact(asset_id="service-box-2", attached=True, sim_time_s=1.0)

    with pytest.raises(ValueError, match="active target"):
        fleet.update_contact(asset_id="service-box-3", attached=True, sim_time_s=1.0)


def test_reset_clears_incidents_without_erasing_recovery_totals() -> None:
    fleet = FleetCoordinator()
    fleet.fail("service-box-8")
    fleet.reset()

    state = fleet.snapshot()
    assert state["summary"]["offline"] == 0
    assert state["pending_failures"] == []
    assert state["robot"]["planned_queue"] == ()
