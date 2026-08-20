from __future__ import annotations

import pytest

from factory_sre.contracts import Transform3D, WORKBENCH_A, WORKBENCH_B
from factory_sre.sim.docking import WrongAssetError, guarded_magnetic_dock, register_dock_target


def registered_target():
    tag = Transform3D((1.8, 0.64, 0.88))
    truth = Transform3D((1.5, 0.64, 0.88))
    return register_dock_target(
        WORKBENCH_A,
        observed_tag_id=WORKBENCH_A.tag_id,
        world_tag=tag,
        hidden_world_dock=truth,
    )


def test_world_tag_times_tag_to_dock_matches_hidden_truth() -> None:
    target = registered_target()
    assert target.world_dock.translation == pytest.approx((1.5, 0.64, 0.88))
    assert target.position_error_m < 1e-12
    assert target.angle_error_deg < 1e-12


def test_neighbor_tag_is_a_terminal_identity_failure() -> None:
    with pytest.raises(WrongAssetError, match="expected tag 10"):
        register_dock_target(
            WORKBENCH_A,
            observed_tag_id=WORKBENCH_B.tag_id,
            world_tag=Transform3D((4.0, 0.64, 0.88)),
            hidden_world_dock=Transform3D((3.7, 0.64, 0.88)),
        )


def test_magnetic_dock_engages_inside_pose_force_and_contact_gates() -> None:
    result = guarded_magnetic_dock(
        registered_target(),
        reached_pose=Transform3D((1.503, 0.64, 0.88)),
        axial_force_n=3.5,
        contact_depth_m=0.006,
        penetration_m=0.001,
    )
    assert result.engaged
    assert result.position_error_m == pytest.approx(0.003)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"axial_force_n": 8.01}, "FORCE_LIMIT"),
        ({"penetration_m": 0.011}, "PENETRATION_LIMIT"),
        ({"contact_depth_m": 0.004}, "NO_CONTACT"),
        ({"reached_pose": Transform3D((1.506, 0.64, 0.88))}, "POSITION_ERROR"),
    ),
)
def test_magnetic_dock_fails_closed(kwargs, reason) -> None:
    values = {
        "reached_pose": Transform3D((1.5, 0.64, 0.88)),
        "axial_force_n": 3.5,
        "contact_depth_m": 0.006,
        "penetration_m": 0.001,
    }
    values.update(kwargs)
    result = guarded_magnetic_dock(registered_target(), **values)
    assert not result.engaged
    assert result.terminal_reason == reason
