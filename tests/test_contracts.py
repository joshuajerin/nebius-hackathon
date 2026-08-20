from math import sqrt

import pytest

from factory_sre.contracts import REGISTRY, Transform3D, VISION_CELL, WorkbenchManifest


def test_registry_is_one_deployable_workcell() -> None:
    assert len(REGISTRY.workbenches) == 1
    assert VISION_CELL.asset_id == "vision-cell-37"
    assert VISION_CELL.coarse_tag_size_m == 0.150
    assert VISION_CELL.fine_tag_size_m == 0.040
    assert VISION_CELL.fine_tag_to_usb_port.translation[0] < 0
    assert REGISTRY.asset("so101").version == "1.3.2"
    assert {asset.name for asset in REGISTRY.assets} == {
        "go2", "go2-physx-policy", "go2-physx-policy-env", "so101", "warehouse",
        "ur5e", "d455", "p61x", "nanoscan3", "inspector83x",
    }


def test_non_left_port_is_rejected() -> None:
    with pytest.raises(ValueError, match="left"):
        WorkbenchManifest(
            "bad", "tag36h11", 1, 0.150, "tag36h11", 2, 0.040,
            VISION_CELL.approach_pose, Transform3D((0.01, 0.0, 0.0)),
            VISION_CELL.diagnostic_target, frozenset(),
        )


def test_transform_composition_rotates_port_offset() -> None:
    parent = Transform3D((1, 2, 3), (sqrt(0.5), 0, 0, sqrt(0.5)))
    assert parent.compose(Transform3D((-0.06, 0, 0))).translation == pytest.approx((1, 1.94, 3))
