"""Immutable bundle identities and catalog pins."""

from __future__ import annotations

from types import MappingProxyType

from .contracts import BundleId, BundleSpec


BUNDLE_SPECS = MappingProxyType(
    {
        BundleId.MOBILE_MANIPULATOR: BundleSpec(
            bundle_id=BundleId.MOBILE_MANIPULATOR,
            catalog_name="factory-sre/mobile-manipulator",
            version="0.1.0",
            default_prim_path="/World/MobileManipulator",
            components=(
                "go2",
                "go2_physx_policy",
                "so101_antioch@1.3.2",
                "d455",
                "nanoscan3",
                "plain_wrist_camera",
                "prototype_mount",
                "prototype_usb_tool",
            ),
        ),
        BundleId.WORKCELL: BundleSpec(
            bundle_id=BundleId.WORKCELL,
            catalog_name="factory-sre/workcell",
            version="0.1.0",
            default_prim_path="/World/Workcell",
            components=(
                "ur5e",
                "inspector83x",
                "prototype_workbench",
                "direct_usb_panel",
                "coarse_apriltag_150mm",
                "fine_apriltag_40mm",
            ),
        ),
        BundleId.WAREHOUSE: BundleSpec(
            bundle_id=BundleId.WAREHOUSE,
            catalog_name="factory-sre/warehouse",
            version="0.1.0",
            default_prim_path="/World/Warehouse",
            components=("warehouse_multiple_shelves",),
        ),
    }
)


def get_bundle_spec(bundle_id: BundleId | str) -> BundleSpec:
    return BUNDLE_SPECS[BundleId(bundle_id)]

