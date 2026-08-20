"""Remote validation entrypoint for the composed Factory SRE asset bundle."""

from __future__ import annotations

import json


def main() -> None:
    import antioch

    antioch.boot(physics_engine="physx", physics_dt=1 / 120, render_dt=1 / 30)

    from factory_sre.sim.direct_usb_asset_bundle import (
        PROTOTYPE_MOUNT,
        PROTOTYPE_WRIST_CAMERA,
        build_direct_usb_asset_bundle,
    )

    report = build_direct_usb_asset_bundle(PROTOTYPE_MOUNT, PROTOTYPE_WRIST_CAMERA)
    expected_root = "/World/FactorySRE/Go2/Geometry/base"
    report["checks"] = {
        "all_required_assets_resolved": all(report["valid_prims"].values()),
        "exactly_one_composite_articulation_root": report["articulation_roots"] == [expected_root],
        "go2_12_plus_so101_6_plus_mount_joint": len(report["active_joints"]) == 19,
        "plain_wrist_camera_used": report["camera"]["camera"].endswith("/Camera"),
        "physical_measurements_ready": report["physical_measurements_ready"],
    }
    print("FACTORY_SRE_ASSET_BUNDLE_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("FACTORY_SRE_ASSET_BUNDLE_END")

    structural = {key: value for key, value in report["checks"].items() if key != "physical_measurements_ready"}
    if not all(structural.values()):
        raise SystemExit("asset bundle failed structural validation")


if __name__ == "__main__":
    main()
