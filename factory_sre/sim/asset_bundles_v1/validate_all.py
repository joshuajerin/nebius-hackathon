"""Remote Isaac 6.0.1 acceptance probe for all three independent bundles."""

from __future__ import annotations

import json


def _subtree(stage, root_path: str):
    from pxr import Usd

    return list(Usd.PrimRange(stage.GetPrimAtPath(root_path)))


def main() -> None:
    import antioch

    antioch.boot(physics_engine="physx", physics_dt=1 / 120, render_dt=1 / 30)

    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from pxr import UsdGeom, UsdPhysics

    from factory_sre.sim.asset_bundles_v1 import BundleId, load_factory_sre_scene

    reports = load_factory_sre_scene("/World/FactorySREBundlesV1Validation")
    stage = antioch.stage()
    mobile = reports[BundleId.MOBILE_MANIPULATOR]
    workcell = reports[BundleId.WORKCELL]
    warehouse = reports[BundleId.WAREHOUSE]

    mobile_prims = _subtree(stage, mobile.root_path)
    mobile_roots = [
        str(prim.GetPath())
        for prim in mobile_prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    mobile_joints = [
        str(prim.GetPath())
        for prim in mobile_prims
        if prim.IsA(UsdPhysics.Joint) and prim.IsActive()
    ]
    mobile_policy = Go2FlatTerrainPolicy(
        prim_path=mobile.prim_paths["go2"],
        policy_path=mobile.asset_sources["go2_policy"],
        env_config_path=mobile.asset_sources["go2_policy_env"],
    )

    workcell_prims = _subtree(stage, workcell.root_path)
    workcell_roots = [
        str(prim.GetPath())
        for prim in workcell_prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    workcell_collisions = [
        str(prim.GetPath()) for prim in workcell_prims if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    cache = UsdGeom.XformCache()
    fine_position = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(workcell.prim_paths["fine_tag_frame"])
    ).ExtractTranslation()
    port_position = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(workcell.prim_paths["usb_port_frame"])
    ).ExtractTranslation()
    tag_to_port = tuple(float(port_position[i] - fine_position[i]) for i in range(3))

    warehouse_prims = _subtree(stage, warehouse.root_path)
    warehouse_collisions = [
        str(prim.GetPath()) for prim in warehouse_prims if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    all_prims = _subtree(stage, "/World/FactorySREBundlesV1Validation")
    all_roots = [
        str(prim.GetPath()) for prim in all_prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]

    checks = {
        "mobile_exactly_one_articulation_root": len(mobile_roots) == 1,
        "mobile_has_19_active_joints": len(mobile_joints) == 19,
        "mobile_go2_policy_constructs": mobile_policy is not None,
        "mobile_d455_valid": stage.GetPrimAtPath(mobile.prim_paths["d455"]).IsValid(),
        "mobile_nanoscan3_valid": stage.GetPrimAtPath(mobile.prim_paths["nanoscan3"]).IsValid(),
        "mobile_plain_wrist_camera_valid": stage.GetPrimAtPath(
            mobile.prim_paths["wrist_camera"]
        ).IsA(UsdGeom.Camera),
        "mobile_usb_tool_and_tip_valid": stage.GetPrimAtPath(
            mobile.prim_paths["usb_tool"]
        ).IsValid()
        and stage.GetPrimAtPath(mobile.prim_paths["usb_tip"]).IsValid(),
        "workcell_exactly_one_ur5e_articulation": len(workcell_roots) == 1,
        "workcell_has_collision_geometry": len(workcell_collisions) > 0,
        "workcell_camera_valid": stage.GetPrimAtPath(
            workcell.prim_paths["inspector83x"]
        ).IsValid(),
        "workcell_tags_valid": stage.GetPrimAtPath(workcell.prim_paths["coarse_tag"]).IsValid()
        and stage.GetPrimAtPath(workcell.prim_paths["fine_tag"]).IsValid(),
        "workcell_tag_sizes_exact": workcell.metadata["coarse_tag_size_m"] == 0.150
        and workcell.metadata["fine_tag_size_m"] == 0.040,
        "workcell_port_is_60mm_left_of_fine_tag": all(
            abs(actual - expected) < 1e-9
            for actual, expected in zip(tag_to_port, (-0.060, 0.0, 0.0), strict=True)
        ),
        "warehouse_reference_valid": stage.GetPrimAtPath(warehouse.root_path).IsValid(),
        "warehouse_has_collision_geometry": len(warehouse_collisions) > 0,
        "composition_has_only_mobile_and_ur5e_roots": len(all_roots) == 2
        and mobile_roots[0] in all_roots
        and workcell_roots[0] in all_roots,
        "bundle_roots_do_not_overlap": len({report.root_path for report in reports.values()}) == 3,
        "prototype_physical_bundles_fail_closed": not mobile.deployment_ready
        and not workcell.deployment_ready,
    }
    payload = {
        "checks": checks,
        "mobile": {
            "root": mobile.root_path,
            "articulation_roots": mobile_roots,
            "active_joint_count": len(mobile_joints),
            "deployment_ready": mobile.deployment_ready,
        },
        "workcell": {
            "root": workcell.root_path,
            "articulation_roots": workcell_roots,
            "collision_count": len(workcell_collisions),
            "fine_tag_to_port_m": tag_to_port,
            "deployment_ready": workcell.deployment_ready,
        },
        "warehouse": {
            "root": warehouse.root_path,
            "collision_count": len(warehouse_collisions),
            "deployment_ready": warehouse.deployment_ready,
        },
        "composition_articulation_roots": all_roots,
    }
    print("FACTORY_SRE_THREE_BUNDLE_VALIDATION_BEGIN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("FACTORY_SRE_THREE_BUNDLE_VALIDATION_END")
    if not all(checks.values()):
        raise SystemExit("three-bundle validation failed")


if __name__ == "__main__":
    main()

