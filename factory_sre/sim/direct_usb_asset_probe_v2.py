"""Second-pass remote asset probe with per-asset failure isolation."""

from __future__ import annotations

import json


STOCK_ASSETS = {
    "go2_policy_robot": "/Isaac/Samples/Policies/go2/go2.usda",
    "go2_description": "/Isaac/Robots/Unitree/Go2/go2.usd",
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "ur5e": "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "d455": "/Isaac/Sensors/RealSense/D455/rsd455.usd",
    "p61x": "/Isaac/Sensors/SICK/InspectorP61x/SICK_InspectorP61x.usd",
    "nanoscan3": "/Isaac/Sensors/SICK/nanoScan3/SICK_nanoScan3.usd",
    "inspector83x": "/Isaac/Sensors/SICK/Inspector83x/SICK_Inspector83x.usd",
}


def _inspect(stage, root_path: str) -> dict[str, object]:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    prims = list(UsdGeom.PrimRange(root)) if root.IsValid() else []
    return {
        "valid": root.IsValid(),
        "prim_count": len(prims),
        "joints": [str(p.GetPath()) for p in prims if p.IsA(UsdPhysics.Joint)],
        "articulation_roots": [
            str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ],
        "rigid_bodies": [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)],
        "collision_prim_count": sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims),
    }


def main() -> None:
    import antioch

    antioch.boot(physics_engine="physx", viewport=(1280, 720))

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.storage.native import get_assets_root_path

    stage = antioch.stage()
    root = get_assets_root_path()
    report: dict[str, object] = {"assets_root": root, "stock": {}}
    for index, (name, path) in enumerate(STOCK_ASSETS.items()):
        prim_path = f"/World/AssetProbeV2/A{index}_{name}"
        item: dict[str, object] = {"relative_path": path, "resolved_path": root + path}
        try:
            stage_utils.add_reference_to_stage(usd_path=root + path, path=prim_path)
            stage.Load()
            for _ in range(3):
                app_utils.update_app()
            item.update(_inspect(stage, prim_path))
        except Exception as exc:
            item.update(valid=False, error=f"{type(exc).__name__}: {exc}")
        report["stock"][name] = item

    try:
        antioch.load_asset("so101_antioch", prim_path="/World/AssetProbeV2/SO101", version="1.3.2")
        stage.Load()
        for _ in range(3):
            app_utils.update_app()
        report["so101"] = {
            "name": "so101_antioch",
            "version": "1.3.2",
            **_inspect(stage, "/World/AssetProbeV2/SO101"),
        }
    except Exception as exc:
        report["so101"] = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}

    print("FACTORY_SRE_ASSET_REPORT_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("FACTORY_SRE_ASSET_REPORT_END")


if __name__ == "__main__":
    main()
