"""Remote Isaac 6.0.1 probe for every stock Factory SRE asset.

Run with ``uv run antioch run factory_sre/sim/direct_usb_asset_probe.py``.
The module intentionally keeps all simulator imports below ``antioch.boot``.
"""

from __future__ import annotations

import json


STOCK_ASSETS = {
    "go2_policy_robot": "/Isaac/Samples/Policies/go2/go2.usda",
    "go2_description": "/Isaac/Robots/Unitree/Go2/go2.usd",
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "ur5e": "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "d455": "/Isaac/Sensors/Realsense/D455/rsd455.usd",
    "p61x": "/Isaac/Sensors/SICK/InspectorP61x/SICK_InspectorP61x.usd",
    "nanoscan3": "/Isaac/Sensors/SICK/nanoScan3/SICK_nanoScan3.usd",
    "inspector83x": "/Isaac/Sensors/SICK/Inspector83x/SICK_Inspector83x.usd",
}


def _inspect_subtree(stage, root_path: str) -> dict[str, object]:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    prims = list(UsdGeom.PrimRange(root)) if root.IsValid() else []
    joints = [str(prim.GetPath()) for prim in prims if prim.IsA(UsdPhysics.Joint)]
    articulation_roots = [
        str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    rigid_bodies = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    unresolved = []
    for prim in prims:
        if prim.HasAuthoredReferences() and not prim.GetPrimStack():
            unresolved.append(str(prim.GetPath()))
    return {
        "valid": root.IsValid(),
        "active": root.IsActive() if root.IsValid() else False,
        "prim_count": len(prims),
        "joints": joints,
        "articulation_roots": articulation_roots,
        "rigid_bodies": rigid_bodies,
        "collision_prim_count": len(colliders),
        "unresolved_references": unresolved,
    }


def main() -> None:
    import antioch

    antioch.boot(physics_engine="physx", viewport=(1280, 720))

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.storage.native import get_assets_root_path

    stage = antioch.stage()
    assets_root = get_assets_root_path()
    report: dict[str, object] = {"assets_root": assets_root, "stock": {}}

    for index, (name, relative_path) in enumerate(STOCK_ASSETS.items()):
        prim_path = f"/World/AssetProbe/A{index}_{name}"
        stage_utils.add_reference_to_stage(usd_path=assets_root + relative_path, path=prim_path)
        report["stock"][name] = {
            "relative_path": relative_path,
            "resolved_path": assets_root + relative_path,
            "prim_path": prim_path,
        }

    antioch.load_asset("so101_antioch", prim_path="/World/AssetProbe/SO101", version="1.3.2")
    stage.Load()
    for _ in range(8):
        app_utils.update_app()

    for name, item in report["stock"].items():
        item.update(_inspect_subtree(stage, item["prim_path"]))
    report["so101"] = {
        "name": "so101_antioch",
        "version": "1.3.2",
        **_inspect_subtree(stage, "/World/AssetProbe/SO101"),
    }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
