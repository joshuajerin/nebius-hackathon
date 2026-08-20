"""Fast robot-only probe used to resolve exact Go2 and SO-101 mount bodies."""

from __future__ import annotations

import json


def _describe(stage, root_path: str) -> dict[str, object]:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    prims = list(Usd.PrimRange(root))
    joints = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        joints.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "body0": [str(path) for path in joint.GetBody0Rel().GetTargets()],
                "body1": [str(path) for path in joint.GetBody1Rel().GetTargets()],
            }
        )
    return {
        "valid": root.IsValid(),
        "articulation_roots": [
            str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ],
        "rigid_bodies": [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)],
        "joints": joints,
    }


def main() -> None:
    import antioch

    antioch.boot(physics_engine="physx")

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.storage.native import get_assets_root_path

    stage = antioch.stage()
    root = get_assets_root_path()
    stage_utils.add_reference_to_stage(
        usd_path=root + "/Isaac/Samples/Policies/go2/go2.usda", path="/World/RobotProbe/Go2"
    )
    antioch.load_asset("so101_antioch", prim_path="/World/RobotProbe/SO101", version="1.3.2")
    stage.Load()
    for _ in range(4):
        app_utils.update_app()
    report = {
        "go2": _describe(stage, "/World/RobotProbe/Go2"),
        "so101": _describe(stage, "/World/RobotProbe/SO101"),
    }
    print("FACTORY_SRE_ROBOT_STRUCTURE_BEGIN")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("FACTORY_SRE_ROBOT_STRUCTURE_END")


if __name__ == "__main__":
    main()
