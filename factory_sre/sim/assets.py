"""Load and inspect the exact robot assets used by Factory SRE."""

from __future__ import annotations

from typing import Any

from factory_sre.contracts import AssetPins


def load_robot_assets(*, pins: AssetPins = AssetPins()) -> dict[str, Any]:
    """Load the policy-backed Go2 and pinned Antioch SO-101."""

    import antioch
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.storage.native import get_assets_root_path

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac asset root is unavailable")

    go2_usd = f"{assets_root.rstrip('/')}{pins.go2_relative_usd}"
    go2 = Go2FlatTerrainPolicy(
        prim_path="/World/Go2",
        usd_path=go2_usd,
        position=[0.0, 0.0, 0.5],
    )
    so101 = antioch.load_asset(
        pins.so101_name,
        prim_path="/World/SO101",
        version=pins.so101_version,
    )
    return {"go2_controller": go2, "so101": so101, "go2_usd": go2_usd}


def inspect_robot_stage() -> dict[str, Any]:
    """Return articulation roots and joint paths without assuming asset layout."""

    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics

    stage = get_current_stage()
    articulation_roots: list[str] = []
    joints: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(path)

    return {
        "articulation_roots": sorted(articulation_roots),
        "go2_joints": sorted(path for path in joints if path.startswith("/World/Go2")),
        "so101_joints": sorted(path for path in joints if path.startswith("/World/SO101")),
    }
