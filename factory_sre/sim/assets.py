"""Version-pinned Go2/SO-101 acquisition and compatibility inspection."""

from __future__ import annotations

from .config import GO2_POLICY, GO2_POLICY_ENV, GO2_USD, SO101_ASSET, SO101_VERSION


def load_exact_assets() -> dict[str, object]:
    import antioch
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdPhysics

    stage = antioch.stage()
    assets_root = get_assets_root_path()
    stage_utils.add_reference_to_stage(usd_path=assets_root + GO2_USD, path="/World/Go2")
    antioch.load_asset(SO101_ASSET, prim_path="/World/SO101", version=SO101_VERSION)
    stage.Load()
    for _ in range(3):
        app_utils.update_app()
    joints = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    roots = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    return {
        "go2_valid": stage.GetPrimAtPath("/World/Go2").IsValid(),
        "so101_valid": stage.GetPrimAtPath("/World/SO101").IsValid(),
        "articulation_roots": roots,
        "go2_joints": [path for path in joints if path.startswith("/World/Go2")],
        "so101_joints": [path for path in joints if path.startswith("/World/SO101")],
        "policy_path": assets_root + GO2_POLICY,
        "policy_env_path": assets_root + GO2_POLICY_ENV,
        "go2_usd": assets_root + GO2_USD,
        "so101_asset": f"{SO101_ASSET}@{SO101_VERSION}",
    }


def make_go2_policy(policy_path: str, env_path: str):
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy

    return Go2FlatTerrainPolicy(prim_path="/World/Go2", policy_path=policy_path, env_config_path=env_path)
