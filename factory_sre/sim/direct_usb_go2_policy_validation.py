"""Asset-only construction check for the pinned Go2 PhysX policy files."""

from __future__ import annotations


def main() -> None:
    import antioch

    antioch.boot(physics_engine="physx", physics_dt=1 / 120, render_dt=1 / 30)

    import inspect
    import json

    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.storage.native import get_assets_root_path

    root = get_assets_root_path()
    stage_utils.add_reference_to_stage(
        usd_path=root + "/Isaac/Samples/Policies/go2/go2.usda",
        path="/World/FactorySREPolicyProbe/Go2",
    )
    antioch.stage().Load()
    for _ in range(3):
        app_utils.update_app()
    policy = Go2FlatTerrainPolicy(
        prim_path="/World/FactorySREPolicyProbe/Go2",
        policy_path=root + "/Isaac/Samples/Policies/go2/physx_policy.pt",
        env_config_path=root + "/Isaac/Samples/Policies/go2/physx_env.yaml",
    )
    print(
        json.dumps(
            {
                "constructed": policy is not None,
                "robot_usd": root + "/Isaac/Samples/Policies/go2/go2.usda",
                "policy": root + "/Isaac/Samples/Policies/go2/physx_policy.pt",
                "environment": root + "/Isaac/Samples/Policies/go2/physx_env.yaml",
                "forward_signature": str(inspect.signature(policy.forward)),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

