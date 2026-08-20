"""List the live Isaac 6.0 Unitree asset tree and candidate policy files."""


def main() -> None:
    import antioch

    antioch.boot(viewport=(640, 480), render_quality="performance")
    import omni.client
    from isaacsim.storage.native import get_assets_root_path
    from pxr import Usd, UsdPhysics

    root = get_assets_root_path()
    for relative in (
        "/Isaac/Robots/Unitree",
        "/Isaac/Robots/Unitree/Go2",
        "/Isaac/Robots/Unitree/Go2/Go2",
        "/Isaac/Samples/Policies/go2",
    ):
        url = root + relative
        result, entries = omni.client.list(url)
        print(f"LIST {relative}: {result}")
        for entry in entries:
            print(f"  {entry.relative_path}")
    for relative in (
        "/Isaac/Robots/Unitree/Go2/go2.usd",
        "/Isaac/Robots/Unitree/Go2/go2.usda",
        "/Isaac/Robots/Unitree/Go2/Go2/go2.usd",
        "/Isaac/Robots/Unitree/Go2/Go2/go2.usda",
    ):
        result, entry = omni.client.stat(root + relative)
        print(f"STAT {relative}: {result} {entry}")
    for relative in ("/Isaac/Robots/Unitree/Go2/go2.usd", "/Isaac/Samples/Policies/go2/go2.usda"):
        stage = Usd.Stage.Open(root + relative)
        stage.Load()
        prims = list(stage.Traverse())
        joints = [str(prim.GetPath()) for prim in prims if prim.IsA(UsdPhysics.Joint)]
        roots = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        print(f"OPEN {relative}: default={stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else None} prims={len(prims)} roots={roots} joints={len(joints)}")
        print(f"  first joints={joints[:20]}")


if __name__ == "__main__":
    main()
