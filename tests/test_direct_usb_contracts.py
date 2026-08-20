from factory_sre.direct_usb.contracts import ASSETS, VISION_CELL


def test_direct_usb_manifest_pins_the_approved_assets() -> None:
    paths = {asset.name: asset.path for asset in ASSETS}
    assert paths["go2"] == "/Isaac/Robots/Unitree/Go2/go2.usd"
    assert paths["go2-policy"] == "/Isaac/Samples/Policies/go2/physx_policy.pt"
    assert paths["warehouse"] == "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    assert paths["ur5e"] == "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
    assert next(asset for asset in ASSETS if asset.name == "so101").version == "1.3.2"


def test_direct_usb_calibration_geometry_is_explicit() -> None:
    assert VISION_CELL.coarse_tag_size_m == 0.150
    assert VISION_CELL.fine_tag_size_m == 0.040
    assert VISION_CELL.fine_tag_to_usb_port.translation == (-0.060, 0.0, 0.0)
    assert len(VISION_CELL.permitted_actions) == 3

