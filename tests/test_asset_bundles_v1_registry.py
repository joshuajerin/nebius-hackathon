from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from factory_sre.sim.asset_bundles_v1 import BundleId, get_bundle_spec
from factory_sre.sim.asset_bundles_v1.contracts import WorkcellConfiguration
from factory_sre.sim.asset_bundles_v1.loaders import ISAAC_PATHS


MANIFEST_ROOT = Path("assets/direct_usb/bundles_v1")


def test_registry_exposes_exactly_three_bundles() -> None:
    assert set(BundleId) == {
        BundleId.MOBILE_MANIPULATOR,
        BundleId.WORKCELL,
        BundleId.WAREHOUSE,
    }
    assert get_bundle_spec(BundleId.MOBILE_MANIPULATOR).catalog_name == (
        "factory-sre/mobile-manipulator"
    )
    assert get_bundle_spec(BundleId.WORKCELL).catalog_name == "factory-sre/workcell"
    assert get_bundle_spec(BundleId.WAREHOUSE).catalog_name == "factory-sre/warehouse"


def test_registry_import_is_simulator_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import factory_sre.sim.asset_bundles_v1; "
            "assert 'pxr' not in sys.modules; assert 'isaacsim' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_case_sensitive_isaac_paths_are_pinned() -> None:
    assert ISAAC_PATHS["d455"] == "/Isaac/Sensors/RealSense/D455/rsd455.usd"
    assert ISAAC_PATHS["warehouse"].endswith("warehouse_multiple_shelves.usd")
    assert ISAAC_PATHS["go2_policy"].endswith("physx_policy.pt")


def test_workcell_geometry_contract_is_fail_closed() -> None:
    config = WorkcellConfiguration()
    assert config.coarse_tag_size_m == 0.150
    assert config.fine_tag_size_m == 0.040
    assert config.fine_tag_to_port_m == (-0.060, 0.0, 0.0)
    with pytest.raises(ValueError, match="60 mm"):
        WorkcellConfiguration(fine_tag_to_port_m=(-0.055, 0.0, 0.0))


@pytest.mark.parametrize(
    ("filename", "bundle_id", "catalog_name"),
    (
        ("mobile-manipulator.json", "mobile_manipulator", "factory-sre/mobile-manipulator"),
        ("workcell.json", "workcell", "factory-sre/workcell"),
        ("warehouse.json", "warehouse", "factory-sre/warehouse"),
    ),
)
def test_catalog_manifests_match_the_registry(filename, bundle_id, catalog_name) -> None:
    manifest = json.loads((MANIFEST_ROOT / filename).read_text())
    spec = get_bundle_spec(bundle_id)
    assert manifest["name"] == catalog_name == spec.catalog_name
    assert manifest["version"] == spec.version == "0.1.0"
    assert manifest["bundle_id"] == spec.bundle_id.value
    assert manifest["default_prim_path"] == spec.default_prim_path
