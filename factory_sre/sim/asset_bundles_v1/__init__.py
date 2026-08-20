"""Agent-facing, independently loadable Factory SRE asset bundles."""

from .contracts import (
    BundleId,
    BundleLoadReport,
    MobileManipulatorConfiguration,
    WarehouseConfiguration,
    WorkcellConfiguration,
)
from .loaders import load_bundle, load_factory_sre_scene
from .registry import BUNDLE_SPECS, get_bundle_spec

__all__ = [
    "BUNDLE_SPECS",
    "BundleId",
    "BundleLoadReport",
    "MobileManipulatorConfiguration",
    "WarehouseConfiguration",
    "WorkcellConfiguration",
    "get_bundle_spec",
    "load_bundle",
    "load_factory_sre_scene",
]

