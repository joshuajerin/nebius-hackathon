"""Simulator-free public contracts for the three bundle registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias


class BundleId(StrEnum):
    MOBILE_MANIPULATOR = "mobile_manipulator"
    WORKCELL = "workcell"
    WAREHOUSE = "warehouse"


@dataclass(frozen=True, slots=True)
class MobileManipulatorConfiguration:
    chassis_to_arm_base_m: tuple[float, float, float] = (0.0, 0.0, 0.205)
    mount_plate_size_m: tuple[float, float, float] = (0.160, 0.120, 0.012)
    mount_plate_mass_kg: float = 0.22
    wrist_camera_translation_m: tuple[float, float, float] = (0.035, 0.0, 0.025)
    wrist_camera_rotation_xyz_deg: tuple[float, float, float] = (0.0, 90.0, 0.0)
    camera_focal_length_mm: float = 2.8
    camera_horizontal_aperture_mm: float = 4.8
    camera_vertical_aperture_mm: float = 3.6
    camera_clipping_range_m: tuple[float, float] = (0.03, 10.0)
    usb_tool_size_m: tuple[float, float, float] = (0.025, 0.018, 0.090)
    usb_tool_mass_kg: float = 0.08
    mount_measured: bool = False
    wrist_camera_calibrated: bool = False
    usb_tool_measured: bool = False


@dataclass(frozen=True, slots=True)
class WorkcellConfiguration:
    coarse_tag_id: int = 10
    coarse_tag_size_m: float = 0.150
    fine_tag_id: int = 20
    fine_tag_size_m: float = 0.040
    fine_tag_to_port_m: tuple[float, float, float] = (-0.060, 0.0, 0.0)
    table_size_m: tuple[float, float, float] = (1.80, 0.90, 0.10)
    table_height_m: float = 0.75
    panel_size_m: tuple[float, float, float] = (0.55, 0.035, 0.30)
    panel_measured: bool = False
    tag_to_port_calibrated: bool = False

    def __post_init__(self) -> None:
        if self.coarse_tag_size_m != 0.150:
            raise ValueError("coarse AprilTag must be 150 mm")
        if self.fine_tag_size_m != 0.040:
            raise ValueError("fine AprilTag must be 40 mm")
        if self.fine_tag_to_port_m != (-0.060, 0.0, 0.0):
            raise ValueError("USB-C port must be exactly 60 mm left of the fine tag")


@dataclass(frozen=True, slots=True)
class WarehouseConfiguration:
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


BundleConfiguration: TypeAlias = (
    MobileManipulatorConfiguration | WorkcellConfiguration | WarehouseConfiguration
)


@dataclass(frozen=True, slots=True)
class BundleSpec:
    bundle_id: BundleId
    catalog_name: str
    version: str
    default_prim_path: str
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BundleLoadReport:
    bundle_id: BundleId
    root_path: str
    prim_paths: dict[str, str]
    asset_sources: dict[str, str]
    deployment_ready: bool
    warnings: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

