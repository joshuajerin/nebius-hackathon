"""Typed, simulator-independent contracts for the Factory SRE mission."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetPins:
    """Robot assets whose compatibility must be proven before integration."""

    so101_name: str = "so101_antioch"
    so101_version: str = "1.3.2"
    # Isaac Sim 6's policy wrapper expects the multi-physics Menagerie asset.
    # The older /Isaac/Robots/Unitree/Go2 asset composes visually but does not
    # expose an articulation in the Antioch PhysX runtime.
    go2_relative_usd: str = "/Isaac/Samples/Mujoco_Menagerie/unitree_go2/go2/go2.usda"


@dataclass(frozen=True)
class WorkbenchManifest:
    """Identity and geometric contract for one serviceable factory cell."""

    asset_id: str
    tag_id: int
    approach_xyz: tuple[float, float, float]
    tag_to_dock_xyz: tuple[float, float, float] = (-0.30, 0.0, 0.0)


WORKBENCHES = (
    WorkbenchManifest(asset_id="vision-cell-a", tag_id=10, approach_xyz=(1.5, -1.5, 0.0)),
    WorkbenchManifest(asset_id="vision-cell-b", tag_id=20, approach_xyz=(1.5, 1.5, 0.0)),
)
