"""Pure-Python contracts for the eight-cell industrial demo.

This module intentionally has no Isaac imports so it remains safe during
Antioch scenario discovery on machines without the simulator installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndustrialCellConfig:
    """Configuration for the uniform eight-cell visual demo."""

    loop_period_s: float = 8.0

    def __post_init__(self) -> None:
        if self.loop_period_s <= 0.0:
            raise ValueError("loop_period_s must be positive")


@dataclass(frozen=True, slots=True)
class CellSnapshot:
    """JSON-safe state report for one cell at a point in simulated time."""

    asset_id: str
    box_number: int
    phase: str
    block_position_m: tuple[float, float, float]
    articulation_root: str
    service_dock_path: str
    tag_paths: tuple[str, ...]
    joint_positions_rad: tuple[float, ...] | None
