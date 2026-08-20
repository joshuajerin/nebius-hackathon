"""Deterministic occupancy-grid A* used for coarse workbench routing."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GridMap:
    width: int
    height: int
    resolution_m: float
    origin_xy: tuple[float, float]
    occupied: frozenset[tuple[int, int]]

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (round((x - self.origin_xy[0]) / self.resolution_m), round((y - self.origin_xy[1]) / self.resolution_m))

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (self.origin_xy[0] + cell[0] * self.resolution_m, self.origin_xy[1] + cell[1] * self.resolution_m)

    def free(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height and cell not in self.occupied


def inflate(occupied: set[tuple[int, int]], radius_cells: int) -> frozenset[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for x, y in occupied:
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    result.add((x + dx, y + dy))
    return frozenset(result)


def astar(grid: GridMap, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]]:
    start, goal = grid.world_to_cell(*start_xy), grid.world_to_cell(*goal_xy)
    if not grid.free(start) or not grid.free(goal):
        raise ValueError("start and goal must be in free space")
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost: dict[tuple[int, int], float] = {start: 0.0}
    moves = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx, dy in moves:
            nxt = current[0] + dx, current[1] + dy
            if not grid.free(nxt):
                continue
            if dx and dy and (
                not grid.free((current[0] + dx, current[1]))
                or not grid.free((current[0], current[1] + dy))
            ):
                continue
            step = math.sqrt(2.0) if dx and dy else 1.0
            new_cost = cost[current] + step
            if new_cost >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = new_cost
            priority = new_cost + math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
            heapq.heappush(frontier, (priority, nxt))
            came_from[nxt] = current
    if goal not in came_from:
        raise RuntimeError("no collision-free path")
    cells: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        cells.append(cursor)
        cursor = came_from[cursor]
    return [grid.cell_to_world(cell) for cell in reversed(cells)]


def hero_factory_grid() -> GridMap:
    resolution = 0.10
    width, height = 70, 50
    raw: set[tuple[int, int]] = set()
    # One guarded vision workcell occupies the north side of the aisle.
    for x in range(20, 40):
        for y in range(30, 46):
            raw.add((x, y))
    # One pallet forces a non-trivial aisle route.
    for x in range(35, 43):
        for y in range(12, 21):
            raw.add((x, y))
    return GridMap(width, height, resolution, (-1.0, -3.0), inflate(raw, radius_cells=4))
