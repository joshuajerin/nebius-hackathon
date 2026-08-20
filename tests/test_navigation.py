from __future__ import annotations

import pytest

from factory_sre.sim.navigation import GridMap, astar, hero_factory_grid


@pytest.mark.parametrize("goal", [(1.5, -0.9), (3.7, -0.9)])
def test_factory_path_reaches_both_staging_areas_without_occupied_cells(goal) -> None:
    grid = hero_factory_grid()
    path = astar(grid, (-0.5, -2.5), goal)
    assert path[0] == pytest.approx((-0.5, -2.5))
    assert path[-1] == pytest.approx(goal)
    assert all(grid.free(grid.world_to_cell(*point)) for point in path)


def test_astar_does_not_cut_diagonally_between_obstacles() -> None:
    grid = GridMap(3, 3, 1.0, (0.0, 0.0), frozenset({(1, 0), (0, 1)}))
    with pytest.raises(RuntimeError, match="no collision-free path"):
        astar(grid, (0.0, 0.0), (2.0, 2.0))
