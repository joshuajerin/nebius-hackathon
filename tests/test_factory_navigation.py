from __future__ import annotations

import pytest

from factory_sre.contracts import WORKBENCH_A, WORKBENCH_B
from factory_sre.sim.factory_navigation import plan_to_manifest, validate_route


@pytest.mark.parametrize("manifest", (WORKBENCH_A, WORKBENCH_B))
def test_manifest_selected_route_reaches_only_its_approach_pose(manifest) -> None:
    route = plan_to_manifest(manifest)
    assert route.asset_id == manifest.asset_id
    assert route.goal == manifest.approach_pose
    assert route.final_error_m < 1e-9
    assert route.length_m > 0.0
    assert validate_route(route) == ()


def test_neighbor_workbench_has_a_distinct_terminal_pose() -> None:
    route_a = plan_to_manifest(WORKBENCH_A)
    route_b = plan_to_manifest(WORKBENCH_B)
    assert route_a.waypoints[-1] != route_b.waypoints[-1]
    assert route_a.asset_id != route_b.asset_id
