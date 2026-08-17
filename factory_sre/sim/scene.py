"""Small deterministic factory fixture used by the compatibility gate."""

from __future__ import annotations

from typing import Any

from factory_sre.contracts import WORKBENCHES


def add_smoke_fixture(world: Any) -> dict[str, list[str]]:
    """Create two service panels, dock targets, and tagged identity plates."""

    import numpy as np
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.prims import get_prim_at_path

    panels: list[str] = []
    docks: list[str] = []
    tags: list[str] = []
    for index, manifest in enumerate(WORKBENCHES):
        y = -1.2 if index == 0 else 1.2
        prim_name = manifest.asset_id.replace("-", "_")
        panel_path = f"/World/Factory/{prim_name}/panel"
        dock_path = f"/World/Factory/{prim_name}/dock"
        tag_path = f"/World/Factory/{prim_name}/tag_{manifest.tag_id}"

        world.scene.add(
            FixedCuboid(
                prim_path=panel_path,
                name=f"panel_{index}",
                position=np.array([2.5, y, 0.8]),
                scale=np.array([0.12, 1.2, 1.4]),
                color=np.array([0.18, 0.22, 0.28]),
            )
        )
        world.scene.add(
            FixedCuboid(
                prim_path=dock_path,
                name=f"dock_{index}",
                position=np.array([2.40, y - 0.30, 0.85]),
                scale=np.array([0.04, 0.16, 0.12]),
                color=np.array([0.10, 0.65, 0.95]),
            )
        )
        world.scene.add(
            FixedCuboid(
                prim_path=tag_path,
                name=f"tag_{manifest.tag_id}",
                position=np.array([2.39, y, 1.10]),
                scale=np.array([0.02, 0.12, 0.12]),
                color=np.array([0.95, 0.95, 0.95]),
            )
        )
        tag_prim = get_prim_at_path(tag_path)
        tag_prim.SetCustomDataByKey("tag_family", "tag36h11")
        tag_prim.SetCustomDataByKey("tag_id", manifest.tag_id)
        panels.append(panel_path)
        docks.append(dock_path)
        tags.append(tag_path)

    return {"panels": panels, "docks": docks, "tags": tags}
