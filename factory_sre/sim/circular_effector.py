"""Reusable circular magnetic service tool for the SO-101 terminal link."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ASSET_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "custom"
    / "so101_circular_service_effector.usda"
)


@dataclass(frozen=True, slots=True)
class CircularEffectorSpecification:
    asset_path: Path = ASSET_PATH
    radius_m: float = 0.055
    contact_radius_m: float = 0.052
    depth_m: float = 0.080
    contact_axis: str = "X"
    chassis_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if not self.asset_path.is_file():
            raise FileNotFoundError(f"circular effector USD is missing: {self.asset_path}")
        if self.radius_m <= 0.0 or self.contact_radius_m <= 0.0 or self.depth_m <= 0.0:
            raise ValueError("circular effector dimensions must be positive")
        if self.contact_radius_m > self.radius_m:
            raise ValueError("contact face cannot exceed the tool-body radius")
        if self.contact_axis not in {"X", "Y", "Z"}:
            raise ValueError("contact axis must be X, Y, or Z")


DEFAULT_CIRCULAR_EFFECTOR = CircularEffectorSpecification()


@dataclass(frozen=True, slots=True)
class CircularEffectorReport:
    asset_path: str
    prim_path: str
    terminal_link: str
    contact_frame: str
    contact_face: str
    radius_m: float
    contact_radius_m: float
    contact_axis: str
    hidden_source_geometry: tuple[str, ...]
    disabled_source_colliders: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _source_gripper_prims(stage, terminal_link) -> list:
    """Find the pre-existing finger/jaw subtree, never the authored tool."""
    from pxr import Usd

    candidates = []
    root = stage.GetPrimAtPath("/World/SO101")
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        lowered = path.lower()
        if prim.IsInPrototype():
            continue
        if path.startswith(str(terminal_link.GetPath()) + "/") or any(
            token in lowered for token in ("/jaw", "/gripper", "/finger")
        ):
            candidates.append(prim)
    return candidates


def author_circular_service_effector(
    stage,
    terminal_link,
    chassis,
    *,
    spec: CircularEffectorSpecification = DEFAULT_CIRCULAR_EFFECTOR,
    hide_source_gripper: bool = True,
) -> CircularEffectorReport:
    """Replace the visible gripper with a chassis-forward circular service puck.

    The referenced tool is collision geometry owned by ``terminal_link``. It
    deliberately carries no rigid-body API of its own, so it cannot float or
    form a second articulation.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    spec.validate()
    hidden_geometry: list[str] = []
    disabled_colliders: list[str] = []
    if hide_source_gripper:
        for prim in _source_gripper_prims(stage, terminal_link):
            if prim.IsA(UsdGeom.Boundable):
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeInvisible()
                hidden_geometry.append(str(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision = UsdPhysics.CollisionAPI(prim)
                attribute = collision.GetCollisionEnabledAttr() or collision.CreateCollisionEnabledAttr()
                attribute.Set(False)
                disabled_colliders.append(str(prim.GetPath()))

    tool_path = str(terminal_link.GetPath()) + "/FactorySRECircularServiceEffector"
    tool_prim = stage.DefinePrim(tool_path, "Xform")
    tool_prim.GetReferences().AddReference(str(spec.asset_path))

    # Keep the puck normal aligned with the Go2's forward (+X) axis regardless
    # of the parked SO-101 wrist rotation. The mount plane remains at the
    # terminal-link origin and the referenced asset extends forward from it.
    cache = UsdGeom.XformCache()
    chassis_world = cache.GetLocalToWorldTransform(chassis)
    terminal_world = cache.GetLocalToWorldTransform(terminal_link)
    terminal_in_chassis = terminal_world * chassis_world.GetInverse()
    terminal_position = terminal_in_chassis.ExtractTranslation()
    tool_in_chassis = Gf.Matrix4d(1.0)
    tool_in_chassis.SetTranslateOnly(
        terminal_position + Gf.Vec3d(*spec.chassis_offset_m)
    )
    tool_world = tool_in_chassis * chassis_world
    tool_local = tool_world * terminal_world.GetInverse()
    UsdGeom.Xformable(tool_prim).MakeMatrixXform().Set(tool_local)

    asset_contact_frame = tool_path + "/ContactFrame"
    contact_face = tool_path + "/ContactFace"
    if not stage.GetPrimAtPath(asset_contact_frame).IsValid() or not stage.GetPrimAtPath(contact_face).IsValid():
        raise RuntimeError("circular effector reference did not compose its contact geometry")

    # Robot Poser's kinematic-chain resolver maps a SiteAPI to its direct
    # parent link.  The reference asset's ContactFrame is nested one level
    # below this terminal rigid body, so mirror that exact frame as a direct
    # child of the terminal link.  This keeps the reusable puck asset intact
    # while making its physical face a valid Isaac robot-schema endpoint.
    asset_contact_world = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(asset_contact_frame)
    )
    contact_frame = str(terminal_link.GetPath()) + "/FactorySREMagneticContactSite"
    contact_site = UsdGeom.Xform.Define(stage, contact_frame)
    contact_site_local = asset_contact_world * terminal_world.GetInverse()
    UsdGeom.Xformable(contact_site).MakeMatrixXform().Set(contact_site_local)

    return CircularEffectorReport(
        asset_path=str(spec.asset_path),
        prim_path=tool_path,
        terminal_link=str(terminal_link.GetPath()),
        contact_frame=contact_frame,
        contact_face=contact_face,
        radius_m=float(spec.radius_m),
        contact_radius_m=float(spec.contact_radius_m),
        contact_axis=spec.contact_axis,
        hidden_source_geometry=tuple(hidden_geometry),
        disabled_source_colliders=tuple(disabled_colliders),
    )
