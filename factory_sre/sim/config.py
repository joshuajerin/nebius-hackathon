"""Frozen hero-release simulation configuration."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimConfig:
    physics_dt: float = 1.0 / 120.0
    render_dt: float = 1.0 / 30.0
    max_axial_force_n: float = 8.0
    max_roll_pitch_deg: float = 15.0
    tag_translation_tolerance_m: float = 0.005
    tag_rotation_tolerance_deg: float = 3.0
    minimum_insert_depth_m: float = 0.005
    connector_clearance_m: float = 0.0005
    contact_offset_m: float = 0.002
    solver_position_iterations: int = 64
    solver_velocity_iterations: int = 16


HERO_CONFIG = SimConfig()

# The stock description USD is visual-only in Isaac Sim 6.0.1 (43 prims,
# zero joints). The supplied policy wrapper composes that source into the
# 12-joint articulation expected by Go2FlatTerrainPolicy.
GO2_USD = "/Isaac/Samples/Policies/go2/go2.usda"
GO2_DESCRIPTION_USD = "/Isaac/Robots/Unitree/Go2/go2.usd"
GO2_POLICY = "/Isaac/Samples/Policies/go2/physx_policy.pt"
GO2_POLICY_ENV = "/Isaac/Samples/Policies/go2/physx_env.yaml"
SO101_ASSET = "so101_antioch"
SO101_VERSION = "1.3.2"
WAREHOUSE_USD = "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
UR5E_USD = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
D455_USD = "/Isaac/Sensors/Realsense/D455/rsd455.usd"
P61X_USD = "/Isaac/Sensors/SICK/InspectorP61x/SICK_InspectorP61x.usd"
NANOSCAN3_USD = "/Isaac/Sensors/SICK/nanoScan3/SICK_nanoScan3.usd"
INSPECTOR83X_USD = "/Isaac/Sensors/SICK/Inspector83x/SICK_Inspector83x.usd"
