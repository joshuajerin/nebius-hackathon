"""Antioch evidence scenarios."""
from __future__ import annotations
import os
import antioch
from factory_sre.sim.config import HERO_CONFIG

PROFILE=antioch.BootProfile(physics_dt=HERO_CONFIG.physics_dt,render_dt=HERO_CONFIG.render_dt,physics_engine="physx",render_quality="performance",viewport=(1280,720))
LOCOMOTION_PROFILE=antioch.BootProfile(physics_dt=1.0/200.0,render_dt=1.0/50.0,physics_engine="physx",render_quality="performance",viewport=(1280,720))

@antioch.scenario(name="go2_forward_stop_smoke",description="Drive the supplied Go2 flat-terrain policy forward at 1 m/s and prove it decelerates on a zero command.",tags=("smoke","go2","locomotion"),sim=LOCOMOTION_PROFILE,capture=True)
def go2_forward_stop_smoke(run:antioch.ScenarioRun)->None:
    import numpy as np
    import torch
    from isaacsim.core.experimental.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy
    from isaacsim.storage.native import get_assets_root_path
    import antioch
    # This sequence follows the Isaac Sim 6.0.1 Go2 policy test: create a
    # ground environment and policy, reset/play, initialize, then forward
    # commands at the physics timestep.
    SimulationManager.set_physics_sim_device("cuda")
    root=get_assets_root_path()
    add_reference_to_stage(usd_path=root+"/Isaac/Environments/Grid/default_environment.usd",path="/World/GroundEnvironment")
    go2=Go2FlatTerrainPolicy(prim_path="/World/Go2",position=[-1.5,-1.2,0.50],policy_path=root+"/Isaac/Samples/Policies/go2/physx_policy.pt",env_config_path=root+"/Isaac/Samples/Policies/go2/physx_env.yaml")
    world=antioch.world();world.reset();go2.initialize();go2.post_reset()
    set_camera_view(eye=[-4.8,-5.5,3.1],target=[-0.2,-1.2,0.35],camera_prim_path="/OmniverseKit_Persp")
    command=torch.tensor([1.0,0.0,0.0],dtype=torch.float32,device="cuda")
    start=np.asarray(go2.robot.get_world_poses()[0].numpy()[0],dtype=float)
    for _ in range(320):
        go2.forward(1.0/200.0,command);world.step(render=True)
    moving=np.asarray(go2.robot.get_world_poses()[0].numpy()[0],dtype=float)
    velocity=np.asarray(go2.robot.get_velocities()[0].numpy()[0],dtype=float)
    for _ in range(160):
        go2.forward(1.0/200.0,torch.zeros(3,dtype=torch.float32,device="cuda"));world.step(render=True)
    stopped=np.asarray(go2.robot.get_world_poses()[0].numpy()[0],dtype=float)
    stopped_velocity=np.asarray(go2.robot.get_velocities()[0].numpy()[0],dtype=float)
    forward_distance=float(moving[0]-start[0]); residual_speed=float(np.linalg.norm(stopped_velocity[:3]))
    logger=antioch.Logger("factory_sre/go2")
    logger.scalar("forward_distance_m",forward_distance);logger.scalar("moving_speed_mps",float(np.linalg.norm(velocity[:3])));logger.scalar("stop_residual_speed_mps",residual_speed)
    frame=antioch.capture_viewport()
    run.check("Go2 moves forward under policy",forward_distance>=0.20,detail=f"{forward_distance:.3f} m in 1.60 s")
    run.check("Go2 remains above ground",float(stopped[2])>=0.20,detail=f"base z {float(stopped[2]):.3f} m")
    run.check("Go2 decelerates after zero command",residual_speed<=1.00,detail=f"residual {residual_speed:.3f} m/s")
    run.check("locomotion review frame captured",frame is not None,detail="viewport after forward-stop")
    if frame is not None: logger.image("camera/forward_stop",np.asarray(frame)[...,:3])
    run.add_results({"forward_distance_m":forward_distance,"residual_speed_mps":residual_speed,"start_position":start.tolist(),"moving_position":moving.tolist(),"stopped_position":stopped.tolist()})

@antioch.scenario(name="asset_compatibility_smoke",description="Load the exact 12-joint Go2 policy articulation and so101_antioch@1.3.2.",tags=("smoke","assets","go2","so101"),sim=PROFILE,capture=False)
def asset_compatibility_smoke(run:antioch.ScenarioRun)->None:
    import inspect
    from factory_sre.sim.assets import load_exact_assets,make_go2_policy
    report=load_exact_assets()
    run.check("stock Go2 policy articulation composes",bool(report["go2_valid"]),detail=str(report["go2_usd"]))
    run.check("pinned SO-101 asset composes",bool(report["so101_valid"]),detail=str(report["so101_asset"]))
    run.check("Go2 exposes 12 locomotion joints",len(report["go2_joints"])==12,detail=str(report["go2_joints"]))
    run.check("SO-101 exposes 7 articulated joints",len(report["so101_joints"])==7,detail=str(report["so101_joints"]))
    policy=make_go2_policy(str(report["policy_path"]),str(report["policy_env_path"])); signature=str(inspect.signature(policy.forward))
    run.check("Go2 flat-terrain policy constructs",policy is not None,detail=f"forward{signature}")
    run.add_result("asset_report",report|{"policy_forward_signature":signature})

@antioch.scenario(name="two_workbench_scene_smoke",description="Compose two adjacent guarded workbenches and verify exact AprilTag-to-dock geometry.",tags=("smoke","scene","apriltag","dock"),sim=PROFILE,capture=True)
def two_workbench_scene_smoke(run:antioch.ScenarioRun)->None:
    import numpy as np
    from isaacsim.core.utils.viewports import set_camera_view
    from factory_sre.contracts import WORKBENCHES
    from factory_sre.sim.scene import build_factory_scene
    report=build_factory_scene(); world=antioch.world(); world.reset()
    set_camera_view(eye=[2.6,-5.2,3.4],target=[2.6,0.9,0.8],camera_prim_path="/OmniverseKit_Persp")
    for _ in range(8):world.step(render=True)
    frame=antioch.capture_viewport(); run.check("factory review frame captured",frame is not None,detail="1280x720")
    if frame is not None:
        rgb=np.asarray(frame)[...,:3]; run.check("factory frame has contrast",float(rgb.std())>=5,detail=f"std {float(rgb.std()):.2f}"); antioch.Logger("factory_sre").image("camera/overview",rgb)
    for manifest in WORKBENCHES:
        points=report["transforms"][manifest.asset_id]; offset=points["dock"][0]-points["tag"][0]
        run.check(f"{manifest.asset_id} tag identity",points["tag_id"]==manifest.tag_id,detail=f"tag {points['tag_id']}")
        run.check(f"{manifest.asset_id} dock offset",abs(offset+0.30)<1e-9,detail=f"{offset:.6f} m")
    run.check("moving machinery remains guarded",report["moving_machinery_guarded"],detail="outside approach volume")
    run.add_result("scene",report)

CASES=(antioch.case({"asset_id":"workbench-a"},id="target-a",tags=("hero","certification")),antioch.case({"asset_id":"workbench-b"},id="target-b",tags=("hero","certification")))
@antioch.scenario(name="hero_diagnostic_recovery",description="Exercise the dock-gated target and the single whitelisted restart.",tags=("hero","diagnostics","recovery"),cases=CASES,sim=None,capture=False,restart=["diagnostic-target"])
def hero_diagnostic_recovery(run:antioch.ScenarioRun,asset_id:str="workbench-b")->None:
    from factory_sre.contracts import FaultCode,Incident,Transform3D,manifest_for_asset
    from factory_sre.hardware import DryRunArmAdapter,DryRunBaseAdapter,DryRunCameraAdapter
    from factory_sre.mission import FLOW,MissionController
    from factory_sre.transport import HttpDiagnosticTransport
    manifest=manifest_for_asset(asset_id); transport=HttpDiagnosticTransport(os.environ.get("FACTORY_SRE_CONTROL_URL",manifest.diagnostic_target.control_endpoint),manifest.diagnostic_target.diagnostic_endpoint)
    transport.inject_fault(asset_id,FaultCode.VISION_SERVICE_CRASHED)
    result=MissionController().run(Incident(f"hero-{asset_id}",asset_id,FaultCode.VISION_SERVICE_CRASHED.value),manifest,DryRunBaseAdapter(),DryRunArmAdapter(asset_id=asset_id),DryRunCameraAdapter({0.120:((manifest.tag_id,Transform3D((2,0.8,0.9))),)}),transport)
    states=[e.state.value for e in result.events]
    run.check("mission recovered",result.success,detail=f"{result.terminal_state}/{result.terminal_reason}")
    run.check("one guarded dock attempt",result.dock_attempts==1,detail=str(result.dock_attempts))
    run.check("every required state recorded",all(s.value in states for s in FLOW),detail=str(states))
    run.add_result("mission",{"asset_id":asset_id,"terminal_reason":result.terminal_reason.value,"states":states})
