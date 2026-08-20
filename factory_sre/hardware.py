"""No-motor dry-run adapters implementing the hardware contracts."""
from dataclasses import dataclass,field
from .contracts import DockObservation,Pose2D,Transform3D
@dataclass(slots=True)
class DryRunBaseAdapter:
    navigation_ok:bool=True; plant_ok:bool=True; log:list[str]=field(default_factory=list)
    def navigate(self,pose:Pose2D)->bool:self.log.append(f"navigate:{pose}");return self.navigation_ok
    def plant(self)->bool:self.log.append("plant");return self.plant_ok
    def stable(self)->bool:return self.plant_ok
    def emergency_stop(self)->None:self.log.append("emergency_stop")
@dataclass(slots=True)
class DryRunArmAdapter:
    asset_id:str="workbench-b"; measured_force_n:float=2.0; depth_m:float=0.006; contact:bool=True; log:list[str]=field(default_factory=list)
    def move_to_inspection(self)->bool:self.log.append("inspect");return True
    def align(self,target:Transform3D)->bool:self.log.append(f"align:{target}");return True
    def approach(self,target:Transform3D)->bool:self.log.append(f"approach:{target}");return True
    def guarded_dock(self,target:Transform3D)->DockObservation:
        self.log.append(f"dock:{target}");return DockObservation(self.asset_id,target,(0.0,)*36,self.contact,self.depth_m,self.measured_force_n,0)
    def retreat(self)->bool:self.log.append("retreat");return True
    def emergency_stop(self)->None:self.log.append("emergency_stop")
@dataclass(slots=True)
class DryRunCameraAdapter:
    detections_by_size:dict[float,tuple[tuple[int,Transform3D],...]]=field(default_factory=dict)
    def detect_tags(self,family:str,size_m:float):return self.detections_by_size.get(size_m,()) if family=="tag36h11" else ()
    def capture_rgb(self)->object:return b"dry-run-frame"
DryRunLocomotionAdapter=DryRunBaseAdapter
