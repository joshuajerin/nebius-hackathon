"""Two adjacent guarded workbenches with tag-to-dock truth."""
from pathlib import Path
from factory_sre.contracts import WORKBENCHES
def _mat(stage,path,color):
    from pxr import Gf,Sdf,UsdShade
    m=UsdShade.Material.Define(stage,path);s=UsdShade.Shader.Define(stage,path+"/Shader");s.CreateIdAttr("UsdPreviewSurface");s.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),"surface");return m
def _box(stage,path,pos,size,color,collision=True):
    from pxr import Gf,UsdGeom,UsdPhysics,UsdShade
    c=UsdGeom.Cube.Define(stage,path);c.CreateSizeAttr(1.0);x=UsdGeom.Xformable(c);x.AddTranslateOp().Set(Gf.Vec3d(*pos));x.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(_mat(stage,path+"/Material",color))
    return c
def _tag(stage,path,center,tag_id,*,face_axis="y",face_sign=-1):
    from pxr import Gf,Sdf,UsdGeom,UsdShade
    if face_axis not in {"x", "y"} or face_sign not in {-1, 1}:
        raise ValueError("AprilTag face must use axis x/y and sign -1/+1")
    x,y,z=center;s=.120;m=UsdGeom.Mesh.Define(stage,path)
    if face_axis == "y":
        points=[Gf.Vec3f(x-s/2,y,z-s/2),Gf.Vec3f(x+s/2,y,z-s/2),Gf.Vec3f(x+s/2,y,z+s/2),Gf.Vec3f(x-s/2,y,z+s/2)]
    else:
        points=[Gf.Vec3f(x,y+s/2,z-s/2),Gf.Vec3f(x,y-s/2,z-s/2),Gf.Vec3f(x,y-s/2,z+s/2),Gf.Vec3f(x,y+s/2,z+s/2)]
    m.CreatePointsAttr(points);m.CreateFaceVertexCountsAttr([4]);m.CreateFaceVertexIndicesAttr([0,1,2,3] if face_sign < 0 else [0,3,2,1]);m.CreateDoubleSidedAttr(True)
    st=UsdGeom.PrimvarsAPI(m).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex)
    # The same vertex ordering is viewed from opposite sides of a box.  Flip
    # U on positive-axis faces so their outward view is a canonical AprilTag,
    # not a mirrored code that a real detector must reject.
    uvs = (
        [Gf.Vec2f(0,0),Gf.Vec2f(1,0),Gf.Vec2f(1,1),Gf.Vec2f(0,1)]
        if face_sign < 0
        else [Gf.Vec2f(1,0),Gf.Vec2f(0,0),Gf.Vec2f(0,1),Gf.Vec2f(1,1)]
    )
    st.Set(uvs)
    mat=UsdShade.Material.Define(stage,path+"/Material");sh=UsdShade.Shader.Define(stage,path+"/Material/Preview");sh.CreateIdAttr("UsdPreviewSurface");tex=UsdShade.Shader.Define(stage,path+"/Material/Texture");tex.CreateIdAttr("UsdUVTexture");tex.CreateInput("file",Sdf.ValueTypeNames.Asset).Set(str(Path(f"/workspace/project/assets/tags/tag36h11_{tag_id}.png")));rd=UsdShade.Shader.Define(stage,path+"/Material/Reader");rd.CreateIdAttr("UsdPrimvarReader_float2");rd.CreateInput("varname",Sdf.ValueTypeNames.Token).Set("st");tex.CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(),"result");sh.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),"rgb");mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),"surface");UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
def build_factory_scene():
    import antioch
    from isaacsim.core.experimental.objects import GroundPlane
    from pxr import Usd,UsdGeom,UsdLux
    stage=antioch.stage();UsdGeom.SetStageMetersPerUnit(stage,1.0);stage.SetStartTimeCode(0);stage.SetEndTimeCode(240);GroundPlane("/World/Ground",templates=None);_box(stage,"/World/Aisle/Pallet",(2.6,-1.5,.25),(.65,.65,.5),(.42,.26,.1))
    out={}
    for manifest,x,suffix in zip(WORKBENCHES,(1.5,3.7),("A","B"),strict=True):
        root=f"/World/Workbench{suffix}";_box(stage,root+"/Top",(x,1.1,.7),(1.8,.9,.1),(.22,.28,.34));_box(stage,root+"/BackGuard",(x,1.52,1.25),(1.8,.05,1.2),(.85,.72,.1));_box(stage,root+"/MachineBase",(x,1.15,.92),(.25,.25,.35),(.7,.12,.1));machine_arm=_box(stage,root+"/MachineArm",(x,1.15,1.2),(.75,.1,.1),(.9,.32,.08),False)
        motion=UsdGeom.Xformable(machine_arm).AddRotateZOp(opSuffix="deterministic_motion")
        phase=1.0 if suffix=="A" else -1.0
        for time,value in ((0,-25.0*phase),(60,25.0*phase),(120,-25.0*phase),(180,25.0*phase),(240,-25.0*phase)):motion.Set(value,Usd.TimeCode(time))
        for gx,n in ((x-.88,"L"),(x+.88,"R")):_box(stage,root+f"/Guard{n}",(gx,1.1,1.05),(.05,.85,.8),(.85,.72,.1))
        tag=(x+.30,.64,.88);dock=(tag[0]-.30,tag[1],tag[2]);_tag(stage,root+f"/Tag{manifest.tag_id}",tag,manifest.tag_id);_box(stage,root+"/MagneticDock",dock,(.1,.035,.05),(.06,.09,.12));out[manifest.asset_id]={"tag":tag,"dock":dock,"tag_id":manifest.tag_id,"motion":"deterministic +/-25deg behind guard"}
    UsdLux.DomeLight.Define(stage,"/World/DomeLight").CreateIntensityAttr(500);UsdLux.DistantLight.Define(stage,"/World/KeyLight").CreateIntensityAttr(1800)
    return {"transforms":out,"moving_machinery_guarded":True}
