from pxr import Usd, UsdGeom

stage = Usd.Stage.Open("/home/thakk100/Projects/csci8551/grasp/assets/spot_arm_camera.usd")
for prim in stage.Traverse():
    print(prim.GetPath())
