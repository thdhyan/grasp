import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg


OFFICE_CHAIR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/OfficeChair",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/thakk100/Projects/ebasa/grasp_heir/source/grasp_heir/grasp_heir/assets/objects/office_chair.usda",
        scale=(0.7, 0.7, 0.7),
    ),
    init_state= ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={},
)
