import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetLSTMCfg, DCMotorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from pathlib import Path

HERE = Path(__file__).parent


SPOT_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path= f"{HERE}/spot_arm.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*_hy": 0.872665,
            ".*_kn": -1.79769,
            "[fh]l_hx":0.122173,
            "[fh]r_hx":-0.122173,
            "arm.*":0.0,
        },
        joint_vel={
            ".*":0.0,
        }
    ),
    actuators = {
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=["arm.*"],
            effort_limit=45.0,
            # kp=100.0,
            # kd=1.0,
            # max_velocity=100.0,
            stiffness=60.0,
            damping=1.5,
            min_delay=0,
            max_delay=4
        ),
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["[fh].*"],
            effort_limit=45.0,
            # kp=100.0,
            # kd=1.0,
            # max_velocity=100.0,
            stiffness=60.0,
            damping=1.5,
            min_delay=0,
            max_delay=4
        )            
    }
)
    