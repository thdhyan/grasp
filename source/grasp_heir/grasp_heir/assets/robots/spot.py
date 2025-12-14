import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import os

USD_PATH = os.path.join(os.path.dirname(__file__), "spot_arm_01_colliders.usda")

SPOT_CFG : ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
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
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.61),  # Spawn above ground (Spot body height ~0.5m)
            joint_pos={
                #".*_hx": 0.0,
                ".*_hy": 0.9,
                ".*_kn": -1.502,
                "arm0_sh1": -1.309,
                "arm0_el0": 1.32,
                "arm0_el1": 0.0,
                "arm0_wr0": 1.26,
                "arm0_wr1": 0.0,
                "arm0_f1x": -0.855,
                #"arm0_f2x": -50.4,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hx", ".*_hy", ".*_kn"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm0_sh.*", "arm0_el.*"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["arm0_wr.*"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["arm0_f1x"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=100.0,
                damping=10.0,
            ),
        },
    )

