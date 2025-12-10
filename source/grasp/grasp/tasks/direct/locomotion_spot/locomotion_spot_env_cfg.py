# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.envs.mdp as mdp

def no_op_spawn(prim_path, cfg, *args, **kwargs):
    return prim_path

@configclass
class NoOpSpawnCfg(sim_utils.PinholeCameraCfg):
    func = no_op_spawn

@configclass
class EventCfg:
    """Configuration for events."""
    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class LocomotionSpotEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 20.0
    action_space = 19
    observation_space = 51 # Base (13) + Joint pos (19) + vel (19)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    
    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/thakk100/Projects/csci8551/grasp/assets/spot_arm_camera.usd",
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
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.7),
            joint_pos={
                ".*_hx": 0.0,
                ".*_hy": 0.8,
                ".*_kn": -1.5,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hx", ".*_hy", ".*_kn"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=0.0,
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

    # chair
    chair: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Chair",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/thakk100/Projects/csci8551/grasp/assets/office chair.usda",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.0),
        ),
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=0.0,
                velocity_limit_sim=0.0,
                stiffness=0.0,
                damping=0.1,
            ),
        },
    )

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/spot_arm_01/body/frontright_fisheye", 
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=NoOpSpawnCfg(),
        data_types=["rgb"],
        width=128,
        height=128,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_dist_to_chair = -1.0
    rew_scale_base_height = 2.0
    rew_scale_facing_chair = 1.0
    rew_scale_leg_movement = -0.5
    
    # thresholds
    min_base_height = 0.35
    leg_movement_threshold = 0.01
    leg_stall_steps = 20
    
    # action scale
    action_scale = 0.5
