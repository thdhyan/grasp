# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg

from . import mdp
from grasp_heir.tasks.manager_based.grasp_heir import mdp as heir_mdp

import torch
##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from grasp_heir.assets.robots.spot import SPOT_CFG

from grasp_heir.assets.objects.articulations import OFFICE_CHAIR_CFG   

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
##
# Scene definition
##


@configclass
class GraspHeirSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #objects
    chair: ArticulationCfg = OFFICE_CHAIR_CFG.replace(prim_path="{ENV_REGEX_NS}/Chair", 
                                                                init_state=ArticulationCfg.InitialStateCfg(
                                                                    pos=(2.0, 0.0, 0.1),
                                                                    rot=(1.0, 0.0, 0.0, 0.0)
                                                                )
                                                            ) 
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/.*", history_length=3, track_air_time=True)

    # foot contact sensors
    # foot_contacts: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*_lleg/.*",
    #     update_period=0.0,
    #     history_length=3,
    #     track_air_time=False,
    #     filter_prim_paths_expr=[],
    #       # Detect contact with anything (ground, obstacles)
    # )
    # body_contacts: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/.*",  # ✓ Match body under base
    #     update_period=0.0,
    #     history_length=3,
    #     track_air_time=False,
    #     filter_prim_paths_expr=[],  # Detect contact with anything (ground, obstacles)
    # )

    # gripper_contacts: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*",
    #     update_period=0.0,
    #     history_length=3,
    #     track_air_time=False,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Chair/.*"],  # Detect contact with Chair
    # )
##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=100.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=[".*"])}
        )

        # foot_contact_forces  = ObsTerm(
        #     func=mdp.contact_forces,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("foot_contacts"),
        #         "threshold": 10.0,
        #     },
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    

@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot base position and orientation to initial state
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.6, 0.7)},
            "velocity_range": {},
        },
    )
    
    # Reset robot joint positions to default configuration
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.9, 1.1),  # Small variation around default
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # Reset chair position and orientation
    reset_chair_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("chair"),
            "pose_range": {"x": (0.0,0.0), "y": (0.0, 0.0), "z": (0.0, 0.001), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (math.pi/2, math.pi/2)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    
    # Reset chair joint positions and velocities to zero (stationary)
    reset_chair_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("chair"),
            "position_range": (1.0, 1.0),  # Keep at default position
            "velocity_range": (0.0, 0.0),  # Zero velocity
        },
    )

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )
    # pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # (3) Penalize too much change in joint velocities (smoothness)
    joint_accel = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.7,
        },
    )
    # (4) Reward when chair is grasped
    # chair_grasped = RewTerm(
    #     func=heir_mdp.chair_grasped_reward,
    #     weight=5.0,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "chair_cfg": SceneEntityCfg("chair"),
    #         "contact_threshold": 0.5,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_jumped = DoneTerm(
        func=mdp.base_height_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.2,
            "max_height": 2.0,
            "grace_period_s": 1.0,
        },
    )
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )
    # body_contact =DoneTerm(
    #     func=mdp.undesired_contacts,
    #     params={
    #         "sensor_cfg" : SceneEntityCfg("body_contacts"),
    #         "threshold" : 5.0,

    #     },
    # )
    # Excessive Tilt
    excessive_tilt = DoneTerm(
        func=heir_mdp.excessive_tilt,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_tilt_rad": 1.0472,  # 60 degrees in radians (pi/3)
            "grace_period_s": 1.0,  # Disable for first 1 second after reset
        },
    )
    # If chair goes out of a 10m diameter circle
    chair_out_of_bounds = DoneTerm(
        func=heir_mdp.pos_out_of_radius,
        params={
            "asset_cfg": SceneEntityCfg("chair"),
            "radius": 5.0,
        },
    ) 

##
# Environment configuration
##


@configclass
class GraspHeirEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: GraspHeirSceneCfg = GraspHeirSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation