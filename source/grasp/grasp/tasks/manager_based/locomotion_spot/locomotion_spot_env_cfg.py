# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based locomotion spot environment configuration.

This environment trains a Spot robot with arm to:
1. Navigate to a chair
2. Grasp the chair
3. Move the chair to a randomly generated goal position

Features:
- Manager-based workflow with loggable reward terms
- Random goal position generation per environment
- Contact sensors on gripper fingers
- Camera observations with CNN policy support
- Debug visualization arrows for velocities and forces
- Increased environment spacing
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from .mdp import commands as custom_commands
from .mdp import observations as custom_observations
from .mdp import terminations as custom_terminations


# =============================================================================
# Spawn configurations
# =============================================================================

# No spawner needed for cameras that exist in USD - set spawn=None


# =============================================================================
# Scene configuration
# =============================================================================

@configclass
class LocomotionSpotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the locomotion spot environment."""
    
    # Environment settings - increased spacing
    num_envs: int = 64
    env_spacing: float = 8.0  # Increased from 4.0
    replicate_physics: bool = True
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    
    # Robot configuration
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/thakk100/Projects/grasp/assets/spot_arm_camera.usd",
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
    
    # Chair configuration
    chair: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Chair",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/thakk100/Projects/grasp/assets/office chair.usda",
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
    
    # Contact sensor for gripper finger
    gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/arm0_link_fngr",
        update_period=0.0,  # Update every physics step
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Chair"],  # Only detect chair contact
    )
    
    # Contact sensor for wrist (detects arm approaching chair)
    wrist_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/arm0_link_wr1",
        update_period=0.0,  # Update every physics step
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Chair"],  # Only detect chair contact
    )
    
    # Contact sensor for feet (locomotion)
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/.*_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=[],
    )
    
    # Contact sensor for body (fall detection)
    # Detects if main body/torso touches ground (indicates fallen)
    body_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/body",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=[],  # Detect contact with anything (ground, obstacles)
    )
    
    # Tiled camera for RGB, depth, and instance segmentation observations (front camera)
    # Camera already exists in USD, so spawn=None
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/body/frontright_fisheye",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
        data_types=["rgb", "depth", "instance_id_segmentation_fast"],
        width=128,
        height=128,
        colorize_instance_id_segmentation=False,  # Return as int32 IDs, not colorized
    )
    
    # Wrist camera for depth and instance segmentation (close-up view for grasping)
    # Camera already exists in USD, so spawn=None
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/arm0_link_wr1/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
        data_types=["rgb", "depth", "instance_id_segmentation_fast"],
        width=128,
        height=128,
        colorize_instance_id_segmentation=False,  # Return as int32 IDs, not colorized
    )
    
    # Dome light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


# =============================================================================
# Actions configuration
# =============================================================================

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # Joint position actions for all joints
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.2,
        use_default_offset=True,
    )


# =============================================================================
# Observations configuration
# =============================================================================

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    
    Multiple observation groups for different policies:
    - policy: Main policy observations (locomotion + manipulation)
    - locomotion: Locomotion-specific observations for leg control
    - manipulation: Arm-specific observations for grasping
    - critic: Full state observations for value function
    - camera: Vision observations (depth, segmentation)
    """
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Main policy observations (proprioceptive + task + spatial)."""
        
        # Robot proprioception
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        
        # Velocity commands
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        
        # Goal position (relative to robot)
        goal_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "goal_position"},
        )
        
        # Contact observations - feet (using custom observation)
        feet_contact = ObsTerm(
            func=custom_observations.contact_binary_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
        )
        
        # Contact observations - gripper finger
        gripper_contact_status = ObsTerm(
            func=custom_observations.gripper_contact_state,
            params={"sensor_cfg": SceneEntityCfg("gripper_contact"), "threshold": 1.0},
        )
        
        # Contact observations - wrist
        wrist_contact_status = ObsTerm(
            func=custom_observations.gripper_contact_state,
            params={"sensor_cfg": SceneEntityCfg("wrist_contact"), "threshold": 1.0},
        )
        
        # === Spatial observations for navigation and manipulation ===
        
        # Chair position relative to robot body (for navigation)
        chair_pos_rel = ObsTerm(
            func=custom_observations.chair_position_relative,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
            },
        )
        
        # Goal position relative to robot body (for navigation)
        goal_pos_rel = ObsTerm(
            func=custom_observations.goal_position_relative,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        
        # Direction vector to chair from robot body (for navigation)
        direction_to_chair = ObsTerm(
            func=custom_observations.direction_to_chair,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
            },
        )
        
        # Direction vector to goal from robot (for transporting chair)
        direction_to_goal = ObsTerm(
            func=custom_observations.direction_to_goal,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        
        # Distance from robot to chair (scalar)
        dist_to_chair = ObsTerm(
            func=custom_observations.distance_robot_to_chair,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
            },
        )
        
        # Distance from chair to goal (scalar)
        dist_chair_to_goal = ObsTerm(
            func=custom_observations.distance_chair_to_goal,
            params={"chair_cfg": SceneEntityCfg("chair")},
        )
        
        # === Arm-specific observations ===
        
        # Direction to chair from end-effector (for grasping)
        direction_ee_to_chair = ObsTerm(
            func=custom_observations.direction_to_chair_from_ee,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
                "ee_body_name": "arm0_link_fngr",
            },
        )
        
        # Distance from end-effector to chair (for grasping)
        dist_ee_to_chair = ObsTerm(
            func=custom_observations.distance_ee_to_chair,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
                "ee_body_name": "arm0_link_fngr",
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class LocomotionCfg(ObsGroup):
        """Locomotion-specific observations for leg control policy."""
        
        # Robot base state
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # Leg joints only
        leg_joint_pos = ObsTerm(
            func=custom_observations.leg_joint_positions,
            params={"robot_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        leg_joint_vel = ObsTerm(
            func=custom_observations.leg_joint_velocities,
            params={"robot_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        
        # Velocity command for locomotion
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        
        # Foot contacts
        feet_contact = ObsTerm(
            func=custom_observations.contact_binary_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
        )
        
        # Target direction for navigation
        direction_to_chair = ObsTerm(
            func=custom_observations.direction_to_chair,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class ManipulationCfg(ObsGroup):
        """Arm-specific observations for manipulation/grasping policy."""
        
        # Arm joint state
        arm_joint_pos = ObsTerm(
            func=custom_observations.arm_joint_positions,
            params={"robot_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        arm_joint_vel = ObsTerm(
            func=custom_observations.arm_joint_velocities,
            params={"robot_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        
        # End-effector state
        ee_position = ObsTerm(
            func=custom_observations.end_effector_position,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["arm0_link_fngr"])},
        )
        ee_velocity = ObsTerm(
            func=custom_observations.end_effector_velocity,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["arm0_link_fngr"])},
        )
        
        # Gripper contact
        gripper_contact = ObsTerm(
            func=custom_observations.gripper_contact_state,
            params={"sensor_cfg": SceneEntityCfg("gripper_contact"), "threshold": 1.0},
        )
        gripper_force = ObsTerm(
            func=custom_observations.gripper_contact_force,
            params={"sensor_cfg": SceneEntityCfg("gripper_contact")},
        )
        
        # Target (chair) relative to end-effector
        direction_ee_to_chair = ObsTerm(
            func=custom_observations.direction_to_chair_from_ee,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
                "ee_body_name": "arm0_link_fngr",
            },
        )
        dist_ee_to_chair = ObsTerm(
            func=custom_observations.distance_ee_to_chair,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
                "ee_body_name": "arm0_link_fngr",
            },
        )
        
        # Chair state for manipulation
        chair_pos_rel = ObsTerm(
            func=custom_observations.chair_position_relative,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
            },
        )
        chair_velocity = ObsTerm(
            func=custom_observations.chair_velocity,
            params={"chair_cfg": SceneEntityCfg("chair")},
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):
        """Full state observations for critic/value function (privileged info)."""
        
        # All robot state
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)
        
        # Commands
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        goal_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "goal_position"},
        )
        
        # Contacts
        feet_contact = ObsTerm(
            func=custom_observations.contact_binary_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
        )
        gripper_contact = ObsTerm(
            func=custom_observations.gripper_contact_state,
            params={"sensor_cfg": SceneEntityCfg("gripper_contact"), "threshold": 1.0},
        )
        
        # Global positions (privileged)
        robot_pos_global = ObsTerm(
            func=custom_observations.robot_position_global,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        chair_pos_global = ObsTerm(
            func=custom_observations.chair_position_global,
            params={"chair_cfg": SceneEntityCfg("chair")},
        )
        goal_pos_global = ObsTerm(
            func=custom_observations.goal_position_global,
        )
        
        # Relative positions
        chair_pos_rel = ObsTerm(
            func=custom_observations.chair_position_relative,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
            },
        )
        goal_pos_rel = ObsTerm(
            func=custom_observations.goal_position_relative,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        
        # Distances
        dist_to_chair = ObsTerm(
            func=custom_observations.distance_robot_to_chair,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
            },
        )
        dist_chair_to_goal = ObsTerm(
            func=custom_observations.distance_chair_to_goal,
            params={"chair_cfg": SceneEntityCfg("chair")},
        )
        dist_ee_to_chair = ObsTerm(
            func=custom_observations.distance_ee_to_chair,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "chair_cfg": SceneEntityCfg("chair"),
                "ee_body_name": "arm0_link_fngr",
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False  # No noise for critic
            self.concatenate_terms = True
    
    @configclass
    class CameraCfg(ObsGroup):
        """Camera observations for CNN policy with RGB, depth, and segmentation."""
        
        # Front camera RGB
        front_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"},
        )
        
        # Front camera depth
        front_depth = ObsTerm(
            func=custom_observations.front_camera_depth,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera")},
        )
        
        # Front camera instance segmentation
        front_segmentation = ObsTerm(
            func=custom_observations.front_camera_segmentation,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera")},
        )
        
        # Wrist camera RGB
        wrist_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "rgb"},
        )
        
        # Wrist camera depth
        wrist_depth = ObsTerm(
            func=custom_observations.camera_depth_wrist,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )
        
        # Wrist camera instance segmentation
        wrist_segmentation = ObsTerm(
            func=custom_observations.camera_segmentation_wrist,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )
        
        # Estimated depth to segmented object (chair) from wrist camera
        object_depth = ObsTerm(
            func=custom_observations.object_depth_from_segmentation,
            params={
                "camera_cfg": SceneEntityCfg("wrist_camera"),
                "target_object_cfg": SceneEntityCfg("chair"),
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # Keep as separate image observations
    
    # === Observation group assignments ===
    # Main policy (combines locomotion + manipulation)
    policy: PolicyCfg = PolicyCfg()
    
    # Critic has full state access (privileged information)
    critic: CriticCfg = CriticCfg()
    
    # Specialized policy groups (for hierarchical/multi-policy training)
    locomotion: LocomotionCfg = LocomotionCfg()
    manipulation: ManipulationCfg = ManipulationCfg()
    
    # Camera observations (enabled separately for vision-based training)
    # camera: CameraCfg = CameraCfg()  # Uncomment for CNN policy


# =============================================================================
# Events configuration
# =============================================================================

@configclass
class EventsCfg:
    """Event/randomization configuration."""
    
    # Startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    # Reset events
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Interval events - push robot occasionally
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# =============================================================================
# Rewards configuration - Loggable with hyperparameters
# =============================================================================

@configclass
class RewardsCfg:
    """Reward terms for the MDP with configurable weights.
    
    Reward Hyperparameters:
    - All rewards are scaled by dt automatically by RewardManager
    - Positive weights = rewards (maximize)
    - Negative weights = penalties (minimize)
    
    Training Phases:
    1. Navigation: Focus on robot-chair distance and locomotion
    2. Grasping: Focus on gripper-chair contact
    3. Manipulation: Focus on chair-goal distance
    """
    
    # === Alive and termination rewards ===
    is_alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0,
    )
    
    # === Base height rewards (standing between 0.5 and 1.0) ===
    # Reward for maintaining proper standing height
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.7,  # Target height in meters
        },
    )
    
    # === Orientation rewards ===
    # Penalize non-flat base orientation
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === Velocity tracking rewards ===
    # Penalize vertical velocity (bouncing)
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Penalize roll/pitch angular velocity
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === Velocity command tracking ===
    # Reward tracking linear velocity commands
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,  # std for exponential kernel
        },
    )
    # Reward tracking angular velocity commands
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,  # std for exponential kernel
        },
    )
    
    # === Joint regularization ===
    # Penalize high joint torques
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Penalize high joint velocities
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Penalize high joint accelerations
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Penalize joint position deviation from default
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # === Action regularization ===
    # Penalize large action changes
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    
    # === Contact rewards ===
    # Penalize undesired body contacts (contact on non-foot bodies)
    # Note: contact_forces sensor only tracks feet, so this is effectively disabled
    # Reward desired foot contacts
    feet_contact = RewTerm(
        func=mdp.desired_contacts,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "threshold": 1.0,
        },
    )
    
    # === Gripper contact reward ===
    # Reward when gripper contacts chair
    gripper_contact = RewTerm(
        func=mdp.contact_forces,
        weight=5.0,  # High reward for grasping
        params={
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
            "threshold": 1.0,
        },
    )
    
    # Reward when wrist approaches chair (intermediate contact)
    wrist_contact = RewTerm(
        func=mdp.contact_forces,
        weight=2.0,  # Medium reward for approach
        params={
            "sensor_cfg": SceneEntityCfg("wrist_contact"),
            "threshold": 1.0,
        },
    )
    
    # Note: Distance-based rewards (robot-chair, chair-goal, etc.) require
    # custom reward functions that access env.goal_positions.
    # These should be added using the custom reward functions in mdp/rewards.py
    # Example usage in extended config:
    # 
    # distance_robot_chair = RewTerm(
    #     func=custom_mdp.distance_robot_to_chair,
    #     weight=2.0,
    #     params={"std": 2.0, "robot_cfg": SceneEntityCfg("robot"), "chair_cfg": SceneEntityCfg("chair")},
    # )
    # distance_chair_goal = RewTerm(
    #     func=custom_mdp.distance_chair_to_goal,
    #     weight=5.0,
    #     params={"std": 2.0, "chair_cfg": SceneEntityCfg("chair")},
    # )


# =============================================================================
# Terminations configuration
# =============================================================================

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    # Time limit
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    
    # Fall detection - contact on body (torso) only, NOT feet
    # Only terminate if the main body/torso touches the ground
    # Feet contacts are expected and should not cause termination
    # Uses the dedicated body_contact sensor
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("body_contact"),
            "threshold": 10.0,  # Higher threshold to avoid false positives
        },
    )
    
    # Excessive tilt termination (>60 degrees from horizontal)
    # Uses the angle between robot's up vector and world up vector
    # Relaxed to 60 degrees to allow more time for stabilization during landing
    excessive_tilt = DoneTerm(
        func=custom_terminations.excessive_tilt,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_tilt_rad": 1.0472,  # 60 degrees in radians (pi/3)
            "grace_period_s": 1.0,  # Disable for first 1 second after reset
        },
    )
    
    # Base height out of bounds (fallen or flying)
    # Relaxed min_height to allow robot to compress when landing
    base_height_limit = DoneTerm(
        func=custom_terminations.base_height_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.05,  # Below this = truly fallen flat
            "max_height": 2.0,   # Above this = flying/glitched
            "grace_period_s": 1.0,  # Disable for first 1 second after reset
        },
    )


# =============================================================================
# Commands configuration
# =============================================================================

@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    
    # Velocity command for locomotion
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 2.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.5, 1.5),
        ),
    )
    
    # Goal position command for chair target
    goal_position = custom_commands.GoalPositionCommandCfg(
        class_type=custom_commands.GoalPositionCommand,
        pos_x_range=(-4.0, 4.0),
        pos_y_range=(-4.0, 4.0),
        pos_z=0.0,
        resampling_time_range=(30.0, 30.0),
        debug_vis=True,
        goal_marker_scale=(0.5, 0.5, 0.5),
        goal_marker_color=(0.0, 1.0, 0.0, 0.8),
    )


# =============================================================================
# Main environment configuration
# =============================================================================

@configclass
class LocomotionSpotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the manager-based locomotion spot environment."""
    
    # Scene settings
    scene: LocomotionSpotSceneCfg = LocomotionSpotSceneCfg(num_envs=64, env_spacing=8.0)
    
    # Simulation settings - use cuda:0 which maps to first visible GPU
    sim: SimulationCfg = SimulationCfg(device="cuda:0")
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    
    # No curriculum by default
    curriculum = None
    
    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4  # 50 Hz control
        self.episode_length_s = 30.0
        
        # Simulation settings
        self.sim.dt = 0.005  # 200 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        
        # Update sensor periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.gripper_contact is not None:
            self.scene.gripper_contact.update_period = self.sim.dt
        if self.scene.wrist_contact is not None:
            self.scene.wrist_contact.update_period = self.sim.dt


@configclass
class LocomotionSpotEnvCfg_PLAY(LocomotionSpotEnvCfg):
    """Configuration for playing/evaluating the policy."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 10.0
        
        # Disable randomization
        self.events.push_robot = None
        
        # Longer episodes
        self.episode_length_s = 60.0


@configclass  
class LocomotionSpotVisionEnvCfg(LocomotionSpotEnvCfg):
    """Configuration with vision policy enabled (depth + segmentation)."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Enable camera observations
        # Uncomment the camera observation group by enabling it
        self.observations.camera = ObservationsCfg.CameraCfg()
        
        # Reduce num_envs for vision-based training (more GPU memory needed)
        self.scene.num_envs = 16


@configclass
class LocomotionSpotVisionEnvCfg_PLAY(LocomotionSpotVisionEnvCfg):
    """Configuration for playing/evaluating the vision policy."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Even smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 10.0
        
        # Disable randomization
        self.events.push_robot = None
        
        # Longer episodes
        self.episode_length_s = 60.0
