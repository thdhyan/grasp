# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level locomanipulation environment configuration for Spot.

This environment trains a high-level policy that:
1. Uses a pre-trained low-level locomotion policy for base movement
2. Controls the arm via IK to manipulate objects
3. Coordinates locomotion and manipulation for pick-and-place tasks

Architecture:
- High-level policy outputs: base velocity commands + arm IK targets
- Low-level locomotion policy: converts velocity commands to joint positions
- Arm IK controller: converts EE targets to arm joint positions

Based on the Isaac Lab G1 locomanipulation architecture.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

from . import mdp as locomanip_mdp


# =============================================================================
# Scene Configuration
# =============================================================================

@configclass
class LocomanipulationSpotSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Spot locomanipulation."""
    
    # Environment settings
    num_envs: int = 256
    env_spacing: float = 8.0
    replicate_physics: bool = True
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    
    # Spot robot with arm
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
            pos=(0.0, 0.0, 0.55),
            joint_pos={
                # Legs in standing position
                "fl_hx": 0.0,
                "fl_hy": 0.8,
                "fl_kn": -1.5,
                "fr_hx": 0.0,
                "fr_hy": 0.8,
                "fr_kn": -1.5,
                "hl_hx": 0.0,
                "hl_hy": 0.8,
                "hl_kn": -1.5,
                "hr_hx": 0.0,
                "hr_hy": 0.8,
                "hr_kn": -1.5,
                # Arm ready for manipulation (within joint limits)
                "arm0_sh0": 0.0,
                "arm0_sh1": -1.0,   # Within limit [-3.14, 0.34]
                "arm0_el0": 0.0,
                "arm0_el1": 1.5,    # Within limit [-2.79, 2.79]
                "arm0_wr0": 0.0,
                "arm0_wr1": 0.0,
                "arm0_f1x": 0.0,
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
    
    # Target object for manipulation
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 0.9),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.1),
        ),
    )
    
    # Goal marker (visualization only)
    goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                opacity=0.5,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.1),
        ),
    )
    
    # Contact sensor for feet
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/.*_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=[],
    )
    
    # Contact sensor for gripper
    gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/arm0_link_fngr",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    
    # Contact sensor for body (fall detection)
    body_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/body",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=[],
    )
    
    # Front camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/body/frontright_fisheye",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
        data_types=["rgb", "depth"],
        width=128,
        height=128,
    )
    
    # Wrist camera
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/arm0_link_wr1/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=None,
        data_types=["rgb", "depth"],
        width=128,
        height=128,
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
# Actions Configuration
# =============================================================================

@configclass
class LocomanipulationSpotActionsCfg:
    """Action specifications for locomanipulation.
    
    High-level policy outputs:
    - Base velocity commands (3D: vx, vy, omega_z) -> sent to low-level policy
    - Arm IK target (6D or 7D: position + orientation)
    - Gripper command (1D: open/close)
    
    For now, we use direct joint control until the low-level policy is trained.
    """
    
    # Direct joint control for legs (will be replaced by low-level policy)
    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hx", ".*_hy", ".*_kn"],
        scale=0.25,
        use_default_offset=True,
    )
    
    # Direct joint control for arm (will be replaced by IK)
    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm0_sh.*", "arm0_el.*", "arm0_wr.*"],
        scale=0.1,
        use_default_offset=True,
    )
    
    # Gripper control
    gripper = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm0_f1x"],
        scale=1.0,
        use_default_offset=False,
    )


# =============================================================================
# Observations Configuration
# =============================================================================

@configclass
class LocomanipulationSpotObservationsCfg:
    """Observation specifications for locomanipulation."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations combining locomotion and manipulation."""
        
        # === Robot State ===
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # Joint states
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        
        # Previous actions
        actions = ObsTerm(func=mdp.last_action)
        
        # === End-Effector State ===
        ee_position = ObsTerm(
            func=locomanip_mdp.ee_position_w,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "arm0_link_fngr",
            },
        )
        ee_velocity = ObsTerm(
            func=locomanip_mdp.ee_velocity_b,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "arm0_link_fngr",
            },
        )
        
        # === Object State ===
        object_pos_rel = ObsTerm(
            func=locomanip_mdp.object_pos_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        
        # Distance from EE to object
        dist_ee_to_object = ObsTerm(
            func=locomanip_mdp.object_to_ee_distance,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "body_name": "arm0_link_fngr",
            },
        )
        
        # Object to EE position (in body frame)
        object_to_ee_pos = ObsTerm(
            func=locomanip_mdp.object_to_ee_pos_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
                "body_name": "arm0_link_fngr",
            },
        )
        
        # === Goal State ===
        goal_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "goal_position"},
        )
        
        # === Contact State ===
        gripper_contact_status = ObsTerm(
            func=locomanip_mdp.gripper_in_contact,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_contact"),
                "threshold": 1.0,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations with full state."""
        
        # Robot state
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_pos = ObsTerm(
            func=mdp.root_pos_w,
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
        
        # EE state
        ee_position = ObsTerm(
            func=locomanip_mdp.ee_position_w,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "arm0_link_fngr",
            },
        )
        
        # Object state
        object_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_vel = ObsTerm(
            func=locomanip_mdp.object_vel_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        
        # Goal
        goal_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "goal_position"},
        )
        
        # Contact
        gripper_contact_status = ObsTerm(
            func=locomanip_mdp.gripper_in_contact,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_contact"),
                "threshold": 1.0,
            },
        )
        feet_contact = ObsTerm(
            func=locomanip_mdp.feet_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "threshold": 1.0,
            },
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# =============================================================================
# Events (Randomization) Configuration
# =============================================================================

@configclass
class LocomanipulationSpotEventsCfg:
    """Event/randomization configuration for locomanipulation."""
    
    # Reset robot
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # Reset object
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (0.4, 0.8),
                "y": (-0.3, 0.3),
                "z": (0.05, 0.1),
            },
            "velocity_range": {},
        },
    )


# =============================================================================
# Rewards Configuration
# =============================================================================

@configclass
class LocomanipulationSpotRewardsCfg:
    """Reward specifications for locomanipulation."""
    
    # === Reaching Rewards ===
    reach_object = RewTerm(
        func=locomanip_mdp.object_reach_reward,
        weight=5.0,
        params={
            "std": 0.3,
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "body_name": "arm0_link_fngr",
        },
    )
    
    # === Grasping Rewards ===
    grasp_object = RewTerm(
        func=locomanip_mdp.object_grasp_reward,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
            "threshold": 5.0,
        },
    )
    
    # === Lifting Rewards ===
    lift_object = RewTerm(
        func=locomanip_mdp.object_lift_reward,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "min_height": 0.1,
            "target_height": 0.3,
            "std": 0.1,
        },
    )
    
    # === Transport Rewards ===
    object_to_goal = RewTerm(
        func=locomanip_mdp.object_to_goal_reward,
        weight=5.0,
        params={
            "std": 0.5,
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_position",
        },
    )
    
    # === Success Reward ===
    object_at_goal = RewTerm(
        func=locomanip_mdp.object_at_goal_reward,
        weight=100.0,
        params={
            "threshold": 0.2,
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_position",
        },
    )
    
    # === Locomotion Stability Rewards ===
    flat_orientation = RewTerm(
        func=locomanip_mdp.flat_orientation_reward,
        weight=0.5,
        params={
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    base_height = RewTerm(
        func=locomanip_mdp.base_height_reward,
        weight=0.5,
        params={
            "target_height": 0.5,
            "std": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # === Regularization (Penalties) ===
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-6,
    )
    
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
    )
    
    # Termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )
    
    # Alive bonus
    is_alive = RewTerm(
        func=locomanip_mdp.is_alive,
        weight=0.5,
    )


# =============================================================================
# Terminations Configuration
# =============================================================================

@configclass
class LocomanipulationSpotTerminationsCfg:
    """Termination specifications for locomanipulation."""
    
    # Time out
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    
    # Bad orientation (fallen over)
    bad_orientation = DoneTerm(
        func=locomanip_mdp.excessive_tilt,
        params={
            "max_tilt_rad": 1.0,  # ~57 degrees
            "asset_cfg": SceneEntityCfg("robot"),
            "grace_period_s": 1.0,
        },
    )
    
    # Base height limit
    base_height = DoneTerm(
        func=locomanip_mdp.base_height_limit,
        params={
            "minimum_height": 0.1,
            "maximum_height": 1.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "grace_period_s": 1.0,
        },
    )
    
    # Object dropped
    object_dropped = DoneTerm(
        func=locomanip_mdp.object_dropped,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "min_height": -0.1,  # Allow some margin below ground
        },
    )
    
    # Success: object at goal
    goal_reached = DoneTerm(
        func=locomanip_mdp.goal_reached,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "goal_position",
            "threshold": 0.15,
        },
    )


# =============================================================================
# Commands Configuration
# =============================================================================

@configclass
class LocomanipulationSpotCommandsCfg:
    """Command specifications for locomanipulation."""
    
    # Goal position for object placement
    goal_position = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="body",
        resampling_time_range=(20.0, 30.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(1.5, 3.0),
            pos_y=(-1.5, 1.5),
            pos_z=(0.1, 0.1),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-math.pi, math.pi),
        ),
    )


# =============================================================================
# Curriculum Configuration
# =============================================================================

@configclass
class LocomanipulationSpotCurriculumCfg:
    """Curriculum configuration for locomanipulation."""
    
    pass  # Add curriculum terms as needed


# =============================================================================
# Main Environment Configuration
# =============================================================================

@configclass
class LocomanipulationSpotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Spot locomanipulation environment.
    
    This is the high-level policy that coordinates locomotion and manipulation
    for pick-and-place tasks.
    """
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        device="cuda:0",
    )
    
    # Scene
    scene: LocomanipulationSpotSceneCfg = LocomanipulationSpotSceneCfg()
    
    # MDP settings
    observations: LocomanipulationSpotObservationsCfg = LocomanipulationSpotObservationsCfg()
    actions: LocomanipulationSpotActionsCfg = LocomanipulationSpotActionsCfg()
    commands: LocomanipulationSpotCommandsCfg = LocomanipulationSpotCommandsCfg()
    rewards: LocomanipulationSpotRewardsCfg = LocomanipulationSpotRewardsCfg()
    terminations: LocomanipulationSpotTerminationsCfg = LocomanipulationSpotTerminationsCfg()
    events: LocomanipulationSpotEventsCfg = LocomanipulationSpotEventsCfg()
    curriculum: LocomanipulationSpotCurriculumCfg = LocomanipulationSpotCurriculumCfg()
    
    # Episode settings
    episode_length_s: float = 30.0
    decimation: int = 4
    
    def __post_init__(self):
        """Post-initialization configuration."""
        super().__post_init__()
        
        # Scene settings
        self.scene.num_envs = 256
        
        # Viewer settings
        self.viewer.eye = (5.0, 5.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class LocomanipulationSpotEnvCfg_PLAY(LocomanipulationSpotEnvCfg):
    """Play configuration for locomanipulation."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Reduce number of environments for visualization
        self.scene.num_envs = 50
        
        # Disable randomization
        self.observations.policy.enable_corruption = False


# =============================================================================
# Hierarchical Locomanipulation (with pre-trained locomotion policy)
# =============================================================================

@configclass
class HierarchicalLocomanipulationSpotEnvCfg(LocomanipulationSpotEnvCfg):
    """Hierarchical locomanipulation with pre-trained locomotion policy.
    
    This configuration is for when a low-level locomotion policy has been trained
    and can be used to control the legs while the high-level policy focuses on
    arm control and coordination.
    
    Action space:
    - Base velocity commands (3D) -> sent to low-level policy
    - Arm IK target (6D or 7D)
    - Gripper command (1D)
    """
    
    def __post_init__(self):
        super().__post_init__()
        
        # Note: To use hierarchical control, you need to:
        # 1. Train a low-level locomotion policy using SpotLocomotionFlatEnvCfg
        # 2. Replace leg_joint_pos action with LocomotionPolicyActionCfg
        # 3. Replace arm_joint_pos action with SpotArmIKActionCfg (if using IK)
        #
        # Example (requires trained policy):
        # self.actions.locomotion = locomanip_mdp.LocomotionPolicyActionCfg(
        #     policy_path="path/to/trained/locomotion/policy.pt",
        #     observation_group="policy",
        #     action_scale=1.0,
        # )
        pass
