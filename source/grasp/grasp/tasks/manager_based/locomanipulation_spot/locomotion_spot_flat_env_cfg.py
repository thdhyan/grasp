# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level locomotion environment configuration for Spot.

This environment trains a low-level locomotion policy that:
1. Tracks velocity commands (lin_x, lin_y, ang_z)
2. Maintains stable quadruped locomotion
3. Will be used by the high-level locomanipulation policy

Based on the Isaac Lab Spot flat_env_cfg.py architecture.
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from . import mdp as locomanip_mdp


# =============================================================================
# Scene Configuration
# =============================================================================

@configclass
class SpotLocomotionSceneCfg(InteractiveSceneCfg):
    """Scene configuration for Spot low-level locomotion."""
    
    # Environment settings
    num_envs: int = 1024
    env_spacing: float = 4.0
    replicate_physics: bool = True
    
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )
    
    # Spot robot (legs only for low-level locomotion)
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
            pos=(0.0, 0.0, 0.65),  # Higher spawn for safer landing
            joint_pos={
                # Leg joints at mid-range for easier control learning
                # hx (hip abduction): range ~ [-0.78, 0.78], mid = 0.0
                # hy (hip flexion): range ~ [-0.89, 2.79], mid ~ 0.95
                # kn (knee): range ~ [-2.79, -0.25], mid ~ -1.52
                "fl_hx": 0.0,
                "fl_hy": 0.9,   # Mid-range standing pose
                "fl_kn": -1.5,  # Mid-range knee bend
                "fr_hx": 0.0,
                "fr_hy": 0.9,
                "fr_kn": -1.5,
                "hl_hx": 0.0,
                "hl_hy": 0.9,
                "hl_kn": -1.5,
                "hr_hx": 0.0,
                "hr_hy": 0.9,
                "hr_kn": -1.5,
                # Arm tucked and neutral (mid-range where possible)
                "arm0_sh0": 0.0,      # Mid-range
                "arm0_sh1": -1.4,     # Mid of [-3.14, 0.34] ~ -1.4
                "arm0_el0": 0.0,      # Mid-range
                "arm0_el1": 0.0,      # Mid of [-2.79, 2.79] = 0
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
            # Arm is held fixed during low-level locomotion training
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["arm0_sh.*", "arm0_el.*", "arm0_wr.*", "arm0_f1x"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )
    
    # Contact sensor for feet
    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/.*_foot",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=[],
    )
    
    # Contact sensor for body (fall detection)
    body_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/body",
        update_period=0.0,
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=[],
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
class SpotLocomotionActionsCfg:
    """Action specifications for low-level locomotion.
    
    Only controls the 12 leg joints. Arm is held fixed.
    """
    
    # Joint position actions for legs only
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hx", ".*_hy", ".*_kn"],
        scale=0.25,
        use_default_offset=True,
    )


# =============================================================================
# Observations Configuration
# =============================================================================

@configclass
class SpotLocomotionObservationsCfg:
    """Observation specifications for low-level locomotion."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations for locomotion."""
        
        # Robot proprioception
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
        
        # Velocity commands
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        
        # Goal position (relative to robot base)
        goal_position = ObsTerm(
            func=locomanip_mdp.goal_pos_b,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        
        # Leg joint states
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hx", ".*_hy", ".*_kn"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hx", ".*_hy", ".*_kn"])},
            noise=Unoise(n_min=-1.0, n_max=1.0),
        )
        
        # Previous actions
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations with full state."""
        
        # All policy observations plus more
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
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # Goal position (world frame for critic)
        goal_pos = ObsTerm(
            func=locomanip_mdp.goal_pos_w,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hx", ".*_hy", ".*_kn"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hx", ".*_hy", ".*_kn"])},
        )
        actions = ObsTerm(func=mdp.last_action)
        
        # Feet contact - binary indicators for each foot
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
class SpotLocomotionEventsCfg:
    """Event/randomization configuration for low-level locomotion."""
    
    # Reset events
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            },
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # Push robot periodically
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


# =============================================================================
# Rewards Configuration
# =============================================================================

@configclass
class SpotLocomotionRewardsCfg:
    """Reward specifications for low-level locomotion."""
    
    # Track velocity commands
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
        },
    )
    
    # Penalize linear z velocity
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )
    
    # Penalize angular velocity xy
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    
    # Flat orientation
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.2,
    )
    
    # Base height - penalize deviation from target (L2 penalty)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.6},
    )
    
    # Base height - positive reward for maintaining target height
    base_height_exp = RewTerm(
        func=locomanip_mdp.base_height_reward,
        weight=0.5,
        params={"target_height": 0.6, "std": 0.1, "asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Goal reaching reward - exponential based on distance
    goal_reach = RewTerm(
        func=locomanip_mdp.goal_reach_reward,
        weight=1.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_position"},
    )
    
    # Goal heading reward - face towards the goal
    goal_heading = RewTerm(
        func=locomanip_mdp.goal_heading_reward,
        weight=0.3,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot"), "command_name": "goal_position"},
    )
    
    # Feet air time (for gait) - using custom reward function
    feet_air_time = RewTerm(
        func=locomanip_mdp.feet_air_time_reward,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Penalize action rate
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    
    # Penalize joint accelerations
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    
    # Penalize joint torques
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5,
    )
    
    # Penalize feet slip
    feet_slide = RewTerm(
        func=locomanip_mdp.foot_slip_penalty,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    
    # Undesired contacts
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("body_contact"),
            "threshold": 1.0,
        },
    )
    
    # Termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )


# =============================================================================
# Terminations Configuration
# =============================================================================

@configclass
class SpotLocomotionTerminationsCfg:
    """Termination specifications for low-level locomotion."""
    
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


# =============================================================================
# Commands Configuration
# =============================================================================

@configclass
class SpotLocomotionCommandsCfg:
    """Command specifications for low-level locomotion."""
    
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,  # Shows commanded velocity arrows
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    
    # Goal position command for navigation
    goal_position = locomanip_mdp.GoalPositionCommandCfg(
        resampling_time_range=(20.0, 30.0),  # Resample goal every 20-30 seconds
        pos_x_range=(-5.0, 5.0),
        pos_y_range=(-5.0, 5.0),
        pos_z=0.0,
        debug_vis=True,  # Shows goal position marker
    )


# =============================================================================
# Debug Visualization Configuration  
# =============================================================================

@configclass
class SpotLocomotionDebugVisCfg:
    """Debug visualization for velocity tracking."""
    
    # Velocity arrow visualization is handled by the command manager's debug_vis
    # Additional visualization can be added here if needed
    pass


# =============================================================================
# Curriculum Configuration
# =============================================================================

@configclass
class SpotLocomotionCurriculumCfg:
    """Curriculum configuration for low-level locomotion."""
    
    pass  # Add curriculum terms as needed


# =============================================================================
# Main Environment Configuration
# =============================================================================

@configclass
class SpotLocomotionFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Spot low-level locomotion flat terrain environment.
    
    This is the low-level locomotion policy that will be used by the high-level
    locomanipulation policy to command robot locomotion.
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
    scene: SpotLocomotionSceneCfg = SpotLocomotionSceneCfg()
    
    # MDP settings
    observations: SpotLocomotionObservationsCfg = SpotLocomotionObservationsCfg()
    actions: SpotLocomotionActionsCfg = SpotLocomotionActionsCfg()
    commands: SpotLocomotionCommandsCfg = SpotLocomotionCommandsCfg()
    rewards: SpotLocomotionRewardsCfg = SpotLocomotionRewardsCfg()
    terminations: SpotLocomotionTerminationsCfg = SpotLocomotionTerminationsCfg()
    events: SpotLocomotionEventsCfg = SpotLocomotionEventsCfg()
    curriculum: SpotLocomotionCurriculumCfg = SpotLocomotionCurriculumCfg()
    
    # Episode settings
    episode_length_s: float = 20.0
    decimation: int = 4
    
    def __post_init__(self):
        """Post-initialization configuration."""
        super().__post_init__()
        
        # Scene settings
        self.scene.num_envs = 1024
        
        # Viewer settings
        self.viewer.eye = (4.0, 4.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class SpotLocomotionFlatEnvCfg_PLAY(SpotLocomotionFlatEnvCfg):
    """Play configuration for low-level locomotion."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Reduce number of environments for visualization
        self.scene.num_envs = 50
        
        # Disable randomization
        self.observations.policy.enable_corruption = False
