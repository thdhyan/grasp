# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

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

from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp
from . import mdp as local_mdp

from grasp_heir.assets.robots.spot import SPOT_CFG

##
# Scene definition
##

@configclass
class SpotLocomotionSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # robots
    robot: ArticulationCfg = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Force activate contact sensors (even if in SPOT_CFG, to be safe/explicit based on error hint)
    robot.spawn.activate_contact_sensors = True

    # lights
    lights = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # sensors - USD defaultPrim 'Root' is renamed to 'Robot' when spawned
    # So actual paths are: {ENV_REGEX_NS}/Robot/spot_arm_01/fl_foot etc.
    contact_forces_fl = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/fl_foot", 
        history_length=3, 
        track_air_time=True
    )
    contact_forces_fr = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/fr_foot", 
        history_length=3, 
        track_air_time=True
    )
    contact_forces_hl = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/hl_foot", 
        history_length=3, 
        track_air_time=True
    )
    contact_forces_hr = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/hr_foot", 
        history_length=3, 
        track_air_time=True  # Enabled for termination check
    )
    contact_forces_body = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/spot_arm_01/body", 
        history_length=3, 
        track_air_time=False,
    ) # Body contact sensor for undesired contacts penalty
##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Joint efforts for legs
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*_hx", ".*_hy", ".*_kn"], scale=10.0)

    # Arm is controlled via a dummy action (or just randomized events)
    # We define it here to prevent errors if we want to add it later, or we can leave it empty
    # For now, we rely on EventTerms to move the arm, so we don't put it in "actions" 
    # unless we want the policy to control it (which we don't).

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hx", ".*_hy", ".*_kn"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hx", ".*_hy", ".*_kn"])})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    base_soft_lim = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
            "pose_range": {
                "x": (-0.0, 0.0), 
                "y": (-0.0, 0.0), 
                "z": (0.75, 0.75),  # Fix inverted spawn: spawn higher
                "yaw": (-math.pi, math.pi),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
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

    # reset_goal = EventTerm(
    #     func=local_mdp.randomize_goal_pose,
    #     mode="reset",
    #     params={
    #         "position_range": (-1.0, 1.0), # Start small for curriculum
    #     },
    # )

    # interval - robustness
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)},
        },
    )
    
    randomize_arm = EventTerm(
        func=local_mdp.randomize_arm_joint_positions,
        mode="interval",
        interval_range_s=(2.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm0_.*"]),
            "position_range": (-0.5, 0.5), # Offset from default
        },
    )

    # External force on arm end effector
    force_on_arm = EventTerm(
         func=mdp.apply_external_force_torque,
         mode="interval",
         interval_range_s=(1.0, 4.0),
         params={
             "asset_cfg": SceneEntityCfg("robot", body_names="arm0_link_fngr"), # Approximate End Effector
             "force_range": (-20.0, 20.0),
             "torque_range": (-5.0, 5.0),
         }
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- task --
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties --
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0005)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Reward foot contact (requested)
    # rewarding having feet on the ground.
    # We can use 'insufficient_feet_contact' penalty or custom reward.
    # Or just 'feet_contact' with positive weight?
    # mdp.feet_contact usually returns bool.
    # Let's use `feet_air_time` with negative weight (penalty for air time) -> encourages contact.
    # But user said "reward foot contact".
    # I'll add a reward for feet contact forces > 0?
    # Simple workaround: a term that is 1.0 if contact > threshold.
    # Using 'undesired_contacts' with positive weight? No, that counts contacts.
    # Let's use `mdp.feet_contact_forces_mean`? No.
    # Reward foot contact - using fl foot as representative
    feet_contact = RewTerm(
        func=local_mdp.feet_contact_reward,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces_fl"),
            "threshold": 1.0,
        }
    )

    # Penalize body contacts (torso hitting ground = bad)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_body"), "threshold": 1.0},
    )

    # Penalize unequal air time across feet - encourages symmetric gait
    air_time_variance = RewTerm(
        func=local_mdp.air_time_variance_penalty,
        weight=-0.5,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("contact_forces_fl"),
                SceneEntityCfg("contact_forces_fr"),
                SceneEntityCfg("contact_forces_hl"),
                SceneEntityCfg("contact_forces_hr"),
            ],
            "threshold": 1.0,
        }
    )

    # -- New Rewards from User Request --
    # track_goal = RewTerm(
    #     func=local_mdp.track_goal_distance_exp,
    #     weight=1.0,
    #     params={"std": 1.0},
    # )
    
    # base_height = RewTerm(
    #     func=local_mdp.base_height_reward,
    #     weight=1.0,
    #     params={"min_height": 0.7, "max_height": 2.0},
    # )

    # Add a penalty for feet air time (copied from A1 but negative weight)
    feet_air_time_penalty = RewTerm(
        func=mdp.feet_air_time,
        weight=-0.005,  # Moderate penalty for total air time
        params={"sensor_cfg": SceneEntityCfg("contact_forces_fl"), "threshold": 0.05} # Or specific joint names/regex
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # Body contact
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_body"), "threshold": 1.0},
    )
    
    # Feet air time > 5s
    feet_air_time = DoneTerm(
        func=local_mdp.feet_air_time_termination,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("contact_forces_fl"),
                SceneEntityCfg("contact_forces_fr"),
                SceneEntityCfg("contact_forces_hl"),
                SceneEntityCfg("contact_forces_hr"),
            ],
            "time_threshold": 5.0,
        },
    )


from isaaclab.managers import CurriculumTermCfg as CurrTerm

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # goal_distance = CurrTerm(
    #     func=local_mdp.goal_distance_curriculum,
    #     params={
    #         "term_name": "reset_goal", 
    #         "start_dist": 1.0, 
    #         "end_dist": 5.0, 
    #         "total_steps": 1000, 
    #     },
    # )


@configclass
class SpotLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: SpotLocomotionSceneCfg = SpotLocomotionSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.spawn.physics_material

