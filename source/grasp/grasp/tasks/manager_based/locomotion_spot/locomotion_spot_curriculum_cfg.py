# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum-based environment configurations for progressive training.

5-Phase Curriculum:
1. Stand Up: Robot learns to stand up from ground
2. Walk: Robot learns stable locomotion
3. Arm Motion: Robot learns to move while controlling arm
4. Grasp: Robot learns to grasp the chair
5. Full Task: Robot learns complete task (stand, move, grasp, transport to goal)
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .locomotion_spot_env_cfg import (
    LocomotionSpotEnvCfg,
    RewardsCfg,
)
from .mdp import rewards as custom_rewards
from .mdp import curriculum as custom_curriculum


# =============================================================================
# Phase 1: Stand Up Rewards
# =============================================================================

@configclass
class StandUpRewardsCfg(RewardsCfg):
    """Reward configuration for Phase 1: Learning to stand up.
    
    Focus:
    - Maintaining proper base height (0.5 to 1.0)
    - Stable base orientation
    - Minimal energy expenditure
    - Disable arm and chair-related rewards
    """
    
    # Higher weight for base height
    base_height = RewTerm(
        func=custom_rewards.base_height_reward,
        weight=10.0,  # Primary reward
        params={
            "min_height": 0.5,
            "max_height": 1.0,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Penalize non-flat orientation heavily
    flat_orientation = RewTerm(
        func=custom_rewards.flat_orientation_penalty,
        weight=-10.0,  # Strong penalty
        params={
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Small penalty for joint effort
    joint_torques = RewTerm(
        func=custom_rewards.joint_torques_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Disable chair/arm rewards for this phase
    gripper_contact = None
    wrist_contact = None


# =============================================================================
# Phase 2: Walk Rewards  
# =============================================================================

@configclass
class WalkRewardsCfg(RewardsCfg):
    """Reward configuration for Phase 2: Learning to walk.
    
    Focus:
    - Velocity tracking
    - Foot air time (proper gait)
    - Stability while moving
    - Keep arm stable
    """
    
    # Velocity tracking
    track_lin_vel_xy = RewTerm(
        func=custom_rewards.track_lin_vel_xy_exp,
        weight=5.0,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    track_ang_vel_z = RewTerm(
        func=custom_rewards.track_ang_vel_z_exp,
        weight=2.5,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Foot air time for gait
    air_time = RewTerm(
        func=custom_rewards.air_time_reward,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "mode_time": 0.3,
        },
    )
    
    # Keep arm stable (penalize arm joint velocities)
    arm_stability = RewTerm(
        func=custom_rewards.arm_joint_vel_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm0_.*"]),
        },
    )
    
    # Disable chair rewards
    gripper_contact = None
    wrist_contact = None


# =============================================================================
# Phase 3: Arm Motion Rewards
# =============================================================================

@configclass
class ArmMotionRewardsCfg(RewardsCfg):
    """Reward configuration for Phase 3: Learning arm motion while walking.
    
    Focus:
    - Move robot towards chair
    - Extend arm towards chair
    - Maintain locomotion stability
    """
    
    # Navigate to chair
    distance_to_chair = RewTerm(
        func=custom_rewards.distance_robot_to_chair,
        weight=3.0,
        params={
            "std": 2.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Face the chair
    facing_chair = RewTerm(
        func=custom_rewards.facing_chair_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # End-effector approaching chair
    ee_to_chair = RewTerm(
        func=custom_rewards.ee_distance_to_chair,
        weight=2.0,
        params={
            "std": 1.0,
            "robot_cfg": SceneEntityCfg("robot", body_names=["arm0_link_wr1"]),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Keep velocity tracking
    track_lin_vel_xy = RewTerm(
        func=custom_rewards.track_lin_vel_xy_exp,
        weight=2.0,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Wrist approach (intermediate reward)
    wrist_contact = RewTerm(
        func=custom_rewards.contact_force_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_contact"),
            "threshold": 1.0,
        },
    )


# =============================================================================
# Phase 4: Grasp Rewards
# =============================================================================

@configclass
class GraspRewardsCfg(RewardsCfg):
    """Reward configuration for Phase 4: Learning to grasp the chair.
    
    Focus:
    - Gripper contact with chair
    - Firm grasp force
    - Stability while grasping
    """
    
    # Navigation (lower weight)
    distance_to_chair = RewTerm(
        func=custom_rewards.distance_robot_to_chair,
        weight=1.0,
        params={
            "std": 2.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # End-effector close to chair (high weight)
    ee_to_chair = RewTerm(
        func=custom_rewards.ee_distance_to_chair,
        weight=5.0,
        params={
            "std": 0.3,
            "robot_cfg": SceneEntityCfg("robot", body_names=["arm0_link_fngr"]),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Gripper contact reward (very high)
    gripper_contact = RewTerm(
        func=custom_rewards.contact_force_reward,
        weight=10.0,  # Very high for grasping
        params={
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
            "threshold": 1.0,
        },
    )
    
    # Wrist contact (intermediate)
    wrist_contact = RewTerm(
        func=custom_rewards.contact_force_reward,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_contact"),
            "threshold": 1.0,
        },
    )
    
    # Grasp firmness (sustained contact)
    grasp_firmness = RewTerm(
        func=custom_rewards.sustained_contact_reward,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
            "min_duration": 0.5,
        },
    )


# =============================================================================
# Phase 5: Full Task Rewards
# =============================================================================

@configclass
class FullTaskCurriculumRewardsCfg(RewardsCfg):
    """Reward configuration for Phase 5: Complete task (stand, move, grasp, transport).
    
    Focus:
    - All previous skills combined
    - Moving chair to goal position
    - Bonus for completing full task
    """
    
    # Standing/stability (always required)
    base_height = RewTerm(
        func=custom_rewards.base_height_reward,
        weight=2.0,
        params={
            "min_height": 0.5,
            "max_height": 1.0,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    
    # Navigation (lower weight in final phase)
    distance_to_chair = RewTerm(
        func=custom_rewards.distance_robot_to_chair,
        weight=0.5,
        params={
            "std": 2.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Grasping
    gripper_contact = RewTerm(
        func=custom_rewards.contact_force_reward,
        weight=3.0,
        params={
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
            "threshold": 1.0,
        },
    )
    
    wrist_contact = RewTerm(
        func=custom_rewards.contact_force_reward,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_contact"),
            "threshold": 1.0,
        },
    )
    
    # Chair to goal (primary objective)
    chair_to_goal = RewTerm(
        func=custom_rewards.distance_chair_to_goal,
        weight=10.0,  # Primary reward
        params={
            "std": 2.0,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Fine control when close to goal
    chair_to_goal_fine = RewTerm(
        func=custom_rewards.distance_chair_to_goal_fine,
        weight=15.0,
        params={
            "std": 0.5,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Big bonus for chair at goal
    chair_at_goal = RewTerm(
        func=custom_rewards.chair_at_goal_reward,
        weight=50.0,  # Large completion bonus
        params={
            "threshold": 0.3,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Velocity tracking
    track_lin_vel_xy = RewTerm(
        func=custom_rewards.track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# =============================================================================
# Curriculum Configuration
# =============================================================================

@configclass
class CurriculumCfg:
    """Curriculum configuration with reward weight scaling per phase.
    
    The curriculum scales reward weights based on training progress,
    measured by either iterations or average episode reward.
    """
    
    # Phase transition thresholds (by average reward or iterations)
    phase1_iterations = CurrTerm(
        func=custom_curriculum.phase_transition_by_iterations,
        params={
            "target_iterations": 50000,
            "phase": 1,
        },
    )
    
    phase2_iterations = CurrTerm(
        func=custom_curriculum.phase_transition_by_iterations,
        params={
            "target_iterations": 150000,
            "phase": 2,
        },
    )
    
    phase3_iterations = CurrTerm(
        func=custom_curriculum.phase_transition_by_iterations,
        params={
            "target_iterations": 300000,
            "phase": 3,
        },
    )
    
    phase4_iterations = CurrTerm(
        func=custom_curriculum.phase_transition_by_iterations,
        params={
            "target_iterations": 450000,
            "phase": 4,
        },
    )


# =============================================================================
# Phase-Specific Environment Configurations
# =============================================================================

@configclass
class LocomotionSpotPhase1Cfg(LocomotionSpotEnvCfg):
    """Phase 1: Stand Up - Robot learns to stand from ground."""
    
    rewards: StandUpRewardsCfg = StandUpRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Short episodes for standing
        self.episode_length_s = 10.0
        # Start from lying down position
        self.events.reset_base.params["pose_range"]["z"] = (0.2, 0.4)


@configclass
class LocomotionSpotPhase2Cfg(LocomotionSpotEnvCfg):
    """Phase 2: Walk - Robot learns stable locomotion."""
    
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Medium episodes for walking
        self.episode_length_s = 15.0


@configclass
class LocomotionSpotPhase3Cfg(LocomotionSpotEnvCfg):
    """Phase 3: Arm Motion - Robot learns to move arm while walking."""
    
    rewards: ArmMotionRewardsCfg = ArmMotionRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Medium-long episodes
        self.episode_length_s = 20.0


@configclass
class LocomotionSpotPhase4Cfg(LocomotionSpotEnvCfg):
    """Phase 4: Grasp - Robot learns to grasp the chair."""
    
    rewards: GraspRewardsCfg = GraspRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Longer episodes for grasping
        self.episode_length_s = 25.0


@configclass
class LocomotionSpotPhase5Cfg(LocomotionSpotEnvCfg):
    """Phase 5: Full Task - Complete stand, move, grasp, transport to goal."""
    
    rewards: FullTaskCurriculumRewardsCfg = FullTaskCurriculumRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Long episodes for full task
        self.episode_length_s = 60.0
        # More envs for stable training
        self.scene.num_envs = 128


@configclass
class LocomotionSpotCurriculumCfg(LocomotionSpotEnvCfg):
    """Full curriculum environment that transitions through all 5 phases.
    
    Uses automatic phase transitions based on training progress.
    """
    
    rewards: FullTaskCurriculumRewardsCfg = FullTaskCurriculumRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Long episodes
        self.episode_length_s = 60.0
        self.scene.num_envs = 128
