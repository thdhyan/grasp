# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended environment configurations with custom reward terms.

These configurations provide phase-specific training setups:
1. Navigation phase: Focus on moving robot to chair
2. Grasping phase: Focus on gripper-chair contact
3. Manipulation phase: Focus on moving chair to goal
"""

from __future__ import annotations

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .locomotion_spot_env_cfg import (
    LocomotionSpotEnvCfg,
    LocomotionSpotSceneCfg,
    ActionsCfg,
    ObservationsCfg,
    EventsCfg,
    TerminationsCfg,
    CommandsCfg,
    RewardsCfg,
)

from .mdp import rewards as custom_rewards


# =============================================================================
# Navigation Phase Rewards
# =============================================================================

@configclass
class NavigationRewardsCfg(RewardsCfg):
    """Reward configuration focused on navigation to chair.
    
    This phase emphasizes:
    - Moving close to the chair
    - Facing the chair
    - Stable locomotion
    """
    
    # Override termination penalty (less harsh during navigation)
    termination_penalty = RewTerm(
        func=custom_rewards.termination_penalty,
        weight=-50.0,
    )
    
    # === Distance rewards ===
    # Reward for getting close to chair
    distance_to_chair = RewTerm(
        func=custom_rewards.distance_robot_to_chair,
        weight=5.0,  # High reward for approaching
        params={
            "std": 2.0,  # Distance scale
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # === Facing rewards ===
    # Reward for facing towards the chair
    facing_chair = RewTerm(
        func=custom_rewards.facing_chair_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # === Locomotion rewards ===
    # Reward for maintaining gait
    air_time = RewTerm(
        func=custom_rewards.air_time_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
        },
    )


# =============================================================================
# Grasping Phase Rewards
# =============================================================================

@configclass
class GraspingRewardsCfg(RewardsCfg):
    """Reward configuration focused on grasping the chair.
    
    This phase emphasizes:
    - End-effector approaching chair
    - Making contact with chair
    - Maintaining stable grasp
    """
    
    # === End-effector distance ===
    # Reward for EE being close to chair
    ee_distance = RewTerm(
        func=custom_rewards.ee_distance_to_chair,
        weight=3.0,
        params={
            "std": 0.5,
            "robot_cfg": SceneEntityCfg("robot", body_names=["arm0_link_wr1"]),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # === Contact rewards ===
    # High reward for gripper contact
    gripper_contact = RewTerm(
        func=custom_rewards.gripper_contact_chair_reward,
        weight=10.0,  # Very high reward for contact
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
        },
    )
    
    # Disable locomotion rewards during grasping (stationary)
    air_time = None


# =============================================================================
# Manipulation Phase Rewards
# =============================================================================

@configclass
class ManipulationRewardsCfg(RewardsCfg):
    """Reward configuration focused on moving chair to goal.
    
    This phase emphasizes:
    - Maintaining grasp on chair
    - Moving chair towards goal
    - Reaching goal position
    """
    
    # === Distance rewards ===
    # Reward for chair being close to goal
    chair_to_goal = RewTerm(
        func=custom_rewards.distance_chair_to_goal,
        weight=5.0,
        params={
            "std": 2.0,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Fine-grained reward when close to goal
    chair_to_goal_fine = RewTerm(
        func=custom_rewards.distance_chair_to_goal_fine,
        weight=10.0,
        params={
            "std": 0.5,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # Reward for robot facing goal
    facing_goal = RewTerm(
        func=custom_rewards.facing_goal_reward,
        weight=1.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    
    # === Success reward ===
    # Bonus for chair reaching goal
    chair_at_goal = RewTerm(
        func=custom_rewards.chair_at_goal_reward,
        weight=50.0,  # Big bonus for success
        params={
            "threshold": 0.5,  # Within 0.5m of goal
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # === Contact maintenance ===
    # Penalty for losing contact (should keep grasping)
    gripper_contact = RewTerm(
        func=custom_rewards.gripper_contact_chair_reward,
        weight=2.0,  # Moderate reward to maintain contact
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
        },
    )


# =============================================================================
# Combined Full Task Rewards
# =============================================================================

@configclass
class FullTaskRewardsCfg(RewardsCfg):
    """Combined reward configuration for full task training.
    
    This configuration includes rewards for all phases:
    - Navigation (lower weight)
    - Grasping (medium weight)
    - Manipulation (high weight)
    """
    
    # === Navigation rewards (Phase 1) ===
    distance_to_chair = RewTerm(
        func=custom_rewards.distance_robot_to_chair,
        weight=2.0,
        params={
            "std": 2.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    facing_chair = RewTerm(
        func=custom_rewards.facing_chair_reward,
        weight=0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    
    # === Grasping rewards (Phase 2) ===
    ee_distance = RewTerm(
        func=custom_rewards.ee_distance_to_chair,
        weight=3.0,
        params={
            "std": 0.5,
            "robot_cfg": SceneEntityCfg("robot", body_names=["arm0_link_wr1"]),
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    gripper_contact = RewTerm(
        func=custom_rewards.gripper_contact_chair_reward,
        weight=5.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("gripper_contact"),
        },
    )
    
    # === Manipulation rewards (Phase 3) ===
    chair_to_goal = RewTerm(
        func=custom_rewards.distance_chair_to_goal,
        weight=4.0,
        params={
            "std": 2.0,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    chair_to_goal_fine = RewTerm(
        func=custom_rewards.distance_chair_to_goal_fine,
        weight=8.0,
        params={
            "std": 0.5,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )
    chair_at_goal = RewTerm(
        func=custom_rewards.chair_at_goal_reward,
        weight=20.0,
        params={
            "threshold": 0.5,
            "chair_cfg": SceneEntityCfg("chair"),
        },
    )


# =============================================================================
# Phase-specific Environment Configurations
# =============================================================================

@configclass
class LocomotionSpotNavigationEnvCfg(LocomotionSpotEnvCfg):
    """Environment configuration for navigation phase training."""
    
    rewards: NavigationRewardsCfg = NavigationRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Shorter episodes for navigation
        self.episode_length_s = 15.0


@configclass
class LocomotionSpotGraspingEnvCfg(LocomotionSpotEnvCfg):
    """Environment configuration for grasping phase training."""
    
    rewards: GraspingRewardsCfg = GraspingRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Medium episodes for grasping
        self.episode_length_s = 20.0


@configclass
class LocomotionSpotManipulationEnvCfg(LocomotionSpotEnvCfg):
    """Environment configuration for manipulation phase training."""
    
    rewards: ManipulationRewardsCfg = ManipulationRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Longer episodes for manipulation
        self.episode_length_s = 40.0


@configclass
class LocomotionSpotFullTaskEnvCfg(LocomotionSpotEnvCfg):
    """Environment configuration for full task (all phases) training."""
    
    rewards: FullTaskRewardsCfg = FullTaskRewardsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # Long episodes for full task
        self.episode_length_s = 60.0
        # More environments for stable training
        self.scene.num_envs = 128
