# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def chair_grasped_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    chair_cfg: SceneEntityCfg,
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward for successfully grasping the chair with the robot's gripper.

    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot asset.
        chair_cfg: Configuration for the chair asset.
        contact_threshold: Minimum contact force to consider as a successful grasp.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]

    # Assuming the robot has a gripper end-effector named 'gripper'
    gripper_contacts = robot.get_contact_forces(end_effector_names=['gripper'])

    # Check if any contact force exceeds the threshold
    grasped = (gripper_contacts > contact_threshold).any(dim=1).float()

    return grasped

def joint_accel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalty for high joint accelerations to encourage smooth movements.

    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the asset.
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get joint accelerations for specified joints
    joint_accel = asset.data.joint_accel[:, asset_cfg.joint_ids]

    # Compute L2 norm of joint accelerations
    penalty = torch.sum(torch.square(joint_accel), dim=1)

    return penalty