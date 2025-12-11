# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event/randomization functions for Spot locomanipulation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_robot_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Reset robot to default pose.
    
    Args:
        env: The environment.
        env_ids: Environment indices to reset.
        asset_cfg: The robot asset configuration.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get default state
    default_root_state = robot.data.default_root_state[env_ids].clone()
    
    # Apply environment origins
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    
    # Set root state
    robot.write_root_state_to_sim(default_root_state, env_ids)
    
    # Reset joint positions to default
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def reset_robot_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple = ((-0.5, -0.5, 0.0), (0.5, 0.5, 0.0)),
    orientation_range: tuple = ((-0.1, -0.1, -3.14), (0.1, 0.1, 3.14)),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Reset robot to random pose within range.
    
    Args:
        env: The environment.
        env_ids: Environment indices to reset.
        position_range: Range for random position offset ((min_xyz), (max_xyz)).
        orientation_range: Range for random euler angles ((min_rpy), (max_rpy)).
        asset_cfg: The robot asset configuration.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get default state
    root_state = robot.data.default_root_state[env_ids].clone()
    
    # Random position offset
    pos_min = torch.tensor(position_range[0], device=env.device)
    pos_max = torch.tensor(position_range[1], device=env.device)
    pos_offset = torch.rand((len(env_ids), 3), device=env.device) * (pos_max - pos_min) + pos_min
    
    root_state[:, :3] += pos_offset + env.scene.env_origins[env_ids]
    
    # Random yaw (keep roll/pitch small for stability)
    yaw_min, yaw_max = orientation_range[0][2], orientation_range[1][2]
    random_yaw = torch.rand(len(env_ids), device=env.device) * (yaw_max - yaw_min) + yaw_min
    
    # Convert yaw to quaternion
    half_yaw = random_yaw * 0.5
    quat_w = torch.cos(half_yaw)
    quat_z = torch.sin(half_yaw)
    
    root_state[:, 3] = quat_w  # w
    root_state[:, 4] = 0.0     # x
    root_state[:, 5] = 0.0     # y
    root_state[:, 6] = quat_z  # z
    
    # Set root state
    robot.write_root_state_to_sim(root_state, env_ids)
    
    # Reset joint positions to default with small noise
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_noise = torch.randn_like(joint_pos) * 0.1
    joint_pos = joint_pos + joint_noise
    
    # Clamp to limits
    joint_pos = torch.clamp(
        joint_pos,
        robot.data.soft_joint_pos_limits[env_ids, :, 0],
        robot.data.soft_joint_pos_limits[env_ids, :, 1]
    )
    
    joint_vel = torch.zeros_like(joint_pos)
    
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def reset_object_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple = ((0.3, -0.3, 0.05), (0.8, 0.3, 0.1)),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
):
    """Reset object to random position in front of robot.
    
    Args:
        env: The environment.
        env_ids: Environment indices to reset.
        position_range: Range for random position ((min_xyz), (max_xyz)).
        object_cfg: The object asset configuration.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Random position
    pos_min = torch.tensor(position_range[0], device=env.device)
    pos_max = torch.tensor(position_range[1], device=env.device)
    random_pos = torch.rand((len(env_ids), 3), device=env.device) * (pos_max - pos_min) + pos_min
    
    # Add env origins
    random_pos += env.scene.env_origins[env_ids]
    
    # Create root state
    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = random_pos
    
    # Random yaw
    random_yaw = torch.rand(len(env_ids), device=env.device) * 2 * 3.14159 - 3.14159
    half_yaw = random_yaw * 0.5
    root_state[:, 3] = torch.cos(half_yaw)  # w
    root_state[:, 4] = 0.0  # x
    root_state[:, 5] = 0.0  # y
    root_state[:, 6] = torch.sin(half_yaw)  # z
    
    # Zero velocity
    root_state[:, 7:] = 0.0
    
    obj.write_root_state_to_sim(root_state, env_ids)


def reset_goal_random(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple = ((-2.0, -2.0, 0.0), (2.0, 2.0, 0.0)),
    command_name: str = "goal_position"
):
    """Reset goal position randomly.
    
    Args:
        env: The environment.
        env_ids: Environment indices to reset.
        position_range: Range for goal position ((min_xyz), (max_xyz)).
        command_name: Name of the command to update.
    """
    pos_min = torch.tensor(position_range[0], device=env.device)
    pos_max = torch.tensor(position_range[1], device=env.device)
    
    # Generate random positions
    random_pos = torch.rand((len(env_ids), 3), device=env.device) * (pos_max - pos_min) + pos_min
    
    # Add env origins
    random_pos += env.scene.env_origins[env_ids]
    
    # Update command
    env.command_manager.set_command(command_name, random_pos, env_ids)


def randomize_robot_joint_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_range: tuple = (-0.25, 0.25)
):
    """Randomize joint positions around default.
    
    Args:
        env: The environment.
        env_ids: Environment indices.
        asset_cfg: The robot asset configuration.
        position_range: Range for position offset as fraction of joint limits.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get default positions
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    
    # Get joint ranges
    joint_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_range = joint_limits[:, :, 1] - joint_limits[:, :, 0]
    
    # Random offset
    offset_scale = torch.rand_like(joint_pos) * (position_range[1] - position_range[0]) + position_range[0]
    offset = offset_scale * joint_range
    
    joint_pos = joint_pos + offset
    
    # Clamp to limits
    joint_pos = torch.clamp(joint_pos, joint_limits[:, :, 0], joint_limits[:, :, 1])
    
    # Zero velocity
    joint_vel = torch.zeros_like(joint_pos)
    
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


def randomize_external_forces(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_range: tuple = ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
):
    """Apply random external forces to robot.
    
    Args:
        env: The environment.
        env_ids: Environment indices.
        asset_cfg: The robot asset configuration.
        force_range: Range for random forces ((min_xyz), (max_xyz)).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    force_min = torch.tensor(force_range[0], device=env.device)
    force_max = torch.tensor(force_range[1], device=env.device)
    
    # Random forces
    random_forces = torch.rand((len(env_ids), 3), device=env.device) * (force_max - force_min) + force_min
    
    # Apply to root body
    # Note: This requires external force application API which may vary
    # This is a placeholder implementation
    pass


def randomize_mass_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mass_range: tuple = (0.8, 1.2)
):
    """Randomize robot mass properties for domain randomization.
    
    Args:
        env: The environment.
        env_ids: Environment indices.
        asset_cfg: The robot asset configuration.
        mass_range: Range for mass multiplier (min, max).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Random mass multiplier
    mass_mult = torch.rand(len(env_ids), device=env.device) * (mass_range[1] - mass_range[0]) + mass_range[0]
    
    # Apply mass scaling (placeholder - actual implementation depends on API)
    # robot.set_mass_scale(mass_mult, env_ids)
    pass


def randomize_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    friction_range: tuple = (0.5, 1.5)
):
    """Randomize ground friction for domain randomization.
    
    Args:
        env: The environment.
        env_ids: Environment indices.
        asset_cfg: The robot asset configuration.
        friction_range: Range for friction coefficient (min, max).
    """
    # Placeholder - actual implementation depends on physics API
    pass
