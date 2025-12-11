# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for Spot locomanipulation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0
) -> torch.Tensor:
    """Terminate if there are illegal contacts with specified bodies.
    
    Args:
        env: The environment.
        sensor_cfg: The sensor configuration with body_ids for illegal contact bodies.
        threshold: Force threshold for contact detection.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get net forces
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    
    # Check if any body exceeds threshold
    if sensor_cfg.body_ids is not None:
        force_magnitudes = force_magnitudes[:, sensor_cfg.body_ids]
    
    has_contact = torch.any(force_magnitudes > threshold, dim=-1)
    
    return has_contact


def base_height_limit(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.1,
    maximum_height: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grace_period_s: float = 0.0
) -> torch.Tensor:
    """Terminate if base height is outside limits.
    
    Args:
        env: The environment.
        minimum_height: Minimum allowed height.
        maximum_height: Maximum allowed height.
        asset_cfg: The robot asset configuration.
        grace_period_s: Grace period after reset before checking.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get base height relative to env origin
    base_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Check limits
    too_low = base_height < minimum_height
    too_high = base_height > maximum_height
    
    # Apply grace period
    if grace_period_s > 0.0:
        grace_steps = int(grace_period_s / env.step_dt)
        episode_mask = env.episode_length_buf > grace_steps
        return (too_low | too_high) & episode_mask
    
    return too_low | too_high


def excessive_tilt(
    env: ManagerBasedRLEnv,
    max_tilt_rad: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grace_period_s: float = 0.0
) -> torch.Tensor:
    """Terminate if robot tilts too much (roll or pitch).
    
    Args:
        env: The environment.
        max_tilt_rad: Maximum allowed tilt in radians.
        asset_cfg: The robot asset configuration.
        grace_period_s: Grace period after reset before checking.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get projected gravity
    proj_gravity = robot.data.projected_gravity_b
    
    # Compute tilt angle from gravity projection
    # gravity_z should be close to -1 when upright
    gravity_z = proj_gravity[:, 2]
    
    # Tilt angle = acos(|gravity_z|)
    tilt_angle = torch.acos(torch.clamp(torch.abs(gravity_z), min=0.0, max=1.0))
    
    # Check if exceeds limit
    tilted = tilt_angle > max_tilt_rad
    
    # Apply grace period
    if grace_period_s > 0.0:
        grace_steps = int(grace_period_s / env.step_dt)
        episode_mask = env.episode_length_buf > grace_steps
        return tilted & episode_mask
    
    return tilted


def joint_limits_exceeded(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.05
) -> torch.Tensor:
    """Terminate if joint positions exceed soft limits.
    
    Args:
        env: The environment.
        asset_cfg: The robot asset configuration.
        margin: Margin inside the hard limits.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get joint limits
    joint_pos = robot.data.joint_pos
    joint_pos_limits = robot.data.soft_joint_pos_limits
    
    lower_violated = torch.any(joint_pos < joint_pos_limits[..., 0] + margin, dim=-1)
    upper_violated = torch.any(joint_pos > joint_pos_limits[..., 1] - margin, dim=-1)
    
    return lower_violated | upper_violated


def object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.05
) -> torch.Tensor:
    """Terminate if object is dropped below minimum height.
    
    Args:
        env: The environment.
        object_cfg: The object asset configuration.
        min_height: Minimum allowed object height.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get object height relative to env origin
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    return obj_height < min_height


def object_out_of_reach(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_distance: float = 2.0
) -> torch.Tensor:
    """Terminate if object is too far from robot.
    
    Args:
        env: The environment.
        robot_cfg: The robot asset configuration.
        object_cfg: The object asset configuration.
        max_distance: Maximum allowed distance from robot base.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get positions
    robot_pos = robot.data.root_pos_w[:, :2]
    obj_pos = obj.data.root_pos_w[:, :2]
    
    # Compute distance
    distance = torch.norm(robot_pos - obj_pos, dim=-1)
    
    return distance > max_distance


def goal_reached(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "goal_position",
    threshold: float = 0.15
) -> torch.Tensor:
    """Terminate (success) if object is at goal position.
    
    Args:
        env: The environment.
        object_cfg: The object asset configuration.
        command_name: Name of the command containing goal position.
        threshold: Distance threshold for success.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get positions
    obj_pos = obj.data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command(command_name)[:, :2]
    
    # Compute distance
    distance = torch.norm(obj_pos - goal_pos, dim=-1)
    
    return distance < threshold


def self_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0
) -> torch.Tensor:
    """Terminate if arm collides with body.
    
    Args:
        env: The environment.
        sensor_cfg: The sensor configuration.
        threshold: Force threshold for collision detection.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    
    # Sum forces across all bodies
    total_force = torch.sum(force_magnitudes, dim=-1)
    
    return total_force > threshold


def robot_too_far(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_distance: float = 10.0
) -> torch.Tensor:
    """Terminate if robot has moved too far from env origin.
    
    Args:
        env: The environment.
        asset_cfg: The robot asset configuration.
        max_distance: Maximum allowed distance from env origin.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get robot position relative to env origin
    rel_pos = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    
    distance = torch.norm(rel_pos, dim=-1)
    
    return distance > max_distance


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate if episode has timed out.
    
    Args:
        env: The environment.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    return env.episode_length_buf >= env.max_episode_length


def task_failed(
    env: ManagerBasedRLEnv,
    max_attempts: int = 3
) -> torch.Tensor:
    """Terminate if task has failed too many times.
    
    This is a placeholder for more complex failure tracking.
    
    Args:
        env: The environment.
        max_attempts: Maximum number of attempts before termination.
        
    Returns:
        Boolean tensor indicating termination for each environment.
    """
    # This would require tracking failed attempts across reset
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
