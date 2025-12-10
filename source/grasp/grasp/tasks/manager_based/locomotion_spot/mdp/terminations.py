# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination functions for the locomotion spot navigation and manipulation task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def excessive_tilt(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt_rad: float = 1.0472,  # 60 degrees in radians (pi/3)
    grace_period_s: float = 1.0,  # Disable termination for first N seconds
) -> torch.Tensor:
    """Termination: excessive tilt or roll beyond threshold.
    
    Terminates the episode if the robot's orientation deviates too much
    from the horizontal plane. This detects when the robot has fallen over
    or is tilting dangerously.
    
    The tilt is measured by computing the angle between the robot's up vector
    (z-axis in body frame) and the world up vector (0, 0, 1).
    
    A grace period is provided to allow the robot to stabilize after reset/spawn.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot asset.
        max_tilt_rad: Maximum allowed tilt angle in radians. Default is 60 degrees (pi/3).
        grace_period_s: Time in seconds after reset during which termination is disabled.
        
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Check grace period - don't terminate during first N seconds after reset
    # episode_length_buf counts steps since last reset
    grace_period_steps = int(grace_period_s / env.step_dt)
    in_grace_period = env.episode_length_buf < grace_period_steps
    
    # Get robot quaternion (w, x, y, z)
    quat = robot.data.root_quat_w
    
    # Compute the robot's up vector in world frame
    # Start with (0, 0, 1) in body frame and rotate by quaternion
    up_body = torch.zeros(robot.data.root_pos_w.shape[0], 3, device=robot.device)
    up_body[:, 2] = 1.0
    
    # Rotate body up vector to world frame using quaternion
    up_world = _quat_rotate(quat, up_body)
    
    # The world up vector is (0, 0, 1)
    # Dot product gives cos(angle) between robot up and world up
    cos_angle = up_world[:, 2]  # This is dot product with (0, 0, 1)
    
    # Clamp to avoid numerical issues with acos
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    
    # Compute angle from vertical
    tilt_angle = torch.acos(cos_angle)
    
    # Terminate if tilt exceeds threshold, but not during grace period
    tilt_exceeded = tilt_angle > max_tilt_rad
    return tilt_exceeded & ~in_grace_period


def excessive_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_roll_rad: float = 0.7854,  # 45 degrees in radians
) -> torch.Tensor:
    """Termination: excessive roll beyond threshold.
    
    Terminates if the robot rolls too much (rotation about forward axis).
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot asset.
        max_roll_rad: Maximum allowed roll angle in radians. Default is 45 degrees.
        
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get robot quaternion
    quat = robot.data.root_quat_w
    
    # Extract roll angle from quaternion
    # Roll is rotation about x-axis (forward)
    roll = _quat_to_euler_roll(quat)
    
    # Terminate if roll exceeds threshold (check both directions)
    return torch.abs(roll) > max_roll_rad


def excessive_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_pitch_rad: float = 0.7854,  # 45 degrees in radians
) -> torch.Tensor:
    """Termination: excessive pitch beyond threshold.
    
    Terminates if the robot pitches too much (rotation about lateral axis).
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot asset.
        max_pitch_rad: Maximum allowed pitch angle in radians. Default is 45 degrees.
        
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get robot quaternion
    quat = robot.data.root_quat_w
    
    # Extract pitch angle from quaternion
    # Pitch is rotation about y-axis (lateral)
    pitch = _quat_to_euler_pitch(quat)
    
    # Terminate if pitch exceeds threshold (check both directions)
    return torch.abs(pitch) > max_pitch_rad


def base_height_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.05,
    max_height: float = 2.0,
    grace_period_s: float = 1.0,  # Disable termination for first N seconds
) -> torch.Tensor:
    """Termination: robot base height out of bounds.
    
    Terminates if the robot's base is too low (fallen) or too high (flying).
    
    A grace period is provided to allow the robot to land and stabilize after reset.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot asset.
        min_height: Minimum allowed height in meters.
        max_height: Maximum allowed height in meters.
        grace_period_s: Time in seconds after reset during which termination is disabled.
        
    Returns:
        Boolean tensor of shape (num_envs,) indicating termination.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Check grace period - don't terminate during first N seconds after reset
    grace_period_steps = int(grace_period_s / env.step_dt)
    in_grace_period = env.episode_length_buf < grace_period_steps
    
    # Get base height (z coordinate relative to env origin)
    base_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Terminate if height is out of bounds, but not during grace period
    height_out_of_bounds = (base_height < min_height) | (base_height > max_height)
    return height_out_of_bounds & ~in_grace_period


# =============================================================================
# Utility functions
# =============================================================================

def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion.
    
    Args:
        q: Quaternion (w, x, y, z) of shape (N, 4).
        v: Vector of shape (N, 3).
        
    Returns:
        Rotated vector of shape (N, 3).
    """
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    
    return a + b + c


def _quat_to_euler_roll(q: torch.Tensor) -> torch.Tensor:
    """Extract roll angle from quaternion.
    
    Args:
        q: Quaternion (w, x, y, z) of shape (N, 4).
        
    Returns:
        Roll angle in radians of shape (N,).
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Roll (rotation about x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    return roll


def _quat_to_euler_pitch(q: torch.Tensor) -> torch.Tensor:
    """Extract pitch angle from quaternion.
    
    Args:
        q: Quaternion (w, x, y, z) of shape (N, 4).
        
    Returns:
        Pitch angle in radians of shape (N,).
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Pitch (rotation about y-axis)
    sinp = 2.0 * (w * y - z * x)
    # Clamp to handle numerical issues near gimbal lock
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    return pitch
