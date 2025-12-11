# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for Spot locomanipulation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Locomotion Rewards
# =============================================================================

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """Reward for tracking linear velocity commands (exponential kernel)."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get commanded velocity
    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    
    # Get actual velocity
    vel_actual = robot.data.root_lin_vel_b[:, :2]
    
    # Compute error
    vel_error = torch.sum(torch.square(vel_cmd - vel_actual), dim=-1)
    
    # Exponential reward
    return torch.exp(-vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """Reward for tracking angular velocity command (exponential kernel)."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get commanded angular velocity
    ang_vel_cmd = env.command_manager.get_command(command_name)[:, 2]
    
    # Get actual angular velocity
    ang_vel_actual = robot.data.root_ang_vel_b[:, 2]
    
    # Compute error
    ang_vel_error = torch.square(ang_vel_cmd - ang_vel_actual)
    
    return torch.exp(-ang_vel_error / std**2)


def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.5,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining target base height."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get base height relative to env origin
    base_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Compute error
    height_error = torch.square(base_height - target_height)
    
    return torch.exp(-height_error / std**2)


def flat_orientation_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining flat orientation (minimizing roll and pitch)."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get projected gravity (should be [0, 0, -1] if flat)
    proj_gravity = robot.data.projected_gravity_b
    
    # Penalize deviation from [0, 0, -1]
    orientation_error = torch.sum(torch.square(proj_gravity[:, :2]), dim=-1)
    
    return torch.exp(-orientation_error / std**2)


def feet_air_time_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    mode_time: float = 0.3,
    velocity_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for appropriate feet air time during locomotion."""
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact data
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    
    # Only reward air time when moving
    vel_magnitude = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    moving = vel_magnitude > velocity_threshold
    
    # Reward air time close to mode_time
    air_time_reward = torch.sum(
        (last_air_time - mode_time) * first_contact, 
        dim=-1
    ) / last_air_time.shape[-1]
    
    return air_time_reward * moving.float()


def gait_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    std: float = 0.1,
    velocity_threshold: float = 0.5,
    synced_feet_pair_names: tuple = (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for trotting gait (diagonal feet contact)."""
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    
    # Check if moving
    vel_magnitude = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=-1)
    moving = vel_magnitude > velocity_threshold
    
    # Simple gait reward: encourage some feet in contact, some in air
    num_contacts = torch.sum(force_magnitudes > 1.0, dim=-1).float()
    
    # Ideal is 2 feet in contact (trot)
    gait_error = torch.square(num_contacts - 2.0)
    
    reward = torch.exp(-gait_error / std**2)
    
    return reward * moving.float()


def goal_reach_reward(
    env: ManagerBasedRLEnv,
    std: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position"
) -> torch.Tensor:
    """Reward for reaching the goal position with the robot base.
    
    Uses exponential kernel based on distance to goal.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get robot base position
    base_pos = robot.data.root_pos_w[:, :2]  # xy position
    
    # Get goal position from command manager
    goal_pos = env.command_manager.get_command(command_name)[:, :2]  # xy goal
    
    # Compute distance to goal
    distance = torch.norm(base_pos - goal_pos, dim=-1)
    
    # Exponential reward (closer = higher reward)
    reward = torch.exp(-distance / std)
    
    return reward


def goal_heading_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_position"
) -> torch.Tensor:
    """Reward for facing towards the goal position."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get robot base position and orientation
    base_pos = robot.data.root_pos_w[:, :2]
    base_quat = robot.data.root_quat_w
    
    # Get goal position
    goal_pos = env.command_manager.get_command(command_name)[:, :2]
    
    # Compute direction to goal
    direction_to_goal = goal_pos - base_pos
    goal_angle = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])
    
    # Get robot heading from quaternion (yaw)
    # Extract yaw from quaternion
    qw, qx, qy, qz = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    robot_yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    # Compute heading error (wrapped to [-pi, pi])
    heading_error = goal_angle - robot_yaw
    heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
    
    # Exponential reward
    reward = torch.exp(-torch.square(heading_error) / std**2)
    
    return reward


# =============================================================================
# Manipulation Rewards
# =============================================================================

def ee_position_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "arm0_link_fngr",
    command_name: str = "ee_target"
) -> torch.Tensor:
    """Reward for reaching end-effector target position."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get EE position
    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]
    
    # Get target
    target = env.command_manager.get_command(command_name)[:, :3]
    
    # Compute error
    pos_error = torch.sum(torch.square(ee_pos - target), dim=-1)
    
    return torch.exp(-pos_error / std**2)


def object_reach_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_name: str = "arm0_link_fngr"
) -> torch.Tensor:
    """Reward for reaching the object with end-effector."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get positions
    body_idx = robot.find_bodies(body_name)[0][0]
    ee_pos = robot.data.body_pos_w[:, body_idx]
    obj_pos = obj.data.root_pos_w
    
    # Compute distance
    distance = torch.norm(ee_pos - obj_pos, dim=-1)
    
    return torch.exp(-distance / std)


def object_grasp_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
    threshold: float = 5.0
) -> torch.Tensor:
    """Reward for grasping the object."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact force
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]
    force_magnitude = torch.norm(contact_forces, dim=-1)
    
    # Sum over all bodies in the sensor
    total_force = torch.sum(force_magnitude, dim=-1)
    
    # Binary reward for contact above threshold
    return (total_force > threshold).float()


def object_lift_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = 0.1,
    target_height: float = 0.5,
    std: float = 0.1
) -> torch.Tensor:
    """Reward for lifting the object."""
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get object height relative to env origin
    obj_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    
    # Reward for height above minimum
    height_above_min = torch.clamp(obj_height - min_height, min=0.0)
    
    # Bonus for reaching target height
    height_error = torch.square(obj_height - target_height)
    target_bonus = torch.exp(-height_error / std**2)
    
    return height_above_min + target_bonus


def object_to_goal_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "goal_position"
) -> torch.Tensor:
    """Reward for moving object towards goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get object position
    obj_pos = obj.data.root_pos_w
    
    # Get goal position
    goal_pos = env.command_manager.get_command(command_name)
    
    # Compute distance
    distance = torch.norm(obj_pos[:, :2] - goal_pos[:, :2], dim=-1)
    
    return torch.exp(-distance / std)


def object_at_goal_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "goal_position"
) -> torch.Tensor:
    """Sparse reward for placing object at goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    
    obj_pos = obj.data.root_pos_w
    goal_pos = env.command_manager.get_command(command_name)
    
    distance = torch.norm(obj_pos[:, :2] - goal_pos[:, :2], dim=-1)
    
    return (distance < threshold).float()


# =============================================================================
# Regularization Rewards (Penalties)
# =============================================================================

def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large changes in actions."""
    if hasattr(env, '_prev_actions'):
        action_diff = env.action_manager.action - env._prev_actions
        return torch.sum(torch.square(action_diff), dim=-1)
    return torch.zeros(env.num_envs, device=env.device)


def joint_acc_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_acc), dim=-1)


def joint_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.applied_torque), dim=-1)


def joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel), dim=-1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0
) -> torch.Tensor:
    """Penalize foot slipping while in contact."""
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact status
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0]
    force_magnitudes = torch.norm(contact_forces, dim=-1)
    in_contact = force_magnitudes > threshold
    
    # Get foot velocities - use body_names pattern if set, else use body_ids
    if asset_cfg.body_names is not None:
        body_ids, _ = robot.find_bodies(asset_cfg.body_names)
        foot_velocities = robot.data.body_lin_vel_w[:, body_ids, :2]
    elif asset_cfg.body_ids is not None:
        foot_velocities = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    else:
        # Fallback: use all bodies (not ideal)
        foot_velocities = robot.data.body_lin_vel_w[:, :, :2]
    
    # Velocity magnitude
    vel_magnitude = torch.norm(foot_velocities, dim=-1)
    
    # Make sure dimensions match for element-wise multiplication
    if vel_magnitude.shape[-1] != in_contact.shape[-1]:
        # Different number of bodies, just sum velocities of feet in contact
        slip_penalty = torch.sum(vel_magnitude, dim=-1) * torch.any(in_contact, dim=-1).float()
    else:
        # Penalize velocity when in contact
        slip_penalty = torch.sum(vel_magnitude * in_contact.float(), dim=-1)
    
    return slip_penalty


def base_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize excessive base motion (z velocity, roll/pitch rates)."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Vertical velocity
    z_vel = torch.square(robot.data.root_lin_vel_b[:, 2])
    
    # Roll and pitch rates
    roll_pitch_rate = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=-1)
    
    return z_vel + roll_pitch_rate


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive (not terminated)."""
    return torch.ones(env.num_envs, device=env.device)


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for termination."""
    return env.termination_manager.terminated.float()
