# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the locomotion spot navigation and manipulation task.

These reward functions support multi-phase training:
1. Navigation: Move robot close to chair
2. Grasping: Grasp the chair with the arm
3. Manipulation: Move chair to goal position
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


# =============================================================================
# Distance-based rewards
# =============================================================================

def distance_robot_to_chair(
    env: ManagerBasedRLEnv,
    std: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Reward based on distance between robot and chair using tanh kernel.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for the tanh kernel (smaller = sharper).
        robot_cfg: Configuration for the robot entity.
        chair_cfg: Configuration for the chair entity.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    robot_pos = robot.data.root_pos_w[:, :2]  # XY position
    chair_pos = chair.data.root_pos_w[:, :2]
    
    distance = torch.norm(robot_pos - chair_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def distance_robot_to_goal(
    env: ManagerBasedRLEnv,
    std: float = 2.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward based on distance between robot and goal position using tanh kernel.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for the tanh kernel.
        robot_cfg: Configuration for the robot entity.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    robot_pos = robot.data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(robot_pos - goal_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def distance_chair_to_goal(
    env: ManagerBasedRLEnv,
    std: float = 2.0,
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Reward based on distance between chair and goal position using tanh kernel.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for the tanh kernel.
        chair_cfg: Configuration for the chair entity.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    chair: Articulation = env.scene[chair_cfg.name]
    
    chair_pos = chair.data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(chair_pos - goal_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def distance_chair_to_goal_fine(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Fine-grained reward when chair is close to goal.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for the tanh kernel (smaller for fine control).
        chair_cfg: Configuration for the chair entity.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    chair: Articulation = env.scene[chair_cfg.name]
    
    chair_pos = chair.data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(chair_pos - goal_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)


# =============================================================================
# Base height and stability rewards
# =============================================================================

def base_height_reward(
    env: ManagerBasedRLEnv,
    min_height: float = 0.5,
    max_height: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining proper standing height (between 0.5 and 1.0).
    
    Args:
        env: The RL environment instance.
        min_height: Minimum desired height.
        max_height: Maximum desired height.
        robot_cfg: Configuration for the robot entity.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    base_height = robot.data.root_pos_w[:, 2]
    
    # Reward is 1.0 when height is in range, 0.0 otherwise
    in_range = (base_height >= min_height) & (base_height <= max_height)
    return in_range.float()


def base_height_l2_penalty(
    env: ManagerBasedRLEnv,
    target_height: float = 0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty for deviation from target height.
    
    Args:
        env: The RL environment instance.
        target_height: Target base height.
        robot_cfg: Configuration for the robot entity.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    base_height = robot.data.root_pos_w[:, 2]
    
    return torch.square(base_height - target_height)


def flat_orientation_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat base orientation using projected gravity.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot entity.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)


# =============================================================================
# Facing and heading rewards
# =============================================================================

def facing_chair_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Reward for facing towards the chair.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot entity.
        chair_cfg: Configuration for the chair entity.
        
    Returns:
        Reward tensor of shape (num_envs,) in range [-1, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    # Get robot forward direction from quaternion
    root_quat = robot.data.root_quat_w
    qw, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    forward_x = 1 - 2 * (qy * qy + qz * qz)
    forward_y = 2 * (qx * qy + qw * qz)
    forward_dir = torch.stack([forward_x, forward_y], dim=-1)
    forward_dir = forward_dir / (torch.norm(forward_dir, dim=-1, keepdim=True) + 1e-6)
    
    # Direction to chair
    robot_pos = robot.data.root_pos_w[:, :2]
    chair_pos = chair.data.root_pos_w[:, :2]
    dir_to_chair = chair_pos - robot_pos
    dir_to_chair = dir_to_chair / (torch.norm(dir_to_chair, dim=-1, keepdim=True) + 1e-6)
    
    # Cosine of angle between forward direction and direction to chair
    return torch.sum(forward_dir * dir_to_chair, dim=-1)


def facing_goal_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for facing towards the goal.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot entity.
        
    Returns:
        Reward tensor of shape (num_envs,) in range [-1, 1].
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get robot forward direction from quaternion
    root_quat = robot.data.root_quat_w
    qw, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    forward_x = 1 - 2 * (qy * qy + qz * qz)
    forward_y = 2 * (qx * qy + qw * qz)
    forward_dir = torch.stack([forward_x, forward_y], dim=-1)
    forward_dir = forward_dir / (torch.norm(forward_dir, dim=-1, keepdim=True) + 1e-6)
    
    # Direction to goal
    robot_pos = robot.data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    dir_to_goal = goal_pos - robot_pos
    dir_to_goal = dir_to_goal / (torch.norm(dir_to_goal, dim=-1, keepdim=True) + 1e-6)
    
    return torch.sum(forward_dir * dir_to_goal, dim=-1)


# =============================================================================
# Contact and grasping rewards
# =============================================================================

def gripper_contact_chair_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
) -> torch.Tensor:
    """Reward for gripper making contact with the chair.
    
    Args:
        env: The RL environment instance.
        threshold: Minimum force magnitude to count as contact.
        sensor_cfg: Configuration for the contact sensor.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces
    contact_forces = contact_sensor.data.net_forces_w
    contact_mag = torch.norm(contact_forces, dim=-1)
    
    # Sum over all contact body ids
    if len(contact_mag.shape) > 1:
        contact_mag = contact_mag.sum(dim=-1)
    
    return (contact_mag > threshold).float()


class GripperContactReward(ManagerTermBase):
    """Reward for gripper making contact with chair, with history tracking."""
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.threshold = cfg.params.get("threshold", 1.0)
        self.contact_history = torch.zeros(env.num_envs, device=env.device)
        
    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.contact_history.zero_()
        else:
            self.contact_history[env_ids] = 0
            
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        # Get contact forces
        contact_forces = self.contact_sensor.data.net_forces_w
        contact_mag = torch.norm(contact_forces, dim=-1)
        
        if len(contact_mag.shape) > 1:
            contact_mag = contact_mag.sum(dim=-1)
        
        in_contact = (contact_mag > threshold).float()
        
        # Update history
        self.contact_history = torch.maximum(self.contact_history, in_contact)
        
        return in_contact


# =============================================================================
# End-effector rewards
# =============================================================================

def ee_distance_to_chair(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["arm0_link_wr1"]),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Reward for end-effector being close to chair.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for tanh kernel.
        robot_cfg: Configuration for the robot with end-effector body.
        chair_cfg: Configuration for the chair entity.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    # Get end-effector position
    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
    chair_pos = chair.data.root_pos_w
    
    distance = torch.norm(ee_pos - chair_pos, dim=-1)
    return 1.0 - torch.tanh(distance / std)


# =============================================================================
# Locomotion rewards (from Spot flat env)
# =============================================================================

def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float = 0.3,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward longer feet air and contact time for gait.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot.
        sensor_cfg: Configuration for the contact sensor.
        mode_time: Target time for air/contact phases.
        velocity_threshold: Velocity threshold for stance reward.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    
    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    
    reward = torch.where(
        body_vel > velocity_threshold,
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    std: float = 2.0
) -> torch.Tensor:
    """Reward tracking of angular velocity using exponential kernel.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot.
        std: Standard deviation for exponential kernel.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_command = env.command_manager.get_command("base_velocity")
    
    ang_vel_error = torch.abs(vel_command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    std: float = 1.0,
    ramp_at_vel: float = 1.0,
    ramp_rate: float = 0.5,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands using exponential kernel.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot.
        std: Standard deviation for exponential kernel.
        ramp_at_vel: Velocity at which to start ramping.
        ramp_rate: Ramping rate.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_command = env.command_manager.get_command("base_velocity")
    
    lin_vel_error = torch.sum(
        torch.abs(vel_command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    
    velocity_cmd_norm = torch.norm(vel_command[:, :2], dim=1)
    exp_factor = std * (1.0 + ramp_rate * torch.clamp(velocity_cmd_norm - ramp_at_vel, min=0.0))
    
    return torch.exp(-lin_vel_error / exp_factor)


# =============================================================================
# Regularization penalties
# =============================================================================

def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output.
    
    Args:
        env: The RL environment instance.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(action_diff), dim=1)


def joint_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint accelerations.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_torques_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torques.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot slipping when in contact with ground.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot feet.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Contact force threshold.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact status
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contact_mag = torch.norm(contacts, dim=-1)
    in_contact = contact_mag > threshold
    
    # Get foot velocities
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_speed = torch.norm(foot_vel, dim=-1)
    
    # Penalty: velocity when in contact
    slip_penalty = torch.sum(foot_speed * in_contact.float(), dim=1)
    
    return slip_penalty


# =============================================================================
# Success/goal rewards
# =============================================================================

def chair_at_goal_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Reward when chair reaches goal position.
    
    Args:
        env: The RL environment instance.
        threshold: Distance threshold for success.
        chair_cfg: Configuration for the chair entity.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    chair: Articulation = env.scene[chair_cfg.name]
    
    chair_pos = chair.data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(chair_pos - goal_pos, dim=-1)
    return (distance < threshold).float()


def is_alive_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive (not terminated).
    
    Args:
        env: The RL environment instance.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    return torch.ones(env.num_envs, device=env.device)


def termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for being terminated.
    
    Args:
        env: The RL environment instance.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    return env.reset_terminated.float()


# =============================================================================
# Additional rewards for curriculum phases
# =============================================================================

def joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 penalty on joint torques.
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def arm_joint_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["arm0_.*"]),
) -> torch.Tensor:
    """Penalize arm joint velocities (keep arm stable).
    
    Args:
        env: The RL environment instance.
        asset_cfg: Configuration for the robot arm joints.
        
    Returns:
        Penalty tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    arm_joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(arm_joint_vel), dim=1)


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking linear velocity command.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for exponential kernel.
        command_name: Name of the velocity command.
        asset_cfg: Configuration for the robot.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get commanded velocity
    lin_vel_error = torch.sum(torch.square(
        env.command_manager.get_command(command_name)[:, :2] - 
        asset.data.root_lin_vel_b[:, :2]
    ), dim=1)
    
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward for tracking angular velocity command.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for exponential kernel.
        command_name: Name of the velocity command.
        asset_cfg: Configuration for the robot.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get commanded angular velocity (z component)
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - 
        asset.data.root_ang_vel_b[:, 2]
    )
    
    return torch.exp(-ang_vel_error / std**2)


def contact_force_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward based on contact force magnitude.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Minimum force to count as contact.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contact_forces = contact_sensor.data.net_forces_w
    contact_mag = torch.norm(contact_forces, dim=-1)
    
    # Sum over all bodies if needed
    if len(contact_mag.shape) > 1:
        contact_mag = contact_mag.sum(dim=-1)
    
    return (contact_mag > threshold).float()


def sustained_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_duration: float = 0.5,
) -> torch.Tensor:
    """Reward for maintaining contact over time.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        min_duration: Minimum contact duration for full reward.
        
    Returns:
        Reward tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check if contact time tracking is available
    if hasattr(contact_sensor.data, "current_contact_time"):
        contact_time = contact_sensor.data.current_contact_time
        if len(contact_time.shape) > 1:
            contact_time = contact_time.max(dim=-1)[0]
        
        # Smooth reward based on contact duration
        return torch.clamp(contact_time / min_duration, 0.0, 1.0)
    else:
        # Fallback to binary contact
        contact_forces = contact_sensor.data.net_forces_w
        contact_mag = torch.norm(contact_forces, dim=-1)
        if len(contact_mag.shape) > 1:
            contact_mag = contact_mag.sum(dim=-1)
        return (contact_mag > 1.0).float()
