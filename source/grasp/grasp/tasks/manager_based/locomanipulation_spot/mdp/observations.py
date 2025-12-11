# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for Spot locomanipulation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Base observations
# =============================================================================

def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base linear velocity in base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base angular velocity in base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity vector projected into the robot's base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.projected_gravity_b


def robot_root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Robot root position in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w


def robot_root_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Robot root quaternion in world frame (w, x, y, z)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_quat_w


# =============================================================================
# Joint observations
# =============================================================================

def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions relative to default positions."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos - robot.data.default_joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities relative to default velocities (usually zero)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel - robot.data.default_joint_vel


def leg_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Leg joint positions (12 joints: 4 legs x 3 joints each)."""
    robot: Articulation = env.scene[asset_cfg.name]
    # Get indices for leg joints (hx, hy, kn for each leg)
    leg_joint_indices = []
    for name in ["fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
                 "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn"]:
        for i, joint_name in enumerate(robot.joint_names):
            if name in joint_name:
                leg_joint_indices.append(i)
                break
    if len(leg_joint_indices) == 12:
        return robot.data.joint_pos[:, leg_joint_indices]
    # Fallback: return first 12 joints
    return robot.data.joint_pos[:, :12]


def leg_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Leg joint velocities (12 joints)."""
    robot: Articulation = env.scene[asset_cfg.name]
    leg_joint_indices = []
    for name in ["fl_hx", "fl_hy", "fl_kn", "fr_hx", "fr_hy", "fr_kn",
                 "hl_hx", "hl_hy", "hl_kn", "hr_hx", "hr_hy", "hr_kn"]:
        for i, joint_name in enumerate(robot.joint_names):
            if name in joint_name:
                leg_joint_indices.append(i)
                break
    if len(leg_joint_indices) == 12:
        return robot.data.joint_vel[:, leg_joint_indices]
    return robot.data.joint_vel[:, :12]


def arm_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Arm joint positions (7 joints: shoulder, elbow, wrist, gripper)."""
    robot: Articulation = env.scene[asset_cfg.name]
    arm_joint_indices = []
    arm_patterns = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", 
                    "arm0_wr0", "arm0_wr1", "arm0_f1x"]
    for pattern in arm_patterns:
        for i, joint_name in enumerate(robot.joint_names):
            if pattern in joint_name:
                arm_joint_indices.append(i)
                break
    if len(arm_joint_indices) == 7:
        return robot.data.joint_pos[:, arm_joint_indices]
    # Fallback: return last 7 joints
    return robot.data.joint_pos[:, -7:]


def arm_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Arm joint velocities (7 joints)."""
    robot: Articulation = env.scene[asset_cfg.name]
    arm_joint_indices = []
    arm_patterns = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", 
                    "arm0_wr0", "arm0_wr1", "arm0_f1x"]
    for pattern in arm_patterns:
        for i, joint_name in enumerate(robot.joint_names):
            if pattern in joint_name:
                arm_joint_indices.append(i)
                break
    if len(arm_joint_indices) == 7:
        return robot.data.joint_vel[:, arm_joint_indices]
    return robot.data.joint_vel[:, -7:]


# =============================================================================
# End-effector observations
# =============================================================================

def ee_position_b(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "arm0_link_fngr"
) -> torch.Tensor:
    """End-effector position in base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get body index
    body_idx = robot.find_bodies(body_name)[0][0]
    
    # Get EE position in world frame
    ee_pos_w = robot.data.body_pos_w[:, body_idx]
    
    # Get base position and orientation
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    
    # Transform to base frame
    ee_pos_b = quat_rotate_inverse(base_quat_w, ee_pos_w - base_pos_w)
    
    return ee_pos_b


def ee_position_w(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "arm0_link_fngr"
) -> torch.Tensor:
    """End-effector position in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(body_name)[0][0]
    return robot.data.body_pos_w[:, body_idx]


def ee_velocity_b(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "arm0_link_fngr"
) -> torch.Tensor:
    """End-effector linear velocity in base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(body_name)[0][0]
    
    # Get EE velocity in world frame
    ee_vel_w = robot.data.body_lin_vel_w[:, body_idx]
    
    # Transform to base frame
    base_quat_w = robot.data.root_quat_w
    ee_vel_b = quat_rotate_inverse(base_quat_w, ee_vel_w)
    
    return ee_vel_b


def ee_orientation_w(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "arm0_link_fngr"
) -> torch.Tensor:
    """End-effector orientation (quaternion) in world frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_idx = robot.find_bodies(body_name)[0][0]
    return robot.data.body_quat_w[:, body_idx]


# =============================================================================
# Object observations
# =============================================================================

def object_pos_b(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object position relative to robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get positions in world frame
    obj_pos_w = obj.data.root_pos_w
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    
    # Transform to base frame
    obj_pos_b = quat_rotate_inverse(base_quat_w, obj_pos_w - base_pos_w)
    
    return obj_pos_b


def object_pos_w(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object position in world frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w


def object_vel_b(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Object velocity in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    obj_vel_w = obj.data.root_lin_vel_w
    base_quat_w = robot.data.root_quat_w
    
    obj_vel_b = quat_rotate_inverse(base_quat_w, obj_vel_w)
    
    return obj_vel_b


def object_to_ee_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_name: str = "arm0_link_fngr"
) -> torch.Tensor:
    """Vector from end-effector to object in base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Get body index for EE
    body_idx = robot.find_bodies(body_name)[0][0]
    
    # Get positions in world frame
    ee_pos_w = robot.data.body_pos_w[:, body_idx]
    obj_pos_w = obj.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    
    # Compute vector and transform to base frame
    vec_w = obj_pos_w - ee_pos_w
    vec_b = quat_rotate_inverse(base_quat_w, vec_w)
    
    return vec_b


def object_to_ee_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_name: str = "arm0_link_fngr"
) -> torch.Tensor:
    """Distance from end-effector to object."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    
    body_idx = robot.find_bodies(body_name)[0][0]
    
    ee_pos_w = robot.data.body_pos_w[:, body_idx]
    obj_pos_w = obj.data.root_pos_w
    
    distance = torch.norm(obj_pos_w - ee_pos_w, dim=-1, keepdim=True)
    
    return distance


# =============================================================================
# Goal observations
# =============================================================================

def goal_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Goal position relative to robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get goal positions from environment
    if hasattr(env, 'goal_positions'):
        goal_pos_w = env.goal_positions
    elif hasattr(env, 'command_manager') and 'goal_position' in env.command_manager._terms:
        goal_pos_w = env.command_manager.get_command('goal_position')
    else:
        # Fallback: return zeros
        return torch.zeros(env.num_envs, 3, device=env.device)
    
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    
    goal_pos_b = quat_rotate_inverse(base_quat_w, goal_pos_w - base_pos_w)
    
    return goal_pos_b


def goal_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Goal position in world frame."""
    if hasattr(env, 'goal_positions'):
        return env.goal_positions
    elif hasattr(env, 'command_manager') and 'goal_position' in env.command_manager._terms:
        return env.command_manager.get_command('goal_position')
    else:
        return torch.zeros(env.num_envs, 3, device=env.device)


def object_to_goal_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Distance from object to goal position."""
    obj: RigidObject = env.scene[object_cfg.name]
    
    obj_pos_w = obj.data.root_pos_w
    
    if hasattr(env, 'goal_positions'):
        goal_pos_w = env.goal_positions
    elif hasattr(env, 'command_manager') and 'goal_position' in env.command_manager._terms:
        goal_pos_w = env.command_manager.get_command('goal_position')
    else:
        goal_pos_w = torch.zeros_like(obj_pos_w)
    
    distance = torch.norm(obj_pos_w - goal_pos_w, dim=-1, keepdim=True)
    
    return distance


# =============================================================================
# Contact observations
# =============================================================================

def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0
) -> torch.Tensor:
    """Binary contact indicators for feet (1 if in contact, 0 otherwise)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0]  # Current timestep
    
    # Compute magnitude of contact forces
    force_magnitude = torch.norm(net_forces, dim=-1)
    
    # Binary contact indicator
    in_contact = (force_magnitude > threshold).float()
    
    return in_contact


def gripper_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact")
) -> torch.Tensor:
    """Gripper contact force magnitude."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0]
    
    force_magnitude = torch.norm(net_forces, dim=-1, keepdim=True)
    
    return force_magnitude


def gripper_in_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
    threshold: float = 1.0
) -> torch.Tensor:
    """Binary indicator if gripper is in contact with object."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, 0]
    
    force_magnitude = torch.norm(net_forces, dim=-1, keepdim=True)
    in_contact = (force_magnitude > threshold).float()
    
    return in_contact


# =============================================================================
# Command observations
# =============================================================================

def velocity_commands(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity"
) -> torch.Tensor:
    """Get velocity commands from command manager."""
    return env.command_manager.get_command(command_name)


def goal_commands(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_position"
) -> torch.Tensor:
    """Get goal position commands from command manager."""
    return env.command_manager.get_command(command_name)


# =============================================================================
# Last action observation
# =============================================================================

def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Last action taken by the agent."""
    return env.action_manager.action
