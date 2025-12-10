# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the locomotion spot navigation and manipulation task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Distance observations
# =============================================================================

def distance_robot_to_chair(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Observation: distance between robot and chair (2D).
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        chair_cfg: Configuration for the chair.
        
    Returns:
        Distance tensor of shape (num_envs, 1).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    robot_pos = robot.data.root_pos_w[:, :2]
    chair_pos = chair.data.root_pos_w[:, :2]
    
    distance = torch.norm(robot_pos - chair_pos, dim=-1, keepdim=True)
    return distance


def distance_robot_to_goal(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: distance between robot and goal (2D).
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Distance tensor of shape (num_envs, 1).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    robot_pos = robot.data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(robot_pos - goal_pos, dim=-1, keepdim=True)
    return distance


def distance_chair_to_goal(
    env: ManagerBasedRLEnv,
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Observation: distance between chair and goal (2D).
    
    Args:
        env: The RL environment instance.
        chair_cfg: Configuration for the chair.
        
    Returns:
        Distance tensor of shape (num_envs, 1).
    """
    chair: Articulation = env.scene[chair_cfg.name]
    
    chair_pos = chair.data.root_pos_w[:, :2]
    goal_pos = env.goal_positions[:, :2]
    
    distance = torch.norm(chair_pos - goal_pos, dim=-1, keepdim=True)
    return distance


def direction_to_chair(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Observation: direction vector from robot to chair in robot frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        chair_cfg: Configuration for the chair.
        
    Returns:
        Direction tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    # Direction in world frame
    dir_w = chair.data.root_pos_w - robot.data.root_pos_w
    
    # Transform to robot body frame using quaternion
    quat = robot.data.root_quat_w
    dir_b = _quat_rotate_inverse(quat, dir_w)
    
    return dir_b


def direction_to_goal(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: direction vector from robot to goal in robot frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Direction tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Direction in world frame
    dir_w = env.goal_positions - robot.data.root_pos_w
    
    # Transform to robot body frame using quaternion
    quat = robot.data.root_quat_w
    dir_b = _quat_rotate_inverse(quat, dir_w)
    
    return dir_b


def goal_position_relative(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: goal position relative to robot in robot frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Position tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get relative position in world frame
    rel_pos_w = env.goal_positions - robot.data.root_pos_w
    
    # Transform to body frame
    quat = robot.data.root_quat_w
    rel_pos_b = _quat_rotate_inverse(quat, rel_pos_w)
    
    return rel_pos_b


def chair_position_relative(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Observation: chair position relative to robot in robot frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        chair_cfg: Configuration for the chair.
        
    Returns:
        Position tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    # Get relative position in world frame
    rel_pos_w = chair.data.root_pos_w - robot.data.root_pos_w
    
    # Transform to body frame
    quat = robot.data.root_quat_w
    rel_pos_b = _quat_rotate_inverse(quat, rel_pos_w)
    
    return rel_pos_b


# =============================================================================
# Contact observations
# =============================================================================

def contact_binary_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Observation: binary contact state from contact sensor.
    
    Returns a binary tensor indicating whether each body in the contact sensor
    is in contact (force magnitude above threshold).
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Force threshold for contact detection.
        
    Returns:
        Contact state tensor of shape (num_envs, num_bodies).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get net forces: shape (num_envs, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w
    # Compute magnitude: shape (num_envs, num_bodies)
    contact_mag = torch.norm(contact_forces, dim=-1)
    # Binary contact: shape (num_envs, num_bodies)
    in_contact = (contact_mag > threshold).float()
    
    return in_contact


def gripper_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Observation: binary contact state of gripper.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        threshold: Force threshold for contact.
        
    Returns:
        Contact state tensor of shape (num_envs, 1).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contact_forces = contact_sensor.data.net_forces_w
    contact_mag = torch.norm(contact_forces, dim=-1)
    
    if len(contact_mag.shape) > 1:
        contact_mag = contact_mag.sum(dim=-1)
    
    in_contact = (contact_mag > threshold).float().unsqueeze(-1)
    return in_contact


def gripper_contact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
) -> torch.Tensor:
    """Observation: contact force magnitude at gripper.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the contact sensor.
        
    Returns:
        Contact force tensor of shape (num_envs, 1).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    contact_forces = contact_sensor.data.net_forces_w
    contact_mag = torch.norm(contact_forces, dim=-1)
    
    if len(contact_mag.shape) > 1:
        contact_mag = contact_mag.sum(dim=-1)
    
    return contact_mag.unsqueeze(-1)


# =============================================================================
# Camera observations
# =============================================================================

def camera_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
) -> torch.Tensor:
    """Observation: RGB camera images.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the camera sensor.
        
    Returns:
        RGB tensor of shape (num_envs, H, W, 3).
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output["rgb"]


def camera_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
) -> torch.Tensor:
    """Observation: depth camera images.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the camera sensor.
        
    Returns:
        Depth tensor of shape (num_envs, H, W, 1).
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output["depth"]


def camera_instance_segmentation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
) -> torch.Tensor:
    """Observation: instance segmentation from camera.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the camera sensor.
        
    Returns:
        Instance segmentation tensor of shape (num_envs, H, W, 1) or (num_envs, H, W, 4) if colorized.
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output["instance_id_segmentation_fast"]


def camera_depth_wrist(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
) -> torch.Tensor:
    """Observation: depth from wrist camera.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the wrist camera sensor.
        
    Returns:
        Depth tensor of shape (num_envs, H, W, 1).
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output["depth"]


def camera_segmentation_wrist(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
) -> torch.Tensor:
    """Observation: instance segmentation from wrist camera.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the wrist camera sensor.
        
    Returns:
        Instance segmentation tensor.
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output["instance_id_segmentation_fast"]


def object_depth_from_segmentation(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
    target_object_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Observation: estimated depth to segmented object (chair).
    
    Uses instance segmentation mask to extract the depth values of the target object
    and returns the mean depth to that object.
    
    Args:
        env: The RL environment instance.
        camera_cfg: Configuration for the camera sensor with depth and segmentation.
        target_object_cfg: Configuration for the target object to segment.
        
    Returns:
        Object depth tensor of shape (num_envs, 1).
    """
    camera: TiledCamera = env.scene.sensors[camera_cfg.name]
    
    # Get depth and segmentation data
    depth = camera.data.output["depth"]  # (num_envs, H, W, 1)
    segmentation = camera.data.output["instance_id_segmentation_fast"]  # (num_envs, H, W, 1) or (num_envs, H, W, 4)
    
    # Get segmentation info for object ID mapping
    seg_info = camera.data.info.get("instance_id_segmentation_fast", {})
    
    num_envs = depth.shape[0]
    device = depth.device
    
    # Initialize output
    object_depth = torch.zeros(num_envs, 1, device=device)
    
    # For each environment, compute mean depth of segmented object
    for i in range(num_envs):
        # Get depth for this env
        env_depth = depth[i, :, :, 0]  # (H, W)
        env_seg = segmentation[i]  # (H, W, 1) or (H, W, 4)
        
        if env_seg.shape[-1] == 1:
            # Non-colorized: directly use instance IDs
            seg_ids = env_seg[:, :, 0]
        else:
            # Colorized: use R channel as rough proxy (assumes unique colors per instance)
            seg_ids = env_seg[:, :, 0]
        
        # Find non-zero segmented regions (object pixels vs background)
        # In typical setup, background is 0, objects have non-zero IDs
        object_mask = seg_ids > 0
        
        if object_mask.any():
            # Extract depth values for object pixels
            object_depths = env_depth[object_mask]
            # Filter out invalid depths (inf, nan)
            valid_depths = object_depths[torch.isfinite(object_depths)]
            if valid_depths.numel() > 0:
                object_depth[i, 0] = valid_depths.mean()
            else:
                object_depth[i, 0] = 10.0  # Default far distance
        else:
            object_depth[i, 0] = 10.0  # No object detected, default far distance
    
    return object_depth


def front_camera_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
) -> torch.Tensor:
    """Observation: depth from front camera.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the front camera sensor.
        
    Returns:
        Depth tensor of shape (num_envs, H, W, 1).
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output["depth"]


def front_camera_segmentation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
) -> torch.Tensor:
    """Observation: instance segmentation from front camera.
    
    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the front camera sensor.
        
    Returns:
        Instance segmentation tensor.
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return camera.data.output["instance_id_segmentation_fast"]


# =============================================================================
# Robot state observations
# =============================================================================

def base_height(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: robot base height.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Height tensor of shape (num_envs, 1).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_pos_w[:, 2:3]


def end_effector_position(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["arm0_link_wr1"]),
) -> torch.Tensor:
    """Observation: end-effector position in world frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot with EE body.
        
    Returns:
        Position tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]


def end_effector_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["arm0_link_wr1"]),
) -> torch.Tensor:
    """Observation: end-effector velocity in world frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot with EE body.
        
    Returns:
        Velocity tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.body_lin_vel_w[:, robot_cfg.body_ids[0], :]


def chair_velocity(
    env: ManagerBasedRLEnv,
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Observation: chair linear velocity.
    
    Args:
        env: The RL environment instance.
        chair_cfg: Configuration for the chair.
        
    Returns:
        Velocity tensor of shape (num_envs, 3).
    """
    chair: Articulation = env.scene[chair_cfg.name]
    return chair.data.root_lin_vel_w


# =============================================================================
# Arm-relative observations
# =============================================================================

def direction_to_chair_from_ee(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
    ee_body_name: str = "arm0_link_fngr",
) -> torch.Tensor:
    """Observation: direction vector from end-effector to chair in robot frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        chair_cfg: Configuration for the chair.
        ee_body_name: Name of the end-effector body.
        
    Returns:
        Direction tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    # Get end-effector position
    ee_body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
    
    # Direction from EE to chair in world frame
    dir_w = chair.data.root_pos_w - ee_pos_w
    
    # Transform to robot body frame
    quat = robot.data.root_quat_w
    dir_b = _quat_rotate_inverse(quat, dir_w)
    
    return dir_b


def distance_ee_to_chair(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
    ee_body_name: str = "arm0_link_fngr",
) -> torch.Tensor:
    """Observation: distance from end-effector to chair.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        chair_cfg: Configuration for the chair.
        ee_body_name: Name of the end-effector body.
        
    Returns:
        Distance tensor of shape (num_envs, 1).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chair: Articulation = env.scene[chair_cfg.name]
    
    # Get end-effector position
    ee_body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]
    
    # Distance
    distance = torch.norm(chair.data.root_pos_w - ee_pos_w, dim=-1, keepdim=True)
    
    return distance


def chair_position_global(
    env: ManagerBasedRLEnv,
    chair_cfg: SceneEntityCfg = SceneEntityCfg("chair"),
) -> torch.Tensor:
    """Observation: chair position in world frame.
    
    Args:
        env: The RL environment instance.
        chair_cfg: Configuration for the chair.
        
    Returns:
        Position tensor of shape (num_envs, 3).
    """
    chair: Articulation = env.scene[chair_cfg.name]
    return chair.data.root_pos_w


def goal_position_global(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Observation: goal position in world frame.
    
    Args:
        env: The RL environment instance.
        
    Returns:
        Position tensor of shape (num_envs, 3).
    """
    return env.goal_positions


def robot_position_global(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: robot position in world frame.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Position tensor of shape (num_envs, 3).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_pos_w


def robot_orientation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: robot orientation as quaternion.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Quaternion tensor of shape (num_envs, 4).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_quat_w


def arm_joint_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: arm joint positions only.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Joint position tensor of shape (num_envs, num_arm_joints).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Arm joints are typically indices 12-18 (sh0, sh1, el0, el1, wr0, wr1, f1x)
    arm_joint_names = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1", "arm0_f1x"]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    return robot.data.joint_pos[:, arm_joint_ids]


def leg_joint_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: leg joint positions only.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Joint position tensor of shape (num_envs, num_leg_joints).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Leg joints pattern: *_hx, *_hy, *_kn
    leg_joint_patterns = [".*_hx", ".*_hy", ".*_kn"]
    all_leg_ids = []
    for pattern in leg_joint_patterns:
        ids, _ = robot.find_joints(pattern)
        all_leg_ids.extend(ids)
    return robot.data.joint_pos[:, sorted(all_leg_ids)]


def arm_joint_velocities(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: arm joint velocities only.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Joint velocity tensor of shape (num_envs, num_arm_joints).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    arm_joint_names = ["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1", "arm0_f1x"]
    arm_joint_ids = robot.find_joints(arm_joint_names)[0]
    return robot.data.joint_vel[:, arm_joint_ids]


def leg_joint_velocities(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Observation: leg joint velocities only.
    
    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot.
        
    Returns:
        Joint velocity tensor of shape (num_envs, num_leg_joints).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    leg_joint_patterns = [".*_hx", ".*_hy", ".*_kn"]
    all_leg_ids = []
    for pattern in leg_joint_patterns:
        ids, _ = robot.find_joints(pattern)
        all_leg_ids.extend(ids)
    return robot.data.joint_vel[:, sorted(all_leg_ids)]


# =============================================================================
# Utility functions
# =============================================================================

def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.
    
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
    
    return a - b + c
