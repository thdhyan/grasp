# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def feet_contact_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward having feet in contact with the ground/objects.
    
    Returns the number of feet in contact.
    """
    # Access sensor
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Get forces (num_envs, num_bodies, 3)
    # We filter by body_ids in sensor_cfg (which should match feet)
    # Note: sensor_cfg.body_ids are indices into the sensor's body list IF the sensor covers multiple bodies.
    # contact_sensor.data.net_forces_w has shape matching the sensor's configured bodies.
    # If sensor covers the whole robot, logic needs care.
    # But SceneEntityCfg usually resolves indices relative to the asset?
    # No, SceneEntityCfg resolves indices into the asset. 
    # ContactSensor tracks specific bodies. 
    # If ContactSensor tracks ".*", and we invoke SceneEntityCfg on "contact_forces", 
    # we need to map the body indices from Asset to Sensor indices?
    # Actually `sensor.data.net_forces_w` is (num_envs, num_sensor_bodies, 3).
    # `sensor_cfg.body_ids` refers to bodies in the ASSET (Robot).
    # This mismatch is tricky if the sensor doesn't track exactly the same bodies in the same order.
    # HOWEVER, `undesired_contacts` works this way in default MDP.
    # It assumes `sensor_cfg` points to the Sensor, and `body_names` selects which bodies within that sensor?
    # No, `undesired_contacts` implementation:
    #   sensor = env.scene.sensors[sensor_cfg.name]
    #   return torch.any(torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > threshold, dim=-1)
    # This implies `sensor_cfg.body_ids` MUST be indices into the sensor's data.
    # But `SceneEntityCfg("contact_forces", body_names=...)` resolves ids relative to `contact_forces` entity (the sensor).
    # The sensor entity has `.body_names`? 
    # Yes, ContactSensor has `body_names`.
    # So `SceneEntityCfg` works correctly if `body_names` match the sensor's tracked bodies.
    
    # Use all sensor bodies if body_ids not explicitly set
    if sensor_cfg.body_ids is not None:
        forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    else:
        forces = sensor.data.net_forces_w  # Use all bodies in the sensor
    
    contact = torch.norm(forces, dim=-1) > threshold
    # Handle both single body (dim=1) and multi-body (dim=2) cases
    if forces.dim() == 2:
        return contact.float()  # Single body per env
    return torch.sum(contact.float(), dim=1)


def air_time_variance_penalty(
    env: ManagerBasedRLEnv, 
    sensor_cfgs: list[SceneEntityCfg],
    threshold: float = 1.0
) -> torch.Tensor:
    """Penalize unequal air time across feet - encourages symmetric gait.
    
    Computes variance of air time across all feet sensors and returns negative reward.
    """
    air_times = []
    for sensor_cfg in sensor_cfgs:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # Get contact state: True if in contact
        forces = sensor.data.net_forces_w
        in_contact = torch.norm(forces, dim=-1) > threshold  # (num_envs,) or (num_envs, num_bodies)
        
        # Air time = not in contact
        # We track if feet are in air (0) or contact (1)
        # For variance, we want the ratio of air vs contact over recent history
        # Simple approach: use current state variance
        if in_contact.dim() == 1:
            air_times.append((~in_contact).float())
        else:
            air_times.append((~in_contact).float().mean(dim=1))
    
    # Stack air times across feet: (num_envs, num_feet)
    stacked = torch.stack(air_times, dim=1)
    # Compute variance across feet dimension
    variance = torch.var(stacked, dim=1)
    return variance


def velocity_tracking_slow_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_velocity_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize being too slow when a non-zero velocity command is given.
    
    If command velocity magnitude is > threshold but actual velocity is < threshold,
    apply penalty proportional to the difference.
    """
    # Get commanded velocity
    command = env.command_manager.get_command(command_name)
    # command shape: (num_envs, 3) for [vx, vy, wz]
    cmd_lin_vel = command[:, :2]  # Linear velocity x, y
    cmd_mag = torch.norm(cmd_lin_vel, dim=1)
    
    # Get actual velocity
    asset = env.scene[asset_cfg.name]
    actual_lin_vel = asset.data.root_lin_vel_b[:, :2]  # Base frame x, y
    actual_mag = torch.norm(actual_lin_vel, dim=1)
    
    # Penalty when: command is significant but actual is too slow
    should_move = cmd_mag > min_velocity_threshold
    too_slow = actual_mag < min_velocity_threshold
    
    # Penalty = difference between commanded and actual when too slow
    penalty = torch.where(
        should_move & too_slow,
        cmd_mag - actual_mag,
        torch.zeros_like(cmd_mag)
    )
    return penalty

