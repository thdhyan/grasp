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


def base_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Terminate if the robot's body touches the ground/objects.
    
    Returns True if contact force magnitude > threshold.
    """
    # Add initial grace period (no termination in first 5 seconds)
    if hasattr(env, "episode_step_count") and hasattr(env, "max_episode_length"):
        # Compute elapsed time in seconds
        elapsed_time = env.episode_step_count * env.sim.dt
        if elapsed_time < 5.0:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Only terminate if the body contact sensor (not feet) detects contact
    if sensor_cfg.name != "contact_forces_body":
        # If not the body sensor, never terminate
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get forces (num_envs, num_bodies, 3) or (num_envs, 3)
    if sensor_cfg.body_ids is not None:
        forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    else:
        forces = sensor.data.net_forces_w

    # Check contact
    contact = torch.norm(forces, dim=-1) > threshold

    # If any body part in the sensor is in contact, terminate
    if forces.dim() == 3:
        return torch.any(contact, dim=1)
    return contact


def feet_air_time_termination(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    time_threshold: float = 5.0,
) -> torch.Tensor:
    """Terminate if feet have been in the air for too long.
    
    Requires ContactSensor to have track_air_time=True.
    """
    # Add initial grace period (no termination in first 5 seconds)
    if hasattr(env, "episode_step_count") and hasattr(env, "max_episode_length"):
        elapsed_time = env.episode_step_count * env.sim.dt
        if elapsed_time < 5.0:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    all_feet_air_times = []
    for sensor_cfg in sensor_cfgs:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        air_time = sensor.data.current_air_time
        all_feet_air_times.append(air_time)

    stacked_air_times = torch.cat(all_feet_air_times, dim=1)
    min_air_time, _ = torch.min(stacked_air_times, dim=1)
    return min_air_time > time_threshold
