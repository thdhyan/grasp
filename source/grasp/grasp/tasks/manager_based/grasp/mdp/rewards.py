# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
import re

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

#Adding reward for robot tilt This is negative l2 for roll and pitch
def robot_tilt(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    
    robot = env.scene[asset_cfg.name]
    
    # Filter bodies
    if asset_cfg.body_names is not None:
        # Resolve indices using the provided body names/regex
        valid_indices, _ = robot.find_bodies(asset_cfg.body_names)
    else:
        # Default: filter out feet if no specific bodies requested
        body_names = robot.data.body_names
        valid_indices = [
            i for i, name in enumerate(body_names) 
            if not re.search(".*_foot", name)
        ]
    
    if not valid_indices:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Get orientation (w, x, y, z)
    quat = robot.data.body_rot_w[:, valid_indices, :]
    roll, pitch, _ = euler_xyz_from_quat(quat)
    
    # Penalize roll and pitch
    return torch.sum(torch.square(roll) + torch.square(pitch), dim=1)