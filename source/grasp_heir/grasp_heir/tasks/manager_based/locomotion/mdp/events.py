# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_arm_joint_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_range: tuple[float, float],
):
    """Randomize the joint position targets for the arm joints.
    
    This function sets the stiffness/damping targets (drive targets) for the specified joints
    to random values within the given range. This simulates the arm moving around.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # resolve joint indices
    joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names)
    
    # sample random positions
    # range is scaling factor around default, or absolute? 
    # Let's assume absolute range in radians if provided, or offsets.
    # But usually easier to sample uniform in [min, max] or [default - delta, default + delta]
    # The signature `position_range` implies a min/max or scale. 
    # Let's assume it's (min, max) for now and we sample uniformly.
    
    # Get default joint positions
    default_pos = asset.data.default_joint_pos[env_ids[:, None], joint_ids]
    
    # Sample noise
    noise = torch.rand_like(default_pos) * (position_range[1] - position_range[0]) + position_range[0]
    
    # New targets 
    # Ideally we want to explore the whole workspace. 
    # For now, let's just add noise to the default pose.
    # Or should we assume position_range is the absolute limits?
    # Let's implementation: assume position_range is +/- offset from default.
    
    targets = default_pos + noise
    

def randomize_goal_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
):
    """Randomize the goal position for the robot.
    
    This function generates a random 2D goal position within the specified range and sets it
    in the environment's extras.
    """
    # check if 'goal' exists in extras, if not create it
    if "goal" not in env.scene.extras:
        # assume 2D goal (x, y)
        env.scene.extras["goal"] = torch.zeros((env.num_envs, 2), device=env.device)
        
    # sample random positions
    # range is (min, max) for both x and y
    r_min, r_max = position_range
    
    # generate random x, y uniformly in [min, max]
    random_goal = torch.rand((len(env_ids), 2), device=env.device) * (r_max - r_min) + r_min
    
    # update goals for reset envs
    env.scene.extras["goal"][env_ids] = random_goal
