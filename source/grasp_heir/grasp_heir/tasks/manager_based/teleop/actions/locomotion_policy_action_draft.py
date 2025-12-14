# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

@configclass
class LocomotionPolicyActionCfg:
    """Configuration for the locomotion policy action term."""
    class_type: type = NotImplemented  # To be set to LocomotionPolicyAction
    asset_name: str = MISSING
    policy_path: str = MISSING # Path to the trained policy (.pt file)
    joint_names: list[str] = MISSING # Joint names to control (legs)
    
    # Policy kwargs
    num_actions: int = 12 # 12 joints for Spot legs
    action_scale: float = 0.5



class LocomotionPolicyAction(ActionTerm):
    """Action term that uses a trained policy to control joints."""

    cfg: LocomotionPolicyActionCfg
    _policy: torch.nn.Module | None = None

    def __init__(self, cfg: LocomotionPolicyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._load_policy()
        
        # Prepare buffers
        self.joint_ids, _ = self._asset.find_joints(self.cfg.joint_names)
        self.raw_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.low_level_actions = torch.zeros(self.num_envs, len(self.joint_ids), device=self.device)

    def _load_policy(self):
        # Load the policy
        # Placeholder for policy loading mechanism
        # In practice, usually load torch jit or standard model
        # We'll assume it's a jit script or we use RSL-RL utils if available
        # For this implementation plan, we'll try to load a .pt file directly or use a dummy if not found
        try:
           # self._policy = torch.jit.load(self.cfg.policy_path)
           # self._policy.to(self.device)
           pass
        except Exception as e:
            print(f"[WARN] Failed to load policy from {self.cfg.policy_path}: {e}")
    
    def process_actions(self, actions: torch.Tensor):
        # actions here are the velocity commands from the user/agent (vx, vy, w)
        # We need to construct the observation for the policy 
        # This is the hard part: The policy expects a specific observation vector.
        # Constructing it inside the ActionTerm is difficult because we don't have easy access to the full Obs definitions of the training env.
        #
        # A Better Approach might be:
        # The Environment itself should compute the Policy Actions, OR
        # This ActionTerm needs access to the necessary data from `env` to build observations.
        #
        # Given "The environment step will internally query the trained locomotion policy",
        # implementing this in the Env class (`GraspHeirTeleopEnv`) `step` method might be easier than in an ActionTerm.
        # The ActionTerm sees "actions", but doesn't easily see "observations" (though it has access to env).
        
        # Let's pivot to implementing `GraspHeirTeleopEnv` class instead of a complex ActionTerm.
        pass

    def apply_actions(self):
        pass
