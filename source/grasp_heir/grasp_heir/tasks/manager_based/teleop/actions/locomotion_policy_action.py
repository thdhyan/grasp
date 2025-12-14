# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import os
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

@configclass
class LocomotionPolicyActionCfg:
    """Configuration for the locomotion policy action term."""
    class_type: type = NotImplemented 
    asset_name: str = "robot"
    policy_path: str = "" # Path to the trained policy (.pt file)
    joint_names: list[str] = [".*_hx", ".*_hy", ".*_kn"] # Leg joints
    
    # Observation params (must match training)
    # This is brittle. Ideally policy is self-contained or we share obs config.
    # For now, we assume standard RSL-RL obs:
    # 0-2: lin vel
    # 3-5: ang vel
    # 6-8: gravity
    # 9-11: commands
    # 12-23: joint pos
    # 24-35: joint vel
    # 36-47: last actions
    scale_action: float = 0.5 # Scale of policy output if raw

class LocomotionPolicyAction(ActionTerm):
    """Action term that uses a trained policy to control joints."""

    cfg: LocomotionPolicyActionCfg
    _policy: torch.nn.Module | None = None
    _asset: Articulation

    def __init__(self, cfg: LocomotionPolicyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Resolve joints
        self.joint_ids, self.joint_names = self._asset.find_joints(self.cfg.joint_names)
        self.num_joint_actions = len(self.joint_ids)
        
        # Buffers
        self.last_actions = torch.zeros(self.num_envs, self.num_joint_actions, device=self.device)
        self.obs_buf = torch.zeros(self.num_envs, 48, device=self.device) # Fixed size for Spot RSL-RL
        
        # Load policy
        if self.cfg.policy_path and os.path.exists(self.cfg.policy_path):
            try:
                # Load JIT
                self._policy = torch.jit.load(self.cfg.policy_path).to(self.device).eval()
            except Exception as e:
                print(f"[ERROR] Failed to load policy: {e}")
        else:
             print(f"[WARN] Policy path invalid: {self.cfg.policy_path}")

    def process_actions(self, actions: torch.Tensor):
        # Actions received are [vx, vy, wz] (3 dims)
        # We need to compute leg actions
        
        if self._policy is None:
            # Fallback: zero torques/actions
            self.last_actions[:] = 0.0
            return
        
        # 1. Update Observations
        # Uses explicit knowledge of what the policy expects.
        # (lin_vel, ang_vel, grav, cmd, q_rel, v_rel, last_act)
        
        # Base lin vel (in base frame)
        base_lin_vel = self._asset.data.root_lin_vel_b
        # Base ang vel
        base_ang_vel = self._asset.data.root_ang_vel_b
        # Gravity (projected)
        grav = self._asset.data.projected_gravity_b
        # Commands (from input)
        # Verify scaling? commands in training are [-1, 1] usually scaled by range. 
        # But `actions` here are the commands. Let's assume they are already scaled or raw commands.
        cmds = actions
        # Joint pos/vel (relative/normalized?)
        # Training used mdp.joint_pos_rel -> (pos - default)
        q = self._asset.data.joint_pos[:, self.joint_ids]
        q_default = self._asset.data.default_joint_pos[:, self.joint_ids]
        q_rel = q - q_default
        
        dq = self._asset.data.joint_vel[:, self.joint_ids]
        
        # Assemble obs
        start = 0
        self.obs_buf[:, start:start+3] = base_lin_vel; start += 3
        self.obs_buf[:, start:start+3] = base_ang_vel; start += 3
        self.obs_buf[:, start:start+3] = grav; start += 3
        self.obs_buf[:, start:start+3] = cmds; start += 3 # 3 commands? Policy trained with 3?
        # Wait, policy might have 12 joint pos.
        self.obs_buf[:, start:start+12] = q_rel; start += 12
        self.obs_buf[:, start:start+12] = dq; start += 12
        self.obs_buf[:, start:start+12] = self.last_actions; start += 12
        
        # 2. Run Policy
        with torch.no_grad():
            # Policy usually outputs actions
            # If using RslRlOnPolicyRunner policy, it might return actions directly.
            out = self._policy(self.obs_buf)
            # out shape? [env, 12]
        
        self.last_actions[:] = out

    def apply_actions(self):
        # Apply self.last_actions to joints
        # actions from policy are usually position targets diff or scaled stuff.
        # "JointEffortActionCfg" was used in training? No.
        # Training env used: 
        # mdp.JointEffortActionCfg(..., scale=100.0) ??
        # Wait, check `spot_locomotion_env_cfg.py`
        # It used `mdp.JointEffortActionCfg`.
        # So policy outputs EFFORTS (torques) directly? 
        # If so, scale=100.0 means out * 100.0? 
        # Or ActionTerm scales it? 
        # mdp.JointEffortActionCfg applies `action * scale`.
        
        # My ActionTerm should replicate this.
        # Assuming policy outputs raw network output.
        
        # Apply efforts
        targets = self.last_actions * 100.0 # Match training scale
        self._asset.set_joint_effort_target(targets, joint_ids=self.joint_ids)

LocomotionPolicyActionCfg.class_type = LocomotionPolicyAction
