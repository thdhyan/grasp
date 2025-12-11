# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action configurations for Spot locomanipulation tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# =============================================================================
# Joint Position Action
# =============================================================================

class JointPositionAction(ActionTerm):
    """Joint position action term for controlling robot joints.
    
    This action term applies joint position commands to the robot's joints.
    The actions are scaled and added to the default joint positions.
    """
    
    cfg: "JointPositionActionCfg"
    
    def __init__(self, cfg: "JointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        # Get the robot
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Get joint indices
        self._joint_ids, self._joint_names = self._robot.find_joints(cfg.joint_names)
        
        # Get default joint positions for offset
        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        
        # Store configuration
        self._scale = cfg.scale
        self._use_default_offset = cfg.use_default_offset
        
    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        # Store raw actions
        self._raw_actions = actions.clone()
        
        # Scale actions
        scaled_actions = actions * self._scale
        
        # Add default offset if configured
        if self._use_default_offset:
            self._processed_actions = scaled_actions + self._default_joint_pos
        else:
            self._processed_actions = scaled_actions
    
    def apply_actions(self):
        # Set joint position targets
        self._robot.set_joint_position_target(
            self._processed_actions, 
            joint_ids=self._joint_ids
        )
    
    def reset(self, env_ids: Sequence[int]) -> None:
        # Reset default joint positions for reset environments
        self._default_joint_pos[env_ids] = self._robot.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class JointPositionActionCfg(ActionTermCfg):
    """Configuration for joint position action term."""
    
    class_type: type = JointPositionAction
    
    asset_name: str = "robot"
    """Name of the robot asset in the scene."""
    
    joint_names: list[str] = MISSING
    """List of joint name patterns to control."""
    
    scale: float = 1.0
    """Scale factor for actions."""
    
    use_default_offset: bool = True
    """Whether to add default joint positions as offset."""


# =============================================================================
# Leg Joint Position Action (for locomotion)
# =============================================================================

class LegJointPositionAction(ActionTerm):
    """Leg joint position action for Spot locomotion.
    
    Controls only the leg joints (12 joints total: 4 legs x 3 joints each).
    """
    
    cfg: "LegJointPositionActionCfg"
    
    def __init__(self, cfg: "LegJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Find leg joints
        leg_patterns = [".*_hx", ".*_hy", ".*_kn"]
        self._joint_ids = []
        self._joint_names = []
        
        for pattern in leg_patterns:
            ids, names = self._robot.find_joints(pattern)
            self._joint_ids.extend(ids)
            self._joint_names.extend(names)
        
        self._joint_ids = sorted(set(self._joint_ids))
        
        # Get default positions
        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        
        self._scale = cfg.scale
        self._use_default_offset = cfg.use_default_offset
        
    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()
        scaled_actions = actions * self._scale
        
        if self._use_default_offset:
            self._processed_actions = scaled_actions + self._default_joint_pos
        else:
            self._processed_actions = scaled_actions
    
    def apply_actions(self):
        self._robot.set_joint_position_target(
            self._processed_actions, 
            joint_ids=self._joint_ids
        )
    
    def reset(self, env_ids: Sequence[int]) -> None:
        self._default_joint_pos[env_ids] = self._robot.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class LegJointPositionActionCfg(ActionTermCfg):
    """Configuration for leg joint position action."""
    
    class_type: type = LegJointPositionAction
    
    asset_name: str = "robot"
    scale: float = 0.2
    use_default_offset: bool = True


# =============================================================================
# Arm Joint Position Action (for manipulation)
# =============================================================================

class ArmJointPositionAction(ActionTerm):
    """Arm joint position action for Spot manipulation.
    
    Controls only the arm joints (7 joints: shoulder, elbow, wrist, gripper).
    """
    
    cfg: "ArmJointPositionActionCfg"
    
    def __init__(self, cfg: "ArmJointPositionActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Find arm joints
        arm_patterns = ["arm0_sh.*", "arm0_el.*", "arm0_wr.*", "arm0_f1x"]
        self._joint_ids = []
        self._joint_names = []
        
        for pattern in arm_patterns:
            ids, names = self._robot.find_joints(pattern)
            self._joint_ids.extend(ids)
            self._joint_names.extend(names)
        
        self._joint_ids = sorted(set(self._joint_ids))
        
        # Get default positions
        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        
        self._scale = cfg.scale
        self._use_default_offset = cfg.use_default_offset
        
    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()
        scaled_actions = actions * self._scale
        
        if self._use_default_offset:
            self._processed_actions = scaled_actions + self._default_joint_pos
        else:
            self._processed_actions = scaled_actions
    
    def apply_actions(self):
        self._robot.set_joint_position_target(
            self._processed_actions, 
            joint_ids=self._joint_ids
        )
    
    def reset(self, env_ids: Sequence[int]) -> None:
        self._default_joint_pos[env_ids] = self._robot.data.default_joint_pos[env_ids][:, self._joint_ids]


@configclass
class ArmJointPositionActionCfg(ActionTermCfg):
    """Configuration for arm joint position action."""
    
    class_type: type = ArmJointPositionAction
    
    asset_name: str = "robot"
    scale: float = 0.5
    use_default_offset: bool = True


# =============================================================================
# Differential IK Action for Arm
# =============================================================================

class DifferentialIKAction(ActionTerm):
    """Differential IK action for arm end-effector control.
    
    Takes end-effector pose commands (position + orientation) and converts
    them to joint position commands using differential IK.
    """
    
    cfg: "DifferentialIKActionCfg"
    
    def __init__(self, cfg: "DifferentialIKActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Find arm joint indices
        arm_patterns = cfg.arm_joint_names
        self._joint_ids = []
        self._joint_names = []
        
        for pattern in arm_patterns:
            ids, names = self._robot.find_joints(pattern)
            self._joint_ids.extend(ids)
            self._joint_names.extend(names)
        
        self._joint_ids = sorted(set(self._joint_ids))
        
        # Get EE body index
        self._ee_body_idx = self._robot.find_bodies(cfg.ee_body_name)[0][0]
        
        # IK parameters
        self._damping = cfg.ik_damping
        self._position_scale = cfg.position_scale
        self._orientation_scale = cfg.orientation_scale
        
        # Initialize processed actions
        self._raw_actions = torch.zeros(env.num_envs, 7, device=env.device)
        self._processed_actions = self._robot.data.joint_pos[:, self._joint_ids].clone()
        
    @property
    def action_dim(self) -> int:
        # 3 for position delta + 4 for orientation quaternion (or 3 for euler)
        return 7 if self.cfg.use_quaternion else 6
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()
        
        # Get current joint positions
        current_joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        
        # Get current EE pose
        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx]
        
        # Extract position and orientation deltas from actions
        pos_delta = actions[:, :3] * self._position_scale
        
        if self.cfg.use_quaternion:
            # Quaternion control (actions[:, 3:7])
            target_quat = actions[:, 3:7]
            target_quat = target_quat / (torch.norm(target_quat, dim=-1, keepdim=True) + 1e-8)
        else:
            # Euler angle control (actions[:, 3:6])
            euler_delta = actions[:, 3:6] * self._orientation_scale
            # Convert euler to quaternion (simplified)
            target_quat = ee_quat  # For now, keep current orientation
        
        # Compute target position
        target_pos = ee_pos + pos_delta
        
        # Simple differential IK (Jacobian transpose method)
        # Get Jacobian
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._ee_body_idx - 1, :, self._joint_ids]
        
        # Position error
        pos_error = target_pos - ee_pos
        
        # Compute joint position delta using damped least squares
        J = jacobian[:, :3, :]  # Position Jacobian
        JT = J.transpose(-2, -1)
        
        # Damped least squares: delta_q = J^T (J J^T + lambda^2 I)^-1 e
        JJT = torch.bmm(J, JT)
        damping_matrix = self._damping ** 2 * torch.eye(3, device=self._robot.device).unsqueeze(0)
        JJT_damped = JJT + damping_matrix
        
        # Solve for joint deltas
        try:
            JJT_inv = torch.linalg.inv(JJT_damped)
            delta_q = torch.bmm(JT, torch.bmm(JJT_inv, pos_error.unsqueeze(-1))).squeeze(-1)
        except:
            delta_q = torch.zeros_like(current_joint_pos)
        
        # Apply joint position change
        self._processed_actions = current_joint_pos + delta_q
    
    def apply_actions(self):
        self._robot.set_joint_position_target(
            self._processed_actions, 
            joint_ids=self._joint_ids
        )
    
    def reset(self, env_ids: Sequence[int]) -> None:
        self._processed_actions[env_ids] = self._robot.data.joint_pos[env_ids][:, self._joint_ids]


@configclass
class DifferentialIKActionCfg(ActionTermCfg):
    """Configuration for differential IK action."""
    
    class_type: type = DifferentialIKAction
    
    asset_name: str = "robot"
    
    arm_joint_names: list[str] = ["arm0_sh.*", "arm0_el.*", "arm0_wr.*"]
    """Joint name patterns for the arm."""
    
    ee_body_name: str = "arm0_link_fngr"
    """Name of the end-effector body."""
    
    ik_damping: float = 0.05
    """Damping factor for IK solver."""
    
    position_scale: float = 0.1
    """Scale factor for position commands."""
    
    orientation_scale: float = 0.1
    """Scale factor for orientation commands."""
    
    use_quaternion: bool = True
    """Whether to use quaternion (7D) or euler (6D) for orientation."""


# =============================================================================
# Gripper Action
# =============================================================================

class GripperAction(ActionTerm):
    """Gripper action for controlling the Spot arm gripper.
    
    Takes a single value [-1, 1] where -1 is fully open and 1 is fully closed.
    """
    
    cfg: "GripperActionCfg"
    
    def __init__(self, cfg: "GripperActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Find gripper joint
        self._joint_ids, self._joint_names = self._robot.find_joints(cfg.gripper_joint_name)
        
        # Get joint limits
        joint_limits = self._robot.data.joint_limits[:, self._joint_ids, :]
        self._joint_min = joint_limits[:, :, 0]
        self._joint_max = joint_limits[:, :, 1]
        
        self._raw_actions = torch.zeros(env.num_envs, 1, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, len(self._joint_ids), device=env.device)
        
    @property
    def action_dim(self) -> int:
        return 1
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions.clone()
        
        # Map [-1, 1] to [min, max] joint position
        # -1 -> open (min), 1 -> closed (max)
        normalized = (actions + 1.0) / 2.0  # [0, 1]
        self._processed_actions = self._joint_min + normalized * (self._joint_max - self._joint_min)
    
    def apply_actions(self):
        self._robot.set_joint_position_target(
            self._processed_actions, 
            joint_ids=self._joint_ids
        )
    
    def reset(self, env_ids: Sequence[int]) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = (self._joint_min[env_ids] + self._joint_max[env_ids]) / 2.0


@configclass
class GripperActionCfg(ActionTermCfg):
    """Configuration for gripper action."""
    
    class_type: type = GripperAction
    
    asset_name: str = "robot"
    gripper_joint_name: str = "arm0_f1x"
