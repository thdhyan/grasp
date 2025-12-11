# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command generators for Spot locomanipulation tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Velocity Command
# =============================================================================

class UniformVelocityCommand(CommandTerm):
    """Command generator for uniform velocity commands.
    
    Generates random linear and angular velocity commands for the robot base.
    """
    
    cfg: "UniformVelocityCommandCfg"
    
    def __init__(self, cfg: "UniformVelocityCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Initialize command buffer [vx, vy, omega_z]
        self._command = torch.zeros(env.num_envs, 3, device=env.device)
        
        # Resample counter
        self._resample_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        
        # Visualization
        if cfg.debug_vis:
            self._setup_visualization()
        
    def _setup_visualization(self):
        """Setup visualization markers for velocity commands."""
        marker_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/VelocityCommand")
        marker_cfg.markers["arrow"].scale = (0.5, 0.1, 0.1)
        self._goal_marker = VisualizationMarkers(marker_cfg)
    
    @property
    def command(self) -> torch.Tensor:
        return self._command
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands for specified environments."""
        if len(env_ids) == 0:
            return
        
        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_envs = len(env_ids)
        
        # Random linear velocity x
        self._command[env_ids, 0] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.ranges.lin_vel_x[0], self.cfg.ranges.lin_vel_x[1]
        )
        
        # Random linear velocity y
        self._command[env_ids, 1] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.ranges.lin_vel_y[0], self.cfg.ranges.lin_vel_y[1]
        )
        
        # Random angular velocity z
        self._command[env_ids, 2] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.ranges.ang_vel_z[0], self.cfg.ranges.ang_vel_z[1]
        )
        
        # Standing environments (zero velocity)
        if self.cfg.rel_standing_envs > 0:
            num_standing = int(num_envs * self.cfg.rel_standing_envs)
            if num_standing > 0:
                standing_ids = env_ids[:num_standing]
                self._command[standing_ids] = 0.0
    
    def _update_command(self):
        """Update command based on resampling time."""
        # Increment counter
        self._resample_counter += 1
        
        # Calculate resampling threshold in steps
        resample_steps = int(
            (self.cfg.resampling_time_range[0] + self.cfg.resampling_time_range[1]) / 2 
            / self._env.step_dt
        )
        
        # Find environments that need resampling
        resample_ids = (self._resample_counter >= resample_steps).nonzero(as_tuple=False).squeeze(-1)
        
        if len(resample_ids) > 0:
            self._resample_command(resample_ids)
            self._resample_counter[resample_ids] = 0
    
    def _update_metrics(self):
        """Update metrics for velocity command."""
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and hasattr(self, '_goal_marker'):
            # Update visualization
            robot_pos = self._robot.data.root_pos_w.clone()
            robot_pos[:, 2] += 0.5  # Raise above robot
            self._goal_marker.visualize(robot_pos)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        
        self._resample_counter[env_ids] = 0
        self._resample_command(env_ids)
        return {}


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for uniform velocity command generator."""
    
    class_type: type = UniformVelocityCommand
    
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    """Time range (min, max) in seconds for resampling commands."""
    
    rel_standing_envs: float = 0.1
    """Fraction of environments with zero velocity (standing)."""
    
    rel_heading_envs: float = 0.0
    """Fraction of environments with heading command (not implemented)."""
    
    heading_command: bool = False
    """Whether to use heading command instead of angular velocity."""
    
    debug_vis: bool = True
    """Whether to visualize commands."""
    
    @configclass
    class Ranges:
        """Velocity command ranges."""
        lin_vel_x: tuple[float, float] = (-1.0, 1.0)
        lin_vel_y: tuple[float, float] = (-0.5, 0.5)
        ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    
    ranges: Ranges = Ranges()


# =============================================================================
# Goal Position Command
# =============================================================================

class GoalPositionCommand(CommandTerm):
    """Command generator for goal position targets.
    
    Generates random goal positions for the robot/object to reach.
    """
    
    cfg: "GoalPositionCommandCfg"
    
    def __init__(self, cfg: "GoalPositionCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        # Initialize command buffer [x, y, z]
        self._command = torch.zeros(env.num_envs, 3, device=env.device)
        
        # Visualization
        if cfg.debug_vis:
            self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup goal position markers."""
        marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/GoalPosition")
        marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
        self._goal_marker = VisualizationMarkers(marker_cfg)
    
    @property
    def command(self) -> torch.Tensor:
        return self._command
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments."""
        if len(env_ids) == 0:
            return
        
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_envs = len(env_ids)
        
        # Random position within bounds
        self._command[env_ids, 0] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.pos_x_range[0], self.cfg.pos_x_range[1]
        )
        self._command[env_ids, 1] = torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.pos_y_range[0], self.cfg.pos_y_range[1]
        )
        self._command[env_ids, 2] = self.cfg.pos_z
        
        # Add environment origin offset
        self._command[env_ids] += self._env.scene.env_origins[env_ids]
    
    def _update_command(self):
        """Update command (goal positions are typically static until reset)."""
        pass
    
    def _update_metrics(self):
        """Update metrics for the command term.
        
        This is called each step to update any metrics tracked by the command.
        For goal positions, we track distance to goal.
        """
        # No metrics to update for static goal positions
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization for goal position markers."""
        if debug_vis and hasattr(self, '_goal_marker'):
            self._goal_marker.visualize(self._command)
    
    def _debug_vis_callback(self, event):
        """Callback for debug visualization updates."""
        if hasattr(self, '_goal_marker'):
            self._goal_marker.visualize(self._command)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Reset command for specified environments.
        
        Args:
            env_ids: Environment indices to reset. If None, resets all.
            
        Returns:
            Dictionary of metrics (empty for goal position command).
        """
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        
        self._resample_command(env_ids)
        
        # Return empty metrics dict (required by command manager)
        return {}


@configclass
class GoalPositionCommandCfg(CommandTermCfg):
    """Configuration for goal position command generator."""
    
    class_type: type = GoalPositionCommand
    
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """Time range for resampling (very large = only on reset)."""
    
    pos_x_range: tuple[float, float] = (-3.0, 3.0)
    """X position range relative to environment origin."""
    
    pos_y_range: tuple[float, float] = (-3.0, 3.0)
    """Y position range relative to environment origin."""
    
    pos_z: float = 0.0
    """Z position (height) of goal."""
    
    debug_vis: bool = True
    """Whether to visualize goal positions."""


# =============================================================================
# EE Target Command (for manipulation)
# =============================================================================

class EETargetCommand(CommandTerm):
    """Command generator for end-effector target poses.
    
    Generates target positions and orientations for the arm end-effector.
    """
    
    cfg: "EETargetCommandCfg"
    
    def __init__(self, cfg: "EETargetCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        self._robot: Articulation = env.scene[cfg.asset_name]
        
        # Initialize command buffer [x, y, z, qw, qx, qy, qz]
        self._command = torch.zeros(env.num_envs, 7, device=env.device)
        self._command[:, 3] = 1.0  # Default quaternion w = 1
        
        # Visualization
        if cfg.debug_vis:
            self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup EE target markers."""
        marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/EETarget")
        marker_cfg.markers["cuboid"].scale = (0.05, 0.05, 0.05)
        self._goal_marker = VisualizationMarkers(marker_cfg)
    
    @property
    def command(self) -> torch.Tensor:
        return self._command
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample EE target for specified environments."""
        if len(env_ids) == 0:
            return
        
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_envs = len(env_ids)
        
        # Get robot base position for relative positioning
        base_pos = self._robot.data.root_pos_w[env_ids]
        
        # Random position relative to base
        self._command[env_ids, 0] = base_pos[:, 0] + torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.pos_x_range[0], self.cfg.pos_x_range[1]
        )
        self._command[env_ids, 1] = base_pos[:, 1] + torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.pos_y_range[0], self.cfg.pos_y_range[1]
        )
        self._command[env_ids, 2] = base_pos[:, 2] + torch.empty(num_envs, device=self.device).uniform_(
            self.cfg.pos_z_range[0], self.cfg.pos_z_range[1]
        )
        
        # Default orientation (pointing down)
        self._command[env_ids, 3] = 1.0
        self._command[env_ids, 4:7] = 0.0
    
    def _update_command(self):
        pass
    
    def _update_metrics(self):
        """Update metrics for EE target command."""
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis and hasattr(self, '_goal_marker'):
            self._goal_marker.visualize(self._command[:, :3])
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        
        self._resample_command(env_ids)
        return {}


@configclass
class EETargetCommandCfg(CommandTermCfg):
    """Configuration for end-effector target command generator."""
    
    class_type: type = EETargetCommand
    
    asset_name: str = "robot"
    """Name of the robot asset."""
    
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """Time range for resampling."""
    
    pos_x_range: tuple[float, float] = (0.3, 0.6)
    """X position range relative to robot base."""
    
    pos_y_range: tuple[float, float] = (-0.3, 0.3)
    """Y position range relative to robot base."""
    
    pos_z_range: tuple[float, float] = (0.2, 0.6)
    """Z position range relative to robot base."""
    
    debug_vis: bool = True
    """Whether to visualize EE targets."""
