# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based locomotion spot environment.

This environment trains a Spot robot with arm to navigate to a chair, grasp it,
and move it to a randomly generated goal position.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from .locomotion_spot_env_cfg import LocomotionSpotEnvCfg


class LocomotionSpotEnv(ManagerBasedRLEnv):
    """Manager-based environment for Spot locomotion with arm manipulation.
    
    This environment extends the base manager-based RL environment to add:
    - Goal position tracking for each environment
    - Visualization arrows for velocities and forces
    - Custom reset logic for chair and goal positions
    """
    
    cfg: LocomotionSpotEnvCfg
    
    def __init__(self, cfg: LocomotionSpotEnvCfg, render_mode: str | None = None, **kwargs):
        # Pre-initialize _goal_positions to None before super().__init__() 
        # because observation functions access it during manager initialization
        self._goal_positions_fallback = None
        self._goal_positions_initialized = False
        
        # Call parent init (this will initialize managers that need goal_positions)
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize fallback goal positions (used before command manager is ready)
        self._initialize_goal_positions_fallback()
        self._goal_positions_initialized = True
        
        # Setup visualization
        self._setup_visualization()
    
    @property
    def goal_positions(self) -> torch.Tensor:
        """Get goal positions from command manager, or fallback during initialization.
        
        After initialization, returns positions from the goal_position command.
        During manager initialization (before command manager is ready), returns
        a fallback tensor with zeros or random positions.
        """
        # Try to get from command manager first (preferred source)
        if (hasattr(self, 'command_manager') and 
            self.command_manager is not None and 
            hasattr(self.command_manager, '_terms') and
            'goal_position' in self.command_manager._terms):
            return self.command_manager.get_command('goal_position')
        
        # Fallback during initialization
        if self._goal_positions_fallback is not None:
            return self._goal_positions_fallback
            
        # Ultimate fallback - return zeros with correct shape
        if hasattr(self, 'scene') and self.scene is not None:
            num_envs = self.scene.num_envs
            device = self.device
        elif hasattr(self, 'cfg') and hasattr(self.cfg, 'scene'):
            num_envs = self.cfg.scene.num_envs
            device = self.cfg.sim.device if hasattr(self.cfg, 'sim') else 'cuda:0'
        else:
            num_envs = 1
            device = 'cuda:0'
        return torch.zeros(num_envs, 3, device=device)
    
    def _initialize_goal_positions_fallback(self):
        """Initialize fallback goal positions for use during initialization.
        
        These are temporary positions used before the command manager is ready.
        Once the command manager is initialized, goal_positions property will
        return positions from the goal_position command instead.
        """
        # Create fallback goal positions tensor
        self._goal_positions_fallback = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Get goal position ranges from command config
        if hasattr(self.cfg.commands, 'goal_position'):
            cmd_cfg = self.cfg.commands.goal_position
            x_range = cmd_cfg.pos_x_range
            y_range = cmd_cfg.pos_y_range
            z_val = cmd_cfg.pos_z
        else:
            # Default ranges
            x_range = (-4.0, 4.0)
            y_range = (-4.0, 4.0)
            z_val = 0.0
        
        # Generate random positions for fallback
        self._goal_positions_fallback[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(
            x_range[0], x_range[1]
        )
        self._goal_positions_fallback[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(
            y_range[0], y_range[1]
        )
        self._goal_positions_fallback[:, 2] = z_val
        
        # Offset by environment origins
        self._goal_positions_fallback += self.scene.env_origins
    
    def _setup_visualization(self):
        """Setup visualization markers for velocities and forces."""
        # Goal position markers
        goal_marker_cfg = CUBOID_MARKER_CFG.replace(prim_path="/Visuals/GoalMarkers")
        self.goal_markers = VisualizationMarkers(goal_marker_cfg)
        
        # Robot velocity arrows (blue)
        robot_vel_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/RobotVelocity")
        self.robot_vel_arrows = VisualizationMarkers(robot_vel_cfg)
        
        # End-effector velocity/force arrows (green)
        ee_arrow_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/EEVelocity")
        self.ee_vel_arrows = VisualizationMarkers(ee_arrow_cfg)
        
        # Chair velocity arrows (green)
        chair_vel_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/ChairVelocity")
        self.chair_vel_arrows = VisualizationMarkers(chair_vel_cfg)
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at specified indices.
        
        Args:
            env_ids: Indices of environments to reset.
        """
        # Call parent reset (this will reset the command manager which handles goal positions)
        super()._reset_idx(env_ids)
        
        # Reset chair positions
        self._reset_chair_positions(env_ids)
    
    def _reset_chair_positions(self, env_ids: torch.Tensor):
        """Reset chair positions for specified environments.
        
        Args:
            env_ids: Indices of environments to reset chairs for.
        """
        chair = self.scene["chair"]
        
        # Get default chair state
        chair_state = chair.data.default_root_state[env_ids].clone()
        
        # Randomize chair position slightly
        chair_state[:, 0] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        chair_state[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        
        # Add environment origin offsets
        chair_state[:, :3] += self.scene.env_origins[env_ids]
        
        # Write to simulation
        chair.write_root_state_to_sim(chair_state, env_ids)
    
    def _post_physics_step(self):
        """Post physics step callback."""
        # Call parent
        super()._post_physics_step()
        
        # Update visualizations
        if self.sim.has_gui():
            self._update_visualizations()
    
    def _update_visualizations(self):
        """Update debug visualizations."""
        robot = self.scene["robot"]
        chair = self.scene["chair"]
        
        # Update goal markers
        self.goal_markers.visualize(self.goal_positions)
        
        # Update robot velocity arrows
        robot_pos = robot.data.root_pos_w.clone()
        robot_pos[:, 2] += 0.7  # Offset above robot
        robot_vel = robot.data.root_lin_vel_w
        
        # Compute arrow orientation from velocity
        vel_scale, vel_quat = self._velocity_to_arrow(robot_vel[:, :2])
        self.robot_vel_arrows.visualize(robot_pos, vel_quat, vel_scale)
        
        # Update chair velocity arrows
        chair_pos = chair.data.root_pos_w.clone()
        chair_pos[:, 2] += 0.5
        chair_vel = chair.data.root_lin_vel_w
        
        chair_scale, chair_quat = self._velocity_to_arrow(chair_vel[:, :2])
        self.chair_vel_arrows.visualize(chair_pos, chair_quat, chair_scale)
        
        # Update EE velocity arrows (if arm body exists)
        try:
            ee_body_id = robot.find_bodies("arm0_link_wr1")[0][0]
            ee_pos = robot.data.body_pos_w[:, ee_body_id, :]
            ee_vel = robot.data.body_lin_vel_w[:, ee_body_id, :]
            
            ee_scale, ee_quat = self._velocity_to_arrow(ee_vel[:, :2])
            self.ee_vel_arrows.visualize(ee_pos, ee_quat, ee_scale)
        except (IndexError, RuntimeError):
            pass
    
    def _velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert XY velocity to arrow scale and orientation.
        
        Args:
            xy_velocity: XY velocity tensor of shape (num_envs, 2).
            
        Returns:
            Tuple of (scale, quaternion) for arrow markers.
        """
        # Compute arrow scale based on velocity magnitude
        vel_mag = torch.norm(xy_velocity, dim=-1)
        default_scale = torch.tensor([0.3, 0.3, 0.5], device=self.device)
        arrow_scale = default_scale.unsqueeze(0).repeat(self.num_envs, 1)
        arrow_scale[:, 2] *= vel_mag.clamp(min=0.1, max=3.0)
        
        # Compute arrow orientation from velocity direction
        heading = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading)
        
        return arrow_scale, arrow_quat
    
    @property
    def robot(self):
        """Get the robot articulation."""
        return self.scene["robot"]
    
    @property
    def chair(self):
        """Get the chair articulation."""
        return self.scene["chair"]
