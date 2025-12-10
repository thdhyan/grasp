# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command generators for the locomotion spot navigation and manipulation task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class GoalPositionCommandCfg(CommandTermCfg):
    """Configuration for the goal position command generator."""
    
    class_type: type = None  # Will be set after class definition
    
    # Ranges for random goal position generation
    pos_x_range: tuple[float, float] = (-5.0, 5.0)
    pos_y_range: tuple[float, float] = (-5.0, 5.0)
    pos_z: float = 0.0  # Goal height (usually on ground)
    
    # Resampling
    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    
    # Visualization
    debug_vis: bool = True
    goal_marker_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    goal_marker_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.8)


class GoalPositionCommand(CommandTerm):
    """Command term that generates random goal positions for each environment.
    
    The goal position is where the chair should be moved to by the robot.
    """
    
    cfg: GoalPositionCommandCfg
    
    def __init__(self, cfg: GoalPositionCommandCfg, env: ManagerBasedRLEnv):
        # Initialize goal marker attribute BEFORE calling super().__init__()
        # because super().__init__() will call set_debug_vis() which accesses self.goal_marker
        self.goal_marker = None
        
        super().__init__(cfg, env)
        
        # Initialize goal positions
        self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Create visualization markers if debug_vis is enabled
        if self.cfg.debug_vis:
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/GoalMarker"
            marker_cfg.markers["cuboid"].scale = self.cfg.goal_marker_scale
            marker_cfg.markers["cuboid"].visual_material.diffuse_color = self.cfg.goal_marker_color[:3]
            self.goal_marker = VisualizationMarkers(marker_cfg)
            
    def __str__(self) -> str:
        msg = "GoalPositionCommand:\n"
        msg += f"\tPos X range: {self.cfg.pos_x_range}\n"
        msg += f"\tPos Y range: {self.cfg.pos_y_range}\n"
        msg += f"\tResampling time: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        """The current goal positions.
        
        Returns:
            Goal positions of shape (num_envs, 3).
        """
        return self.goal_positions
    
    def _update_metrics(self):
        """Update metrics for logging."""
        pass
    
    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments.
        
        Args:
            env_ids: Environment indices to resample.
        """
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # Random X positions (relative to env origin)
        x = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.pos_x_range[0], self.cfg.pos_x_range[1]
        )
        # Random Y positions (relative to env origin)
        y = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.pos_y_range[0], self.cfg.pos_y_range[1]
        )
        # Fixed Z position (relative to env origin)
        z = torch.full((len(env_ids),), self.cfg.pos_z, device=self.device)
        
        # Get environment origins and offset the goal positions
        env_origins = self._env.scene.env_origins[env_ids_tensor]
        
        self.goal_positions[env_ids, 0] = x + env_origins[:, 0]
        self.goal_positions[env_ids, 1] = y + env_origins[:, 1]
        self.goal_positions[env_ids, 2] = z + env_origins[:, 2]
    
    def _update_command(self):
        """Update the command. Called every step."""
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization state."""
        if debug_vis:
            if self.goal_marker is None:
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/GoalMarker"
                marker_cfg.markers["cuboid"].scale = self.cfg.goal_marker_scale
                self.goal_marker = VisualizationMarkers(marker_cfg)
        else:
            if self.goal_marker is not None:
                self.goal_marker.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        if self.goal_marker is not None:
            # Visualize goal positions
            self.goal_marker.visualize(self.goal_positions)


# Set class type after definition
GoalPositionCommandCfg.class_type = GoalPositionCommand
