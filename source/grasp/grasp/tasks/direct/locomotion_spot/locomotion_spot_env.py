# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import TiledCamera

from .locomotion_spot_env_cfg import LocomotionSpotEnvCfg

class LocomotionSpotEnv(DirectRLEnv):
    cfg: LocomotionSpotEnvCfg

    def __init__(self, cfg: LocomotionSpotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # get leg joint indices after scene is initialized
        self.leg_joint_indices, _ = self.robot.find_joints(".*_hx|.*_hy|.*_kn")
        self.arm_joint_indices, _ = self.robot.find_joints("arm0_sh.*|arm0_el.*|arm0_wr.*")
        self.gripper_joint_indices, _ = self.robot.find_joints("arm0_f1x")
        
        # Track leg positions for stall detection
        self.prev_leg_pos = None
        self.leg_stall_counter = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.chair = Articulation(self.cfg.chair)
        self.tiled_camera = TiledCamera(self.cfg.tiled_camera)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        
        # add to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["chair"] = self.chair
        self.scene.sensors["tiled_camera"] = self.tiled_camera
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # Apply actions to robot joints
        # Actions: [legs (12), arm (6), gripper (1)]
        
        # Legs
        leg_actions = self.actions[:, :12]
        leg_default_pos = self.robot.data.default_joint_pos[:, self.leg_joint_indices]
        leg_target = leg_actions * self.cfg.action_scale + leg_default_pos
        self.robot.set_joint_position_target(leg_target, joint_ids=self.leg_joint_indices)
        
        # Arm
        arm_actions = self.actions[:, 12:18]
        arm_default_pos = self.robot.data.default_joint_pos[:, self.arm_joint_indices]
        arm_target = arm_actions * self.cfg.action_scale + arm_default_pos
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_joint_indices)
        
        # Gripper
        gripper_actions = self.actions[:, 18:]
        gripper_default_pos = self.robot.data.default_joint_pos[:, self.gripper_joint_indices]
        gripper_target = gripper_actions * self.cfg.action_scale + gripper_default_pos
        self.robot.set_joint_position_target(gripper_target, joint_ids=self.gripper_joint_indices)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Camera images
        rgb = self.tiled_camera.data.output["rgb"]
        
        # Joint transformation (positions/velocities)
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # Base state
        root_pos = self.robot.data.root_pos_w - self.scene.env_origins
        root_rot = self.robot.data.root_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_w
        root_ang_vel = self.robot.data.root_ang_vel_w
        
        base_state = torch.cat([root_pos, root_rot, root_lin_vel, root_ang_vel], dim=-1)
        
        # Legs obs
        leg_pos = self.robot.data.joint_pos[:, self.leg_joint_indices]
        leg_vel = self.robot.data.joint_vel[:, self.leg_joint_indices]
        obs_legs = torch.cat([base_state, leg_pos, leg_vel], dim=-1)
        
        # Arm obs
        arm_pos = self.robot.data.joint_pos[:, self.arm_joint_indices]
        arm_vel = self.robot.data.joint_vel[:, self.arm_joint_indices]
        obs_arm = torch.cat([base_state, arm_pos, arm_vel], dim=-1)
        
        obs = torch.cat([base_state, joint_pos, joint_vel], dim=-1)
        
        # Return dictionary with policy observations and other data
        return {
            "policy": obs,
            "policy_legs": obs_legs,
            "policy_arm": obs_arm,
            "policy_cameras": rgb,
            "rgb": rgb
        } 

    def _get_rewards(self) -> torch.Tensor:
        # Distance to chair
        robot_pos = self.robot.data.root_pos_w
        chair_pos = self.chair.data.root_pos_w
        
        dist = torch.norm(robot_pos[:, :2] - chair_pos[:, :2], dim=-1)
        
        # Base height reward - penalize if below minimum
        base_height = self.robot.data.root_pos_w[:, 2]
        height_reward = torch.clamp(base_height - self.cfg.min_base_height, min=0.0)
        
        # Facing chair reward - dot product of forward direction and direction to chair
        # Robot forward is +X in local frame, need to transform by orientation
        root_quat = self.robot.data.root_quat_w  # (w, x, y, z)
        # Extract forward direction from quaternion (simplified: assume robot forward is +X)
        # Forward vector from quaternion: rotate [1, 0, 0] by quat
        qw, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        # Rotation of [1,0,0] by quaternion
        forward_x = 1 - 2 * (qy * qy + qz * qz)
        forward_y = 2 * (qx * qy + qw * qz)
        forward_dir = torch.stack([forward_x, forward_y], dim=-1)
        forward_dir = forward_dir / (torch.norm(forward_dir, dim=-1, keepdim=True) + 1e-6)
        
        # Direction to chair
        dir_to_chair = chair_pos[:, :2] - robot_pos[:, :2]
        dir_to_chair = dir_to_chair / (torch.norm(dir_to_chair, dim=-1, keepdim=True) + 1e-6)
        
        facing_reward = torch.sum(forward_dir * dir_to_chair, dim=-1)  # cos(angle), in [-1, 1]
        
        # Leg movement penalty - penalize if legs don't move for too long
        leg_pos = self.robot.data.joint_pos[:, self.leg_joint_indices]
        if self.prev_leg_pos is None:
            self.prev_leg_pos = leg_pos.clone()
            leg_movement = torch.ones(self.num_envs, device=self.device)
        else:
            leg_delta = torch.abs(leg_pos - self.prev_leg_pos).mean(dim=-1)
            leg_moved = leg_delta > self.cfg.leg_movement_threshold
            # Update stall counter
            self.leg_stall_counter = torch.where(leg_moved, torch.zeros_like(self.leg_stall_counter), self.leg_stall_counter + 1)
            # Penalty when stalled for too long
            leg_movement = (self.leg_stall_counter >= self.cfg.leg_stall_steps).float()
            self.prev_leg_pos = leg_pos.clone()
        
        reward = (
            self.cfg.rew_scale_dist_to_chair * dist +
            self.cfg.rew_scale_alive +
            self.cfg.rew_scale_base_height * height_reward +
            self.cfg.rew_scale_facing_chair * facing_reward +
            self.cfg.rew_scale_leg_movement * leg_movement
        )
        return reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Terminate if fell over or time out
        # Simple fall detection: base height < threshold
        died = self.robot.data.root_pos_w[:, 2] < 0.2
        time_out = self.episode_length_buf >= self.max_episode_length
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._all_indices
        super()._reset_idx(env_ids)
        
        # Reset leg stall tracking for reset envs
        self.leg_stall_counter[env_ids] = 0
        if self.prev_leg_pos is not None:
            self.prev_leg_pos[env_ids] = self.robot.data.joint_pos[env_ids][:, self.leg_joint_indices]
        
        # Note: super()._reset_idx(env_ids) handles events defined in cfg.events
        # We can add additional manual reset logic here if needed, 
        # but for now we rely on the events for robot reset.
        
        # Reset chair state (if not handled by events)
        chair_state = self.chair.data.default_root_state[env_ids]
        chair_state[:, :3] += self.scene.env_origins[env_ids]
        self.chair.write_root_state_to_sim(chair_state, env_ids)
