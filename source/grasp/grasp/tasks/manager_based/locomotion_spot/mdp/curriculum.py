# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for multi-phase training.

5-Phase Curriculum:
1. Stand Up: Robot learns to stand from ground
2. Walk: Robot learns stable locomotion  
3. Arm Motion: Robot learns to move while controlling arm
4. Grasp: Robot learns to grasp the chair
5. Full Task: Robot learns complete task (stand, move, grasp, transport to goal)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def phase_transition_by_iterations(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_iterations: int,
    phase: int,
) -> None:
    """Curriculum term that transitions to next phase based on training iterations.
    
    Args:
        env: The RL environment instance.
        env_ids: The environment indices to apply curriculum to.
        target_iterations: Number of iterations to trigger phase transition.
        phase: The phase number (1-5).
    """
    # Get current training iteration from environment
    current_iteration = getattr(env, "current_iteration", 0)
    
    if current_iteration >= target_iterations:
        # Update phase in environment if not already set
        if not hasattr(env, "curriculum_phase") or env.curriculum_phase < phase:
            env.curriculum_phase = phase
            print(f"[Curriculum] Transitioning to Phase {phase} at iteration {current_iteration}")
            _apply_phase_reward_scaling(env, phase)


def phase_transition_by_reward(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_reward: float,
    phase: int,
) -> None:
    """Curriculum term that transitions based on average episode reward.
    
    Args:
        env: The RL environment instance.
        env_ids: The environment indices to apply curriculum to.
        target_reward: Target average reward to trigger phase transition.
        phase: The phase number (1-5).
    """
    # Get average reward from environment
    avg_reward = getattr(env, "average_episode_reward", 0.0)
    
    if avg_reward >= target_reward:
        if not hasattr(env, "curriculum_phase") or env.curriculum_phase < phase:
            env.curriculum_phase = phase
            print(f"[Curriculum] Transitioning to Phase {phase} with avg reward {avg_reward:.2f}")
            _apply_phase_reward_scaling(env, phase)


def _apply_phase_reward_scaling(env: ManagerBasedRLEnv, phase: int) -> None:
    """Apply reward weight scaling based on current phase.
    
    Phase 1 (Stand): Base height = 1.0x, Others = 0.0x
    Phase 2 (Walk): Base height = 0.5x, Velocity = 1.0x, Others = 0.0x
    Phase 3 (Arm): Navigation = 0.5x, Arm control = 1.0x
    Phase 4 (Grasp): Grasp rewards = 1.0x, Navigation = 0.3x
    Phase 5 (Full): All rewards active with appropriate weights
    
    Args:
        env: The RL environment instance.
        phase: The current phase number.
    """
    # Define reward scaling per phase
    # Format: {reward_name: scale_factor}
    phase_scales = {
        1: {  # Stand Up
            "base_height": 2.0,
            "flat_orientation": 2.0,
            "is_alive": 1.0,
            "termination_penalty": 1.0,
            "joint_torques": 0.5,
            # Disable others
            "track_lin_vel_xy": 0.0,
            "track_ang_vel_z": 0.0,
            "distance_to_chair": 0.0,
            "gripper_contact": 0.0,
            "wrist_contact": 0.0,
            "chair_to_goal": 0.0,
        },
        2: {  # Walk
            "base_height": 1.0,
            "flat_orientation": 1.0,
            "is_alive": 1.0,
            "termination_penalty": 1.0,
            "track_lin_vel_xy": 2.0,
            "track_ang_vel_z": 1.0,
            "air_time": 1.0,
            "arm_stability": 1.0,
            # Disable chair/grasp
            "distance_to_chair": 0.0,
            "gripper_contact": 0.0,
            "wrist_contact": 0.0,
            "chair_to_goal": 0.0,
        },
        3: {  # Arm Motion
            "base_height": 0.5,
            "flat_orientation": 0.5,
            "is_alive": 1.0,
            "termination_penalty": 1.0,
            "track_lin_vel_xy": 1.0,
            "track_ang_vel_z": 0.5,
            "distance_to_chair": 1.5,
            "facing_chair": 1.0,
            "ee_to_chair": 2.0,
            "wrist_contact": 1.0,
            # Disable grasp completion
            "gripper_contact": 0.0,
            "chair_to_goal": 0.0,
        },
        4: {  # Grasp
            "base_height": 0.3,
            "flat_orientation": 0.3,
            "is_alive": 1.0,
            "termination_penalty": 1.0,
            "track_lin_vel_xy": 0.5,
            "distance_to_chair": 1.0,
            "ee_to_chair": 2.0,
            "gripper_contact": 3.0,
            "wrist_contact": 1.5,
            "grasp_firmness": 2.0,
            # Disable chair transport
            "chair_to_goal": 0.0,
        },
        5: {  # Full Task
            "base_height": 0.5,
            "flat_orientation": 0.5,
            "is_alive": 1.0,
            "termination_penalty": 1.0,
            "track_lin_vel_xy": 0.5,
            "distance_to_chair": 0.3,
            "gripper_contact": 1.5,
            "wrist_contact": 0.5,
            "chair_to_goal": 2.0,
            "chair_to_goal_fine": 3.0,
            "chair_at_goal": 5.0,
        },
    }
    
    scales = phase_scales.get(phase, phase_scales[5])
    
    # Apply scales to reward manager if available
    if hasattr(env, "reward_manager") and env.reward_manager is not None:
        for term_name, scale in scales.items():
            if hasattr(env.reward_manager, term_name):
                term = getattr(env.reward_manager, term_name)
                if hasattr(term, "cfg") and hasattr(term.cfg, "weight"):
                    original_weight = term.cfg.weight
                    term.cfg.weight = original_weight * scale


def modify_reward_weight(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_name: str,
    weight: float,
    num_steps: int,
) -> float:
    """Modify a specific reward term weight based on training progress.
    
    Args:
        env: The RL environment instance.
        env_ids: The environment indices.
        term_name: Name of the reward term to modify.
        weight: Target weight value.
        num_steps: Number of steps over which to interpolate.
        
    Returns:
        Current weight value.
    """
    # Get current step count
    current_step = getattr(env, "common_step_counter", 0)
    
    # Linear interpolation
    progress = min(1.0, current_step / max(1, num_steps))
    
    if hasattr(env, "reward_manager") and env.reward_manager is not None:
        if hasattr(env.reward_manager, term_name):
            term = getattr(env.reward_manager, term_name)
            if hasattr(term, "cfg"):
                original = getattr(term.cfg, "_original_weight", term.cfg.weight)
                if not hasattr(term.cfg, "_original_weight"):
                    term.cfg._original_weight = original
                
                new_weight = original + (weight - original) * progress
                term.cfg.weight = new_weight
                return new_weight
    
    return weight


def increase_episode_length(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    target_length: float,
    num_steps: int,
) -> None:
    """Gradually increase episode length as training progresses.
    
    Args:
        env: The RL environment instance.
        env_ids: The environment indices.
        target_length: Target episode length in seconds.
        num_steps: Number of steps to reach target length.
    """
    current_step = getattr(env, "common_step_counter", 0)
    progress = min(1.0, current_step / max(1, num_steps))
    
    initial_length = getattr(env, "_initial_episode_length", env.cfg.episode_length_s)
    if not hasattr(env, "_initial_episode_length"):
        env._initial_episode_length = initial_length
    
    new_length = initial_length + (target_length - initial_length) * progress
    env.max_episode_length = int(new_length / env.cfg.sim.dt / env.cfg.decimation)


def increase_chair_distance(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    max_distance: float = 5.0,
    num_steps: int = 100000,
) -> None:
    """Gradually increase initial chair distance from robot.
    
    Args:
        env: The RL environment instance.
        env_ids: The environment indices.
        max_distance: Maximum starting distance for chair.
        num_steps: Number of steps to reach max distance.
    """
    current_step = getattr(env, "common_step_counter", 0)
    progress = min(1.0, current_step / max(1, num_steps))
    
    # Start with chair close (1.5m) and gradually increase
    min_distance = 1.5
    current_max_distance = min_distance + (max_distance - min_distance) * progress
    
    # Store for use in reset
    env.curriculum_chair_distance = current_max_distance
