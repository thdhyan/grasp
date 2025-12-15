# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_distance_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: list[int],
    term_name: str,
    start_dist: float,
    end_dist: float,
    total_steps: int,
):
    """Curriculum for goal distance.
    
    Updates the 'position_range' parameter of the specified termination/event term.
    Linearly interpolates from start_dist to end_dist over total_steps global steps.
    """
    # Get current step
    # common_step_counter is usually available
    current_step = env.common_step_counter
    
    # Compute progress 0.0 to 1.0
    progress = min(max(current_step / total_steps, 0.0), 1.0)
    
    # Compute current distance range
    # We want range (-dist, dist) for x and y
    current_dist = start_dist + (end_dist - start_dist) * progress
    
    # Update the term's params
    if hasattr(env.event_manager, "terms"):
         term = env.event_manager.terms[term_name]
    elif hasattr(env.event_manager, "_terms"):
         term = env.event_manager._terms[term_name]
    else:
         # Fallback or error logging
         return 0.0
         
    # Update params
    term.params["position_range"] = (-current_dist, current_dist)
    
    return float(current_dist)
