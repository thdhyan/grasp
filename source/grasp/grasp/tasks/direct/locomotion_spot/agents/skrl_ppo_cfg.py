# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class LocomotionSpotSkrlPPOAgentCfg:
    """Configuration for the SKRL PPO agent."""

    # General
    seed = 42

    # Agent
    agent = {
        "rollouts": 24,
        "learning_epochs": 5,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 1e-3,
        "learning_rate_scheduler": "AdaptiveLR",
        "learning_rate_scheduler_kwargs": {
            "kl_threshold": 0.01,
        },
        "state_preprocessor": True,
        "state_preprocessor_kwargs": {
            "size": 51,  # Observation space
        },
        "value_preprocessor": True,
        "value_preprocessor_kwargs": {
            "size": 1,
        },
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 1.0,
        "kl_threshold": 0.01,
        "rewards_shaper": None,
    }
    
    # Note: To implement 3 separate policies (Legs, Arm, Camera), you would need to:
    # 1. Define a Multi-Agent configuration here.
    # 2. Map the observation keys ("policy_legs", "policy_arm", "policy_cameras") to the respective agents.
    # 3. Ensure the action space is split correctly.
    # Currently, this config treats the robot as a single agent using the combined "policy" observation.
