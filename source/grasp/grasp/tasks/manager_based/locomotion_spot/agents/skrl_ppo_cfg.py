# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SKRL PPO agent configuration for the manager-based locomotion spot environment.

This configuration supports CNN policies for camera observations.
"""

from isaaclab.utils import configclass


@configclass
class LocomotionSpotSkrlCfg:
    """SKRL configuration for Spot locomotion with manipulation.
    
    This configuration supports:
    - Multi-modal observations (proprioceptive + camera)
    - CNN encoders for image observations
    - Separate policies for different phases (optional)
    """
    
    # General settings
    seed = 42
    
    # Agent configuration
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
            "size": -1,  # Auto-detect from observation space
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
    
    # Model configuration
    model = {
        "actor": {
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "critic": {
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
    }


@configclass
class LocomotionSpotSkrlCfg_CNN(LocomotionSpotSkrlCfg):
    """SKRL configuration with CNN encoder for camera observations.
    
    This configuration uses a CNN to process RGB camera images and combines
    the features with proprioceptive observations.
    """
    
    # CNN encoder configuration
    cnn_encoder = {
        "input_channels": 3,  # RGB
        "input_height": 128,
        "input_width": 128,
        "conv_layers": [
            {"out_channels": 32, "kernel_size": 8, "stride": 4, "padding": 0},
            {"out_channels": 64, "kernel_size": 4, "stride": 2, "padding": 0},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0},
        ],
        "fc_layers": [512],
        "activation": "relu",
    }
    
    # Model with CNN features
    model = {
        "actor": {
            "hidden_dims": [1024, 512, 256],  # Larger for camera features
            "activation": "elu",
        },
        "critic": {
            "hidden_dims": [1024, 512, 256],
            "activation": "elu",
        },
    }


@configclass
class LocomotionSpotSkrlCfg_MultiPolicy:
    """SKRL configuration for multi-policy training.
    
    This configuration defines separate policies for:
    - Locomotion (legs control)
    - Manipulation (arm control)
    - Camera (visual processing)
    
    Note: This requires custom SKRL agent implementation for multi-agent RL.
    """
    
    seed = 42
    
    # Locomotion policy (legs)
    locomotion_policy = {
        "observation_keys": ["base_lin_vel", "base_ang_vel", "projected_gravity", "leg_joint_pos", "leg_joint_vel"],
        "action_space": 12,  # 4 legs x 3 joints
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
    }
    
    # Manipulation policy (arm)
    manipulation_policy = {
        "observation_keys": ["arm_joint_pos", "arm_joint_vel", "ee_pos", "ee_vel", "gripper_contact"],
        "action_space": 7,  # 6 arm joints + 1 gripper
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
    }
    
    # Camera policy (visual features)
    camera_policy = {
        "observation_keys": ["rgb_camera"],
        "cnn_encoder": {
            "input_channels": 3,
            "conv_layers": [
                {"out_channels": 32, "kernel_size": 8, "stride": 4},
                {"out_channels": 64, "kernel_size": 4, "stride": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 1},
            ],
            "fc_layers": [256],
        },
        "hidden_dims": [256, 128],
        "activation": "elu",
    }
    
    # Shared agent parameters
    agent = {
        "rollouts": 24,
        "learning_epochs": 5,
        "mini_batches": 4,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 1e-3,
        "entropy_loss_scale": 0.01,
        "value_loss_scale": 1.0,
    }
