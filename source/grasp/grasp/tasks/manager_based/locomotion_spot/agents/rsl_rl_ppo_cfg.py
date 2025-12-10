# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configuration for the manager-based locomotion spot environment."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LocomotionSpotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner configuration for Spot locomotion with manipulation.
    
    This configuration supports multi-phase training:
    1. Navigation: Move robot close to chair
    2. Grasping: Grasp the chair with the arm
    3. Manipulation: Move chair to goal position
    
    Uses asymmetric actor-critic:
    - Actor (policy): Uses 'policy' observation group (partial observability)
    - Critic: Uses 'critic' observation group (privileged full state)
    """
    
    # Training settings
    num_steps_per_env = 24
    max_iterations = 3000  # Longer training for complex task
    save_interval = 100
    experiment_name = "locomotion_spot_manager"
    
    # Disable empirical normalization for consistency
    empirical_normalization = False
    
    # Observation groups for asymmetric actor-critic
    # Actor uses 'policy' group, Critic uses 'critic' group with privileged info
    obs_groups = {
        "policy": ["policy"],  # Actor observations
        "critic": ["critic"],  # Critic observations (privileged)
    }
    
    # Policy network configuration
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    # PPO algorithm configuration
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LocomotionSpotPPORunnerCfg_Camera(LocomotionSpotPPORunnerCfg):
    """RSL-RL PPO runner configuration with CNN policy for camera observations.
    
    This extends the base configuration to use a CNN encoder for processing
    camera images alongside the MLP policy for proprioceptive observations.
    
    Uses observation groups:
    - Actor: 'policy' + 'camera' observations
    - Critic: 'critic' observations (privileged full state)
    """
    
    experiment_name = "locomotion_spot_manager_camera"
    
    # Include camera observations for policy
    obs_groups = {
        "policy": ["policy", "camera"],  # Actor uses policy + camera
        "critic": ["critic"],  # Critic uses privileged full state
    }
    
    # Larger networks for camera features
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )


@configclass
class LocomotionSpotPPORunnerCfg_Navigation(LocomotionSpotPPORunnerCfg):
    """Configuration focused on navigation phase training.
    
    Uses locomotion-specific observations for actor.
    """
    
    experiment_name = "locomotion_spot_navigation"
    max_iterations = 1500
    
    # Use locomotion-focused observations for actor
    obs_groups = {
        "policy": ["locomotion"],  # Locomotion-specific observations
        "critic": ["critic"],  # Full state for critic
    }
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )


@configclass
class LocomotionSpotPPORunnerCfg_Manipulation(LocomotionSpotPPORunnerCfg):
    """Configuration focused on manipulation phase training.
    
    Uses manipulation-specific observations for actor.
    """
    
    experiment_name = "locomotion_spot_manipulation"
    max_iterations = 2000
    
    # Use manipulation-focused observations for actor
    obs_groups = {
        "policy": ["manipulation"],  # Arm/manipulation-specific observations
        "critic": ["critic"],  # Full state for critic
    }
    
    # Lower learning rate for fine manipulation
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # Lower entropy for precise control
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=5.0e-4,  # Lower learning rate
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class LocomotionSpotPPORunnerCfg_Hierarchical(LocomotionSpotPPORunnerCfg):
    """Configuration for hierarchical policy with separate locomotion and manipulation.
    
    Uses combined locomotion + manipulation observations for actor.
    This is useful for end-to-end training of the full task.
    """
    
    experiment_name = "locomotion_spot_hierarchical"
    max_iterations = 5000
    
    # Combine locomotion and manipulation observations
    obs_groups = {
        "policy": ["locomotion", "manipulation"],  # Both observation groups
        "critic": ["critic"],  # Full privileged state
    }
    
    # Larger network for combined observations
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[768, 512, 256, 128],
        critic_hidden_dims=[768, 512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=8.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
