# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO agent configurations for Spot locomanipulation tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class SpotLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner configuration for Spot low-level locomotion."""
    
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "spot_locomotion_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
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
class SpotLocomotionAsymmetricPPORunnerCfg(SpotLocomotionPPORunnerCfg):
    """RSL-RL PPO runner with asymmetric actor-critic for Spot locomotion."""
    
    experiment_name = "spot_locomotion_flat_asymmetric"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class LocomanipulationSpotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner configuration for Spot locomanipulation."""
    
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "spot_locomanipulation"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
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
class LocomanipulationSpotAsymmetricPPORunnerCfg(LocomanipulationSpotPPORunnerCfg):
    """RSL-RL PPO runner with asymmetric actor-critic for Spot locomanipulation."""
    
    experiment_name = "spot_locomanipulation_asymmetric"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[768, 512, 256],
        activation="elu",
    )


__all__ = [
    "SpotLocomotionPPORunnerCfg",
    "SpotLocomotionAsymmetricPPORunnerCfg",
    "LocomanipulationSpotPPORunnerCfg",
    "LocomanipulationSpotAsymmetricPPORunnerCfg",
]
