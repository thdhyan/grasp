# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomanipulation tasks for Spot robot with arm.

This package provides two main task configurations:

1. **SpotLocomotionFlatEnvCfg**: Low-level locomotion policy that learns to walk
   and follow velocity commands. This can be used as a pre-trained policy
   for the high-level locomanipulation task.

2. **LocomanipulationSpotEnvCfg**: High-level locomanipulation task that
   combines locomotion and manipulation. Uses IK control for the arm
   and either learned or pre-trained locomotion for the legs.
"""

import gymnasium as gym

from . import agents

from .locomotion_spot_flat_env_cfg import (
    SpotLocomotionFlatEnvCfg,
    SpotLocomotionFlatEnvCfg_PLAY,
)
from .locomanipulation_spot_env_cfg import (
    LocomanipulationSpotEnvCfg,
    LocomanipulationSpotEnvCfg_PLAY,
    HierarchicalLocomanipulationSpotEnvCfg,
)

__all__ = [
    # Low-level locomotion
    "SpotLocomotionFlatEnvCfg",
    "SpotLocomotionFlatEnvCfg_PLAY",
    # High-level locomanipulation
    "LocomanipulationSpotEnvCfg",
    "LocomanipulationSpotEnvCfg_PLAY",
    "HierarchicalLocomanipulationSpotEnvCfg",
]

##
# Register Gym environments.
##

# Low-level locomotion for Spot (flat terrain)
gym.register(
    id="Isaac-Spot-Locomotion-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_flat_env_cfg:SpotLocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:SpotLocomotionPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Spot-Locomotion-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_flat_env_cfg:SpotLocomotionFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:SpotLocomotionPPORunnerCfg",
    },
)

# High-level locomanipulation for Spot
gym.register(
    id="Isaac-Spot-Locomanipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanipulation_spot_env_cfg:LocomanipulationSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:LocomanipulationSpotPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Spot-Locomanipulation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanipulation_spot_env_cfg:LocomanipulationSpotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:LocomanipulationSpotPPORunnerCfg",
    },
)

# Hierarchical locomanipulation (with pre-trained locomotion policy)
gym.register(
    id="Isaac-Spot-Locomanipulation-Hierarchical-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanipulation_spot_env_cfg:HierarchicalLocomanipulationSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:LocomanipulationSpotAsymmetricPPORunnerCfg",
    },
)
