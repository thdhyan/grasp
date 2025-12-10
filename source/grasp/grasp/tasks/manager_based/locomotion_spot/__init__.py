# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based locomotion spot environment package.

This package provides a manager-based RL environment for training a Boston Dynamics
Spot robot with arm to:
1. Navigate to a chair in the environment
2. Grasp the chair with the robot arm
3. Move the chair to a randomly generated goal position

Key Features:
- Manager-based workflow with loggable reward terms
- Random goal position generation per environment
- Contact sensors on gripper fingers, wrist, and feet
- Camera observations with CNN policy support
- Depth and instance segmentation from wrist and front cameras
- Debug visualization arrows for velocities and forces
- Increased environment spacing (8m)
- 5-Phase curriculum training support

Registered Environments:
- Isaac-Locomotion-Spot-Manager-v0: Base manager-based environment
- Isaac-Locomotion-Spot-Manager-Play-v0: Evaluation environment
- Isaac-Locomotion-Spot-Navigation-v0: Navigation phase only
- Isaac-Locomotion-Spot-Grasping-v0: Grasping phase only  
- Isaac-Locomotion-Spot-Manipulation-v0: Manipulation phase only
- Isaac-Locomotion-Spot-FullTask-v0: All phases combined
- Isaac-Locomotion-Spot-Camera-v0: With CNN camera policy

5-Phase Curriculum Environments:
- Isaac-Locomotion-Spot-Phase1-StandUp-v0: Robot learns to stand
- Isaac-Locomotion-Spot-Phase2-Walk-v0: Robot learns locomotion
- Isaac-Locomotion-Spot-Phase3-ArmMotion-v0: Robot learns arm control while walking
- Isaac-Locomotion-Spot-Phase4-Grasp-v0: Robot learns to grasp chair
- Isaac-Locomotion-Spot-Phase5-FullTask-v0: Complete task (stand, walk, grasp, transport)
- Isaac-Locomotion-Spot-Curriculum-v0: Automatic curriculum with phase transitions

Vision-based Environments (with depth and instance segmentation):
- Isaac-Locomotion-Spot-Vision-v0: Vision policy with depth + segmentation from wrist and front cameras
- Isaac-Locomotion-Spot-Vision-Play-v0: Evaluation environment for vision policy
"""

import gymnasium as gym

from . import agents
from .locomotion_spot_env import LocomotionSpotEnv
from .locomotion_spot_env_cfg import (
    LocomotionSpotEnvCfg,
    LocomotionSpotEnvCfg_PLAY,
    LocomotionSpotVisionEnvCfg,
    LocomotionSpotVisionEnvCfg_PLAY,
)
from .locomotion_spot_extended_cfg import (
    LocomotionSpotNavigationEnvCfg,
    LocomotionSpotGraspingEnvCfg,
    LocomotionSpotManipulationEnvCfg,
    LocomotionSpotFullTaskEnvCfg,
)
from .locomotion_spot_curriculum_cfg import (
    LocomotionSpotPhase1Cfg,
    LocomotionSpotPhase2Cfg,
    LocomotionSpotPhase3Cfg,
    LocomotionSpotPhase4Cfg,
    LocomotionSpotPhase5Cfg,
    LocomotionSpotCurriculumCfg,
)

##
# Register Gym environments.
##

# Base manager-based environment
gym.register(
    id="Isaac-Locomotion-Spot-Manager-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_env_cfg:LocomotionSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:LocomotionSpotSkrlCfg",
    },
)

# Play/evaluation environment
gym.register(
    id="Isaac-Locomotion-Spot-Manager-Play-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_env_cfg:LocomotionSpotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:LocomotionSpotSkrlCfg",
    },
)

# Navigation-focused environment (Phase 1)
gym.register(
    id="Isaac-Locomotion-Spot-Navigation-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_extended_cfg:LocomotionSpotNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg_Navigation",
    },
)

# Grasping-focused environment (Phase 2)
gym.register(
    id="Isaac-Locomotion-Spot-Grasping-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_extended_cfg:LocomotionSpotGraspingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

# Manipulation-focused environment (Phase 3)
gym.register(
    id="Isaac-Locomotion-Spot-Manipulation-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_extended_cfg:LocomotionSpotManipulationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg_Manipulation",
    },
)

# Full task environment (all phases)
gym.register(
    id="Isaac-Locomotion-Spot-FullTask-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_extended_cfg:LocomotionSpotFullTaskEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

# Camera-based environment with CNN policy
gym.register(
    id="Isaac-Locomotion-Spot-Camera-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_env_cfg:LocomotionSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg_Camera",
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:LocomotionSpotSkrlCfg_CNN",
    },
)

##
# 5-Phase Curriculum Environments
##

# Phase 1: Stand Up - Robot learns to stand from ground
gym.register(
    id="Isaac-Locomotion-Spot-Phase1-StandUp-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_curriculum_cfg:LocomotionSpotPhase1Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

# Phase 2: Walk - Robot learns stable locomotion
gym.register(
    id="Isaac-Locomotion-Spot-Phase2-Walk-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_curriculum_cfg:LocomotionSpotPhase2Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

# Phase 3: Arm Motion - Robot learns to move arm while walking
gym.register(
    id="Isaac-Locomotion-Spot-Phase3-ArmMotion-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_curriculum_cfg:LocomotionSpotPhase3Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

# Phase 4: Grasp - Robot learns to grasp the chair
gym.register(
    id="Isaac-Locomotion-Spot-Phase4-Grasp-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_curriculum_cfg:LocomotionSpotPhase4Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

# Phase 5: Full Task - Complete stand, move, grasp, transport to goal
gym.register(
    id="Isaac-Locomotion-Spot-Phase5-FullTask-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_curriculum_cfg:LocomotionSpotPhase5Cfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

# Full Curriculum - Automatic phase transitions
gym.register(
    id="Isaac-Locomotion-Spot-Curriculum-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_curriculum_cfg:LocomotionSpotCurriculumCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg",
    },
)

##
# Vision-based Environments (with depth and instance segmentation)
##

# Vision-based training environment with depth + segmentation from wrist and front cameras
gym.register(
    id="Isaac-Locomotion-Spot-Vision-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_env_cfg:LocomotionSpotVisionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg_Camera",
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:LocomotionSpotSkrlCfg_CNN",
    },
)

# Vision-based play/evaluation environment
gym.register(
    id="Isaac-Locomotion-Spot-Vision-Play-v0",
    entry_point=f"{__name__}.locomotion_spot_env:LocomotionSpotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomotion_spot_env_cfg:LocomotionSpotVisionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomotionSpotPPORunnerCfg_Camera",
        "skrl_cfg_entry_point": f"{agents.__name__}.skrl_ppo_cfg:LocomotionSpotSkrlCfg_CNN",
    },
)
