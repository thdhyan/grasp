# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.sensors import ContactSensorCfg
import isaaclab.envs.mdp as mdp
from isaaclab_assets.robots.spot import SPOT_CFG

from grasp_heir.tasks.manager_based.locomotion.spot_locomotion_env_cfg import SpotLocomotionSceneCfg, SpotLocomotionEnvCfg
from .actions.locomotion_policy_action import LocomotionPolicyActionCfg

@configclass
class TeleopActionsCfg:
    """Action specifications for the Teleop env."""
    # Base control via trained policy
    base_action = LocomotionPolicyActionCfg(
        asset_name="robot",
        policy_path="logs/rsl_rl/Isaac-Velocity-Spot-Grasp-v0/gen_policy.pt", # Expected path after training
        joint_names=[".*_hx", ".*_hy", ".*_kn"],
    )
    
    # Arm control via IK (Delta Pose)
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["arm0_.*"],
        body_name="arm0_f1x", # End effector
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        debug_vis=True,
    )
    
    # Gripper?
    gripper_action = mdp.BinaryJointPositionActionCfg(
         asset_name="robot",
         joint_names=["arm0_f1x"], # Wait, f1x is finger?
         # Check spot.py: "arm0_f1x" is used for gripper in actuators.
         open_command=1.0,
         close_command=0.0,
    )

@configclass
class TeleopObservationsCfg(SpotLocomotionEnvCfg.observations.class_type):
    """Observation specifications for Teleop."""
    # We might want camera images for teleop/imitation?
    # For now, keep locomotion obs + arm info
    @configclass
    class PolicyCfg(ObsGroup):
        # Locomotion obs are hidden inside ActionTerm usually?
        # No, ActionTerm computes them for the *policy*.
        # The external agent (Teleoperator) needs to see something.
        
        # Base state
        base_state = ObsTerm(func=mdp.base_lin_vel) # Just examples
        
        # Arm state
        ee_pose = ObsTerm(func=mdp.root_pos_b) # Placeholder
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class GraspHeirTeleopEnvCfg(SpotLocomotionEnvCfg):
    """Configuration for teleoperation environment."""
    
    # Override actions
    actions: TeleopActionsCfg = TeleopActionsCfg()
    
    # Override observations (optional, defaults to locomotion obs which might be irrelevant for teleop user but needed for policy?)
    # "Inference: The environment step will internally query the trained locomotion policy".
    # The ActionTerm does this. So we don't need to expose policy obs to the "Agent" (user).
    # But we might want observations for the Imitation Learning agent later.
    
    # For data collection, we just need the environment to run.
    
    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4 # Match training
        self.episode_length_s = 60.0 # Longer for teleop
        
        # Disable randomization that disturbs the robot too much
        # We want the user to control it
        self.events.randomize_arm = None # Disable random arm movements
        self.events.push_robot = None # Disable pushing
        self.events.force_on_arm = None
        
        # Keep reset events?
        # Yes.
