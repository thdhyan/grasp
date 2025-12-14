# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record demonstrations."""

from __future__ import annotations

import argparse
import os
import pickle
import time
from datetime import datetime
import carb

import gymnasium as gym
import torch
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for imitation learning.")
parser.add_argument("--task", type=str, default="Template-Grasp-Heir-Teleop-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--obs_key", action="store_true", default=False, help="Not relevant for manager based usually")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--save_path", type=str, default="logs/demos", help="Path to save recordings.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
import isaaclab_tasks.utils
from grasp_heir.tasks.manager_based.teleop import *
from teleop_se2_se3_agent import Se2Se3Keyboard # Import from sibling script or redefine

def main():
    """Record demos."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = isaaclab_tasks.utils.parse_env_cfg(
        args_cli.task, device="cpu" if args_cli.cpu else "cuda:0", num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # create controller
    teleop = Se2Se3Keyboard()
    
    # buffers
    recorded_actions = []
    recorded_obs = [] # This might be big. 
    # Usually we save observation DICT or FLATTENED?
    # For IL, usually flattened obs is used if policy uses flattened obs.
    # ManagerBasedEnv `step` returns `obs` as a dict "policy" group.
    
    obs, _ = env.reset()
    
    print("[INFO] Recording started. Press 'R' to reset and discard current episode (if implemented).")
    print("Saving to:", args_cli.save_path)
    if not os.path.exists(args_cli.save_path):
        os.makedirs(args_cli.save_path)

    episode_data = {"obs": [], "actions": [], "rewards": [], "dones": []}
    
    steps = 0
    try:
        while simulation_app.is_running():
            # get action
            action_np = teleop.get_action()
            action = torch.tensor(action_np, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
            
            # Record Pre-Step Obs? Or Post-Step?
            # Standard: (s, a, r, s', d)
            # We record `obs` (s) and `action` (a).
            # We handle single env for simplicity.
            
            # Extract obs: assume 'policy' group exists and is what we want
            current_obs = obs["policy"].cpu().numpy()
            
            # Step
            obs, rew, terminated, truncated, info = env.step(action)
            
            # Store
            episode_data["obs"].append(current_obs)
            episode_data["actions"].append(action_np) # Store single action
            episode_data["rewards"].append(rew.cpu().item())
            episode_data["dones"].append(terminated.cpu().item() or truncated.cpu().item())
            
            steps += 1
            
            # On reset
            if teleop.reset_pressed:
                 env.reset()
                 # Discard or Save? Usually teleop reset means failure or restart.
                 # Let's clean buffer
                 episode_data = {"obs": [], "actions": [], "rewards": [], "dones": []}
                 print("[INFO] Reset pressed. Episode discarded.")
    except KeyboardInterrupt:
        print("Stopping recording...")
        
    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(args_cli.save_path, f"demo_{timestamp}.pkl")
    
    # Add last obs? IL usually needs (s, a) pairs.
    # We have N obs and N actions.
    
    with open(filename, "wb") as f:
        pickle.dump(episode_data, f)
        
    print(f"[INFO] Saved {len(episode_data['actions'])} steps to {filename}")
    
    env.close()

if __name__ == "__main__":
    main()
