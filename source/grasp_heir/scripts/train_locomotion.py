# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL via Isaac Lab.
Modified for Spot Locomotion task with specific logging requirements.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Set CUDA device before importing torch or isaaclab components
# User requested CUDA_VISIBLE_DEVICES=3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=100, help="Interval between video recordings (in steps).")
# If video_interval is 1000 steps, checking if RSL-RL wrapper supports it directly or we rely on the env config?
# Isaac Lab's RSL-RL runner usually doesn't record videos during training by default unless configured.
# We will use the standard arguments and pass them to the runner or env.
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Spot-Grasp-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_project_name", type=str, default="isaac-spot-grasp-locomotion", help="WandB project name.")
parser.add_argument("--log_run_name", type=str, default=None, help="WandB run name. Defaults to timestamp.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import traceback

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from grasp_heir.tasks.manager_based.locomotion import * # Register task
from grasp_heir.tasks.manager_based.locomotion.agents.rsl_rl_ppo_cfg import SpotLocomotionPPORunnerCfg
from isaaclab.utils.dict import class_to_dict

def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = isaaclab_tasks.utils.parse_env_cfg(
        args_cli.task, device="cpu" if args_cli.cpu else "cuda:0", num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Create timestamp once for consistent folder naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args_cli.log_run_name if args_cli.log_run_name else timestamp
    log_dir = os.path.join("logs", "rsl_rl", args_cli.task, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO]: Logging to: {log_dir}")

    if args_cli.video:
        # Video folder inside the run folder
        video_folder = os.path.join(log_dir, "videos")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # Load agent configuration
    agent_cfg = SpotLocomotionPPORunnerCfg()
    agent_cfg_dict = class_to_dict(agent_cfg)
    
    # Override project and run names for wandb
    agent_cfg_dict["experiment_name"] = args_cli.log_project_name
    agent_cfg_dict["wandb_project"] = args_cli.log_project_name  # This is what RSL-RL actually uses
    agent_cfg_dict["run_name"] = run_name
    
    
    # Create runner
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device="cuda:0")

    # print information
    print(f"[INFO]: Random seed: {env_cfg.seed}")
    print(f"[INFO]: Environment: {env_cfg.scene.num_envs} instances")

    # train agent
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    import isaaclab_tasks.utils  # noqa: F401
    main()
