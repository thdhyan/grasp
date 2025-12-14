# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to teleoperate the robot with keyboard.
Controls:
Base: WASD (Lin XY), QE (Ang Z)
Arm: IJKL (Lin XY), UO (Lin Z), 1-6 (Orientation? Hard on keyboard).
"""

from __future__ import annotations

import argparse
import carb
import gymnasium as gym
import numpy as np
import torch
from collections import deque

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperate robot with keyboard.")
parser.add_argument("--task", type=str, default="Template-Grasp-Heir-Teleop-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se3Keyboard # Se3Keyboard exists?
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import wrap_to_pi

# Register tasks
from grasp_heir.tasks.manager_based.teleop import *
import isaaclab_tasks.utils

class Se2Se3Keyboard:
    """Keyboard controller for SE2 (Base) + SE3 (Arm)."""
    
    def __init__(self):
        # We can combine two keyboard controllers or write one.
        # Se2Keyboard handles WASD/QE -> commands (3 dim).
        # Se3Keyboard handles ...
        # If we open two listeners, they might conflict or we can just map different keys.
        # Isaac Lab Se2Keyboard uses: W, S, A, D, Q, E.
        # Let's create our own based on carb inputs to map specifically as requested.
        
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_launcher.app.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )
        
        # State
        self.base_vel = np.zeros(3) # x, y, w
        self.arm_delta_pose = np.zeros(6) # x, y, z, roll, pitch, yaw
        
        # Gains
        self.base_gain = 1.0
        self.arm_pos_gain = 0.01 
        self.arm_rot_gain = 0.01
        
        self.reset_pressed = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            val = 1.0
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            val = 0.0
        else:
            return True
            
        key = event.input
        
        # Base (Velocity) - Holding key sets velocity
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
             if key == carb.input.KeyboardInput.W: self.base_vel[0] = 1.0
             elif key == carb.input.KeyboardInput.S: self.base_vel[0] = -1.0
             elif key == carb.input.KeyboardInput.A: self.base_vel[1] = 1.0
             elif key == carb.input.KeyboardInput.D: self.base_vel[1] = -1.0
             elif key == carb.input.KeyboardInput.Q: self.base_vel[2] = 1.0
             elif key == carb.input.KeyboardInput.E: self.base_vel[2] = -1.0
             
             # Arm (Delta Pose) - Holding key sets delta
             elif key == carb.input.KeyboardInput.I: self.arm_delta_pose[0] = 1.0
             elif key == carb.input.KeyboardInput.K: self.arm_delta_pose[0] = -1.0
             elif key == carb.input.KeyboardInput.J: self.arm_delta_pose[1] = 1.0
             elif key == carb.input.KeyboardInput.L: self.arm_delta_pose[1] = -1.0
             elif key == carb.input.KeyboardInput.U: self.arm_delta_pose[2] = 1.0
             elif key == carb.input.KeyboardInput.O: self.arm_delta_pose[2] = -1.0
             
             elif key == carb.input.KeyboardInput.R: self.reset_pressed = True

             
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
             if key in [carb.input.KeyboardInput.W, carb.input.KeyboardInput.S]: self.base_vel[0] = 0.0
             if key in [carb.input.KeyboardInput.A, carb.input.KeyboardInput.D]: self.base_vel[1] = 0.0
             if key in [carb.input.KeyboardInput.Q, carb.input.KeyboardInput.E]: self.base_vel[2] = 0.0
             
             if key in [carb.input.KeyboardInput.I, carb.input.KeyboardInput.K]: self.arm_delta_pose[0] = 0.0
             if key in [carb.input.KeyboardInput.J, carb.input.KeyboardInput.L]: self.arm_delta_pose[1] = 0.0
             if key in [carb.input.KeyboardInput.U, carb.input.KeyboardInput.O]: self.arm_delta_pose[2] = 0.0
             
             if key == carb.input.KeyboardInput.R: self.reset_pressed = False

        return True

    def get_action(self):
        # Return concatenated action
        # Base: 3
        # Arm: 6
        # Gripper: 1? (Not mapped yet)
        
        # Apply gains
        b = self.base_vel * self.base_gain
        a = self.arm_delta_pose * self.arm_pos_gain # Rotations ignored for now
        
        return np.concatenate([b, a, [0.0]]) # 3 + 6 + 1 = 10? verify env actions

    def __del__(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)


def main():
    """Run teleoperation."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = isaaclab_tasks.utils.parse_env_cfg(
        args_cli.task, device="cpu" if args_cli.cpu else "cuda:0", num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # create controller
    teleop = Se2Se3Keyboard()
    
    # reset environment
    obs, _ = env.reset()
    
    print("[INFO] Teleop started.")
    print("Controls:")
    print("  Base: W/S (X), A/D (Y), Q/E (Yah)")
    print("  Arm: I/K (X), J/L (Y), U/O (Z)")
    
    while simulation_app.is_running():
        # get action
        action_np = teleop.get_action()
        # Shape mismatch handling needed
        # Env expects [num_envs, action_dim]
        # Action dim?
        # Base: 3
        # Arm: 6 (Delta Pose: x,y,z, roll,pitch,yaw) -> DIK controller
        # Gripper: 1
        # Total: 10
        
        # Check env action space
        # env.action_space is typically Box(low, high, shape)
        
        # Convert to tensor
        action = torch.tensor(action_np, device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
        
        # step environment
        obs, rew, terminated, truncated, info = env.step(action)
        
        # reset if needed
        if teleop.reset_pressed:
            env.reset()
            
    # close environment
    env.close()

if __name__ == "__main__":
    main()
