import torch
from typing import TYPE_CHECKING
import re

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ground_contact(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when the body's height is below the threshold.
    
    This filters out any bodies that match the pattern '.*_foot' and checks if any other
    bodies have a z-position below the threshold (implying ground contact).
    """
    # extract the used quantities (to enable type-hinting)
    robot = env.scene[asset_cfg.name]

    # Filter bodies
    if asset_cfg.body_names is not None:
        # Resolve indices using the provided body names/regex
        valid_indices, _ = robot.find_bodies(asset_cfg.body_names)
    else:
        # Default: filter out feet if no specific bodies requested
        body_names = robot.data.body_names
        valid_indices = [
            i for i, name in enumerate(body_names) 
            if not re.search(".*_foot", name)
        ]
    
    if not valid_indices:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Check if any non-foot body is below the threshold height
    # body_pos_w: (env, bodies, 3)
    # We check z component (index 2)
    min_heights = torch.min(robot.data.body_pos_w[:, valid_indices, 2], dim=1)[0]

    return min_heights < threshold


from isaaclab.utils.math import euler_xyz_from_quat

def robot_tilt_termination(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when the robot's roll or pitch exceeds the threshold.
    
    The threshold is expected to be in radians (or degrees if large).
    """
    # extract the used quantities (to enable type-hinting)
    robot = env.scene[asset_cfg.name]

    # Filter bodies
    if asset_cfg.body_names is not None:
        # Resolve indices using the provided body names/regex
        valid_indices, _ = robot.find_bodies(asset_cfg.body_names)
    else:
        # Default: filter out feet if no specific bodies requested
        body_names = robot.data.body_names
        valid_indices = [
            i for i, name in enumerate(body_names) 
            if not re.search(".*_foot", name)
        ]
    
    if not valid_indices:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Get orientation (w, x, y, z)
    quat = robot.data.body_rot_w[:, valid_indices, :]
    # Flatten to (batch * bodies, 4) for conversion
    # But usually euler_xyz_from_quat handles batched inputs.
    # Returns roll, pitch, yaw
    roll, pitch, _ = euler_xyz_from_quat(quat)
    
    # Handle threshold conversion if it seems to be in degrees (heuristic)
    if threshold > 10.0:
        threshold =  threshold * 3.14159 / 180.0

    return torch.any((torch.abs(roll) > threshold) | (torch.abs(pitch) > threshold), dim=1)
    