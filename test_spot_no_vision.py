#!/usr/bin/env python3
"""Test loading Isaac-Locomotion-Spot-NoVision to verify camera observations are the issue."""

import gymnasium as gym

# Import the configuration that doesn't use vision
from source.grasp.grasp.tasks.manager_based.locomotion_spot.locomotion_spot_env_cfg import (
    LocomotionSpotEnvCfg,
)

def main():
    """Test loading the Spot environment without vision."""
    print("Loading Isaac-Locomotion-Spot-NoVision environment...")
    
    try:
        # Try to create the environment without camera observations
        env_cfg = LocomotionSpotEnvCfg()
        print(f"✓ Configuration loaded successfully")
        print(f"  - num_envs: {env_cfg.scene.num_envs}")
        print(f"  - Scene: {list(env_cfg.scene.__dict__.keys())}")
        print(f"  - Observation groups: {list(env_cfg.observations.__dict__.keys())}")
        
        # Check if camera observations are in the config
        if hasattr(env_cfg.observations, 'camera'):
            print(f"  - Camera observations: ENABLED")
        else:
            print(f"  - Camera observations: DISABLED")
        
        # Try to instantiate the environment
        print("\nInstantiating environment...")
        env = gym.make("Isaac-Locomotion-Spot-v0", cfg=env_cfg)
        print("✓ Environment instantiated successfully!")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        # Get first observation
        print("\nGetting first observation...")
        obs, info = env.reset()
        print(f"✓ First observation obtained!")
        print(f"  - Observation shape: {obs.shape}")
        
        env.close()
        print("\n✓ SUCCESS: Spot environment works without vision!")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
