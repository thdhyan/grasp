#!/usr/bin/env python3
"""Quick test to check if environment config loads."""
import sys
try:
    from source.grasp.grasp.tasks.manager_based.locomotion_spot.locomotion_spot_env_cfg import (
        LocomotionSpotEnvCfg,
        LocomotionSpotVisionEnvCfg,
    )
    
    print("✓ LocomotionSpotEnvCfg loaded successfully")
    cfg = LocomotionSpotEnvCfg()
    print(f"  - num_envs: {cfg.scene.num_envs}")
    print(f"  - Terminations: {list(cfg.terminations.__dict__.keys())}")
    
    print("\n✓ LocomotionSpotVisionEnvCfg loaded successfully")
    vision_cfg = LocomotionSpotVisionEnvCfg()
    print(f"  - num_envs: {vision_cfg.scene.num_envs}")
    print(f"  - Has camera observations: {hasattr(vision_cfg.observations, 'camera')}")
    
    sys.exit(0)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
