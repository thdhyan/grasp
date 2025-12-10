#!/bin/bash
# Train Phase 3: Arm Motion
# Robot learns to move arm while walking

conda activate grasp
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase3-ArmMotion-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
