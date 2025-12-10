#!/bin/bash
# Run All Phases Sequentially (Manual Curriculum)
# Trains each phase one after another

conda activate grasp

COMMON_ARGS="--headless --num_envs 4 --enable_cameras --logger wandb --log_project_name csci8551 --video --video_interval 1000 --video_length 1200 --max_iterations 50000"

echo "========================================"
echo "Phase 1: Stand Up"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase1-StandUp-v0 $COMMON_ARGS
if [ $? -ne 0 ]; then
    echo "Phase 1 training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Phase 2: Walk"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase2-Walk-v0 $COMMON_ARGS
if [ $? -ne 0 ]; then
    echo "Phase 2 training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Phase 3: Arm Motion"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase3-ArmMotion-v0 $COMMON_ARGS
if [ $? -ne 0 ]; then
    echo "Phase 3 training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Phase 4: Grasp"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase4-Grasp-v0 $COMMON_ARGS
if [ $? -ne 0 ]; then
    echo "Phase 4 training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "Phase 5: Full Task"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase5-FullTask-v0 $COMMON_ARGS
if [ $? -ne 0 ]; then
    echo "Phase 5 training failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ All phases completed successfully!"
echo "========================================"
