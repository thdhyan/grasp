# Training Commands for Spot Robot Curriculum

## Setup
```bash
conda activate grasp
```

## NEW: Updated Observation Groups

The environment now has multiple observation groups for asymmetric actor-critic training:

| Group | Purpose | Observations |
|-------|---------|--------------|
| `policy` | Main actor | Base state, joints, commands, contacts, spatial (chair/goal relative to robot, direction vectors) |
| `critic` | Value function | Full privileged state (global positions, all distances, full robot state) |
| `locomotion` | Leg control | Base state, leg joints, velocity commands, feet contacts |
| `manipulation` | Arm control | Arm joints, EE state, gripper contacts, chair relative to EE |
| `camera` | Vision | Depth, RGB, instance segmentation from front + wrist cameras |

## Phase 1: Stand Up
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase1-StandUp-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

## Phase 2: Walk
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase2-Walk-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

## Phase 3: Arm Motion
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase3-ArmMotion-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

## Phase 4: Grasp
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase4-Grasp-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

## Phase 5: Full Task
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Phase5-FullTask-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

## Automatic Curriculum (All Phases with Auto-Transitions)
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Curriculum-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

## Vision-Based Training (Depth + Segmentation)
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Vision-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

## Base Manager Environment
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Manager-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

---

## Specialized Training Configurations

### Navigation-Focused (uses locomotion observations)
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Navigation-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

### Manipulation-Focused (uses manipulation observations)
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Manipulation-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

---

## Combined Script (Run All Phases Sequentially)
Save as `train_all_phases.sh`:

```bash
#!/bin/bash

conda activate grasp

COMMON_ARGS="--headless --num_envs 4 --enable_cameras --logger wandb --log_project_name csci8551 --video --video_interval 1000 --video_length 1200 --max_iterations 50000"

echo "========================================"
echo "Phase 1: Stand Up"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py --task Isaac-Locomotion-Spot-Phase1-StandUp-v0 $COMMON_ARGS

echo ""
echo "========================================"
echo "Phase 2: Walk"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py --task Isaac-Locomotion-Spot-Phase2-Walk-v0 $COMMON_ARGS

echo ""
echo "========================================"
echo "Phase 3: Arm Motion"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py --task Isaac-Locomotion-Spot-Phase3-ArmMotion-v0 $COMMON_ARGS

echo ""
echo "========================================"
echo "Phase 4: Grasp"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py --task Isaac-Locomotion-Spot-Phase4-Grasp-v0 $COMMON_ARGS

echo ""
echo "========================================"
echo "Phase 5: Full Task"
echo "========================================"
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py --task Isaac-Locomotion-Spot-Phase5-FullTask-v0 $COMMON_ARGS

echo ""
echo "========================================"
echo "All phases completed!"
echo "========================================"
```

Run with:
```bash
chmod +x train_all_phases.sh
./train_all_phases.sh
```

---

## Or Use the Automatic Curriculum (Recommended)
This single command runs all phases with automatic transitions:

```bash
conda activate grasp && \
CUDA_VISIBLE_DEVICES=2 python scripts/rsl_rl/train.py \
  --task Isaac-Locomotion-Spot-Curriculum-v0 \
  --headless --num_envs 4 --enable_cameras \
  --logger wandb --log_project_name csci8551 \
  --video --video_interval 1000 --video_length 1200 \
  --max_iterations 50000
```

---

## CLI Arguments Explanation
- `--task`: Environment/task to train on
- `--headless`: Run without graphics display
- `--num_envs 4`: Number of parallel environments (4 for memory efficiency)
- `--enable_cameras`: Enable camera sensors for vision tasks
- `--logger wandb`: Log training metrics to Weights & Biases
- `--log_project_name csci8551`: WandB project name
- `--video`: Record videos during training
- `--video_interval 1000`: Record video every 1000 steps
- `--video_length 1200`: Length of each recorded video (steps)
- `--max_iterations 50000`: Total training iterations
- `CUDA_VISIBLE_DEVICES=2`: Use GPU device 2 (change as needed)

---

## New Observation Groups Details

### Policy Group (Actor Input) - ~100+ dims
- `base_lin_vel` (3): Base linear velocity
- `base_ang_vel` (3): Base angular velocity
- `projected_gravity` (3): Gravity in body frame
- `joint_pos` (19): All joint positions
- `joint_vel` (19): All joint velocities
- `actions` (19): Last actions
- `velocity_commands` (3): Velocity command
- `goal_commands` (3): Goal position command
- `feet_contact` (4): Foot contact states
- `gripper_contact_status` (1): Gripper contact
- `wrist_contact_status` (1): Wrist contact
- `chair_pos_rel` (3): Chair position in robot frame
- `goal_pos_rel` (3): Goal position in robot frame
- `direction_to_chair` (3): Direction vector to chair
- `direction_to_goal` (3): Direction vector to goal
- `dist_to_chair` (1): Distance to chair
- `dist_chair_to_goal` (1): Chair-to-goal distance
- `direction_ee_to_chair` (3): EE-to-chair direction
- `dist_ee_to_chair` (1): EE-to-chair distance

### Critic Group (Value Function Input) - Privileged
- All robot state (no noise)
- Global positions (robot, chair, goal)
- All relative positions and distances
- Full contact information

### Locomotion Group (Leg Policy)
- Base state
- Leg joint positions/velocities only
- Velocity commands
- Foot contacts
- Direction to chair

### Manipulation Group (Arm Policy)
- Arm joint positions/velocities
- End-effector position/velocity
- Gripper contact and force
- Chair position relative to EE
- Chair velocity
