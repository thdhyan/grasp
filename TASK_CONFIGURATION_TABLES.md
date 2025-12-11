# Spot Robot Locomanipulation Task Configuration

## Low-Level Locomotion Environment (Isaac-Spot-Locomotion-Flat-v0)

### Scene Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Num Environments** | 1024 | Number of parallel environments |
| **Environment Spacing** | 4.0 m | Distance between environment origins |
| **Episode Length** | 20 s | Maximum episode duration |
| **Simulation dt** | 0.005 s | Physics simulation timestep |
| **Decimation** | 4 | Action update frequency (every 4 steps) |
| **Robot Spawn Height** | 0.65 m | Initial height for robot COM |

---

## Reward Terms (16 Total, Negative Weight = Penalty)

### Locomotion Control Rewards

| Index | Term Name | Function | Weight | Parameters | Purpose |
|-------|-----------|----------|--------|------------|---------|
| 0 | `track_lin_vel_xy_exp` | Exponential kernel | **+1.5** | std=0.5 | Reward tracking linear velocity commands (x, y) |
| 1 | `track_ang_vel_z_exp` | Exponential kernel | **+0.75** | std=0.5 | Reward tracking angular velocity command (yaw) |
| 2 | `lin_vel_z_l2` | L2 norm penalty | **-2.0** | — | Penalize vertical velocity (keep robot level) |
| 3 | `ang_vel_xy_l2` | L2 norm penalty | **-0.05** | — | Penalize roll and pitch angular velocity |
| 4 | `flat_orientation_l2` | L2 norm penalty | **-0.2** | — | Penalize deviation from flat orientation |

### Base Height Control Rewards

| Index | Term Name | Function | Weight | Parameters | Purpose |
|-------|-----------|----------|--------|------------|---------|
| 5 | `base_height_l2` | L2 norm penalty | **-1.0** | target=0.6m, std=0.1m | Penalize deviation from target height |
| 6 | `base_height_exp` | Exponential reward | **+0.5** | target=0.6m | Reward for maintaining target height |

### Navigation Rewards

| Index | Term Name | Function | Weight | Parameters | Purpose |
|-------|-----------|----------|--------|------------|---------|
| 7 | `goal_reach` | Exponential reward | **+1.0** | std=2.0m | Reward distance-based approaching to goal |
| 8 | `goal_heading` | Exponential reward | **+0.3** | std=0.5rad | Reward facing towards goal position |

### Gait and Contact Rewards

| Index | Term Name | Function | Weight | Parameters | Purpose |
|-------|-----------|----------|--------|------------|---------|
| 9 | `feet_air_time` | Custom function | **+0.125** | mode_time=0.3s, vel_threshold=0.5m/s | Encourage proper gait pattern (diagonal pairs) |

### Regularization Penalties

| Index | Term Name | Function | Weight | Parameters | Purpose |
|-------|-----------|----------|--------|------------|---------|
| 10 | `action_rate_l2` | L2 norm penalty | **-0.01** | — | Penalize high action changes (smooth control) |
| 11 | `joint_acc_l2` | L2 norm penalty | **-2.5e-7** | — | Penalize joint accelerations (energy efficiency) |
| 12 | `joint_torques_l2` | L2 norm penalty | **-1e-5** | — | Penalize high joint torques (energy efficiency) |
| 13 | `feet_slide` | Custom function | **-0.25** | threshold=1.0N | Penalize feet slipping on ground |
| 14 | `undesired_contacts` | Contact penalty | **-1.0** | body_contact, threshold=1.0N | Penalize body collisions (fall detection) |
| 15 | `termination_penalty` | Penalty term | **-200.0** | — | Penalize early termination (episode failure) |

**Total Positive Weights:** 3.68 | **Total Negative Weights:** -204.61

---

## Termination Terms (3 Total)

| Index | Term Name | Condition | Grace Period | Parameters | Consequence |
|-------|-----------|-----------|--------------|------------|-------------|
| 0 | `time_out` | Episode length exceeded | — | episode_length=20s | Episode ends normally (time out flag) |
| 1 | `bad_orientation` | Robot tilted too much | 1.0 s | max_tilt=1.0 rad (~57°) | Episode fails (robot fell over) |
| 2 | `base_height` | Height out of limits | 1.0 s | min=0.1m, max=1.5m | Episode fails (robot too low or high) |

---

## Event Terms (Randomization) - 3 Reset + 1 Interval

### Reset Events (Run on Episode Reset)

| Event Name | Function | Mode | Parameters | Purpose |
|-----------|----------|------|------------|---------|
| `reset_base` | Random root state | reset | x∈(-0.5,0.5)m, y∈(-0.5,0.5)m, yaw∈(-π,π) | Randomize robot base position |
| `reset_robot_joints` | Random joint positions | reset | pos_range=(0.5, 1.5)× default | Randomize joint positions around default |

### Interval Events (Run Periodically During Episode)

| Event Name | Function | Mode | Interval | Parameters | Purpose |
|-----------|----------|------|----------|------------|---------|
| `push_robot` | Random external push | interval | (10.0, 15.0) s | velocity∈(-0.5,0.5) m/s | Robustness training: unexpected disturbances |

---

## Command Terms (2 Commands)

### Base Velocity Command

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Type** | UniformVelocityCommand | Random velocity commands |
| **Resampling Time** | 10.0 s | Constant for all environments |
| **Standing Environments** | 2% | Percent of envs with zero velocity |
| **Debug Visualization** | Yes | Green arrows show commanded velocity |
| **Linear Velocity X** | [-1.0, 1.0] m/s | Forward/backward range |
| **Linear Velocity Y** | [-0.5, 0.5] m/s | Left/right range |
| **Angular Velocity Z** | [-1.0, 1.0] rad/s | Rotation range |
| **Heading** | [-π, π] rad | Desired orientation range |

### Goal Position Command

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Type** | GoalPositionCommand | Random navigation goals |
| **Resampling Time** | 20.0-30.0 s | Random resampling interval |
| **Debug Visualization** | Yes | Cube markers show goal location |
| **Position X Range** | [-5.0, 5.0] m | Goal x-coordinate bounds |
| **Position Y Range** | [-5.0, 5.0] m | Goal y-coordinate bounds |
| **Position Z** | 0.0 m | Ground level (fixed) |

---

## Action Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Action Type** | JointPositionControl | PD controller set points |
| **Controlled Joints** | 12 leg joints | fl_hx/hy/kn, fr_hx/hy/kn, hl_hx/hy/kn, hr_hx/hy/kn |
| **Arm Status** | Fixed | Arm joints not controlled (held at initial pose) |
| **Action Scale** | 0.25 | Scales action to joint position range |
| **Action Dimension** | 12 | Output dimension of policy network |
| **Leg Stiffness** | 80 N·m/rad | PD controller stiffness |
| **Leg Damping** | 4 N·m·s/rad | PD controller damping |

---

## Observation Configuration

### Policy Observations (51 Dimensions)

| Dimension | Observation | Noise | Dimensions | Purpose |
|-----------|------------|-------|-----------|---------|
| 0-2 | Base linear velocity | ±0.1 m/s | 3 | Robot body velocity in base frame |
| 3-5 | Base angular velocity | ±0.2 rad/s | 3 | Robot body rotation rate |
| 6-8 | Projected gravity | ±0.05 | 3 | Gravity vector in base frame (for orientation) |
| 9-11 | Velocity commands | None | 3 | Current velocity command (vx, vy, ω) |
| 12-14 | Goal position | None | 3 | Goal location relative to base |
| 15-26 | Leg joint positions | ±0.01 rad | 12 | Relative to default (fl,fr,hl,hr × hx,hy,kn) |
| 27-38 | Leg joint velocities | ±1.0 rad/s | 12 | Angular velocity of leg joints |
| 39-50 | Previous actions | None | 12 | Last executed actions |

### Critic Observations (58 Dimensions)

| Component | Dims | Observations |
|-----------|------|--------------|
| **Policy Obs** | 51 | All of the above |
| **Additional** | 7 | base_pos (3) + goal_pos_w (3) + feet_contact (4) |

**Note:** Critic gets full state (no noise, world frame goal position) for better value function training.

---

## Network Architecture (RSL-RL PPO)

### Actor (Policy) Network

```
Input: 51 dims (policy observations)
  ↓
Linear(51 → 256) + ELU
  ↓
Linear(256 → 256) + ELU
  ↓
Linear(256 → 128) + ELU
  ↓
Linear(128 → 12)  [means]
  ↓
Output: 12 dims (joint position commands)

Std Dev: Learned, initialized to log(0.5)
```

### Critic (Value) Network

```
Input: 58 dims (critic observations)
  ↓
Linear(58 → 256) + ELU
  ↓
Linear(256 → 256) + ELU
  ↓
Linear(256 → 128) + ELU
  ↓
Linear(128 → 1)   [value]
  ↓
Output: 1 dim (scalar value estimate)
```

---

## Training Configuration (RSL-RL PPO)

| Hyperparameter | Value | Description |
|---|---|---|
| **Algorithm** | Proximal Policy Optimization (PPO) | Standard RL algorithm |
| **Learning Rate (Policy)** | 5e-4 | Adam optimizer learning rate |
| **Learning Rate (Critic)** | 5e-4 | Adam optimizer learning rate |
| **Clip Ratio** | 0.2 | PPO clipping parameter |
| **Entropy Coefficient** | 0.01 | Entropy regularization weight |
| **GAE Lambda** | 0.95 | Generalized Advantage Estimation |
| **Discount Factor (γ)** | 0.99 | Future reward discount |
| **N Epochs** | 5 | Gradient steps per rollout |
| **N Steps** | 24 | Steps per rollout collection |
| **Batch Size** | 24576 | Total samples per update (1024 envs × 24 steps) |
| **Max Iterations** | 50,000 | Training duration |

---

## Key Design Choices

### High-Level Strategy
1. **Two-Phase Approach**: Low-level locomotion → High-level manipulation
2. **Hierarchical Learning**: Decouple walking from arm control
3. **Command-Based Interface**: Velocity commands for intuitive control

### Reward Shaping Insights
- **Velocity Tracking (1.5 + 0.75)**: Primary objective - track commands
- **Goal Navigation (1.0 + 0.3)**: Secondary objective - reach goals
- **Base Height (0.5 - 1.0)**: Soft constraint - maintain upright posture
- **Penalties (-204.61)**: Heavy penalties for failures and energy waste

### Stability Measures
1. **Graceful Failure**: 1.0s grace period before termination (noise robustness)
2. **Energy Efficiency**: Joint torque and acceleration penalties
3. **Foot Contact**: Penalize slipping and undesired body contacts
4. **Height Constraints**: Prevent jumping or lying down

### Curriculum Elements
- **Position Randomization**: X,Y uniformly ±0.5m
- **Velocity Push**: Random disturbances every 10-15s
- **Joint Randomization**: 50-150% of default positions
- **No Terrain**: Flat ground for initial learning phase

---

## Performance Metrics

### Expected Learning Curves
- **Early (0-10k steps)**: High variance, agent learns basic walking
- **Mid (10-30k steps)**: Stable walking with velocity tracking
- **Late (30-50k steps)**: Refined gaits, goal-directed navigation

### Evaluation Metrics
1. **Command Tracking Error**: |v_actual - v_command|
2. **Goal Reaching Rate**: % episodes reaching goal
3. **Average Episode Return**: Sum of all rewards
4. **Joint Power**: Sum of |torque × velocity|
5. **Stability**: Episodes without termination

---

