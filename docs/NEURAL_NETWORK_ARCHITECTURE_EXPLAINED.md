# BeamNG AI Neural Network Architecture - Complete Guide

**Created**: December 27, 2025  
**For**: Understanding how the AI driver learns and what it "sees" vs what it "does"

---

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [What the AI "Sees" (Input Sensors)](#what-the-ai-sees-input-sensors)
3. [What the AI "Does" (Output Actions)](#what-the-ai-does-output-actions)
4. [The Brain (Neural Network Architecture)](#the-brain-neural-network-architecture)
5. [How It Learns (Training Process)](#how-it-learns-training-process)
6. [Why It Crashes (Current Behavior Analysis)](#why-it-crashes-current-behavior-analysis)
7. [Data Science Decision Points](#data-science-decision-points)
8. [Next Steps & Improvements](#next-steps--improvements)

---

## High-Level Overview

**What we're building**: An AI that learns to drive on highways by trial and error (reinforcement learning)

**The Learning Loop**:
```
1. AI observes environment (27 sensor values)
2. AI decides action (throttle, steering, brake)
3. BeamNG executes action for 0.3 seconds
4. AI gets reward based on outcome
5. AI adjusts its brain to maximize future rewards
6. Repeat 100,000+ times
```

**Current Algorithm**: SAC (Soft Actor-Critic)
- Type: Model-free reinforcement learning
- Specialty: Continuous control (smooth throttle/steering, not discrete buttons)
- Advantages: Sample efficient, stable, works well for robotics/driving

---

## What the AI "Sees" (Input Sensors)

The AI receives **27 numbers** every 0.3 seconds. These are its "senses":

### State Vector Breakdown (27 dimensions)

```python
TrainingState:
    # ===== POSITION & MOTION (6 values) =====
    position: [x, y, z]           # 3D coordinates in world space
    velocity: [vx, vy, vz]        # 3D velocity vector (m/s)
    
    # ===== DISTANCE TRACKING (3 values) =====
    distance_from_origin: float      # Total distance from start (meters)
    distance_from_checkpoint: float  # Distance since last crash (meters)
    distance_delta: float            # Change in distance this step (m/s)
    
    # ===== VEHICLE DYNAMICS (15 values) =====
    speed: float                  # Current speed (m/s)
    throttle: float               # Current throttle input [0-1]
    steering: float               # Current steering input [-1 to 1]
    brake: float                  # Current brake input [0-1]
    rpm: float                    # Engine RPM
    gear: float                   # Current gear (automatic transmission)
    wheelspeed: float             # Average wheel speed (m/s)
    
    # G-Forces (what the driver would "feel")
    gx: float                     # Lateral G-force (left/right)
    gy: float                     # Longitudinal G-force (forward/back)
    gz: float                     # Vertical G-force (up/down, -1 = upright)
    
    # Vehicle Health & Safety Systems
    damage: float                 # Cumulative damage [0-1]
    abs_active: float             # Anti-lock brakes active [0/1]
    esc_active: float             # Electronic stability control [0/1]
    tcs_active: float             # Traction control active [0/1]
    fuel: float                   # Remaining fuel [0-1]
    
    # ===== EPISODE METADATA (3 values) =====
    episode_time: float           # Seconds since episode start
    crash_count: float            # Number of crashes this episode
    stationary_time: float        # Time spent not moving (seconds)
```

### What's MISSING (Critical Observation!)

**The AI is essentially driving BLIND**:
- ❌ No camera/vision input
- ❌ No road boundaries or lane markers
- ❌ No upcoming obstacles or curve detection
- ❌ No distance to road edges
- ❌ No heading relative to road direction

**This is like asking a human to drive while only looking at the dashboard!**

Current sensors tell the AI:
- ✅ "You're moving at 15 m/s"
- ✅ "You're pulling 0.5 lateral Gs"
- ✅ "Damage is 0.2"

But NOT:
- ❌ "There's a curve 50m ahead"
- ❌ "You're 2m from the road edge"
- ❌ "You're heading 30° off road direction"

---

## What the AI "Does" (Output Actions)

The AI outputs **3 continuous values** every 0.3 seconds:

### Action Space (3 dimensions)

```python
Action Vector: [throttle, steering, brake]

throttle: float [0.0 to 1.0]
    - 0.0 = no gas
    - 0.5 = half throttle
    - 1.0 = full throttle
    
steering: float [-1.0 to 1.0]
    - -1.0 = full left
    -  0.0 = straight
    -  1.0 = full right
    
brake: float [0.0 to 1.0]
    - 0.0 = no brake
    - 0.5 = half brake
    - 1.0 = full brake
```

### What the AI CANNOT Control (Automatic Management)

These are handled automatically by the environment:
- ✅ Parking brake (always released)
- ✅ Gear shifting (automatic transmission)
- ✅ Clutch (N/A - automatic)
- ✅ Lights, blinkers, horn (not relevant for highway)
- ✅ Engine start/stop (always running)

**Design Decision**: Keep action space small (3D) for faster learning. Manual transmission would be 5D+ and much harder.

---

## The Brain (Neural Network Architecture)

We use **SAC (Soft Actor-Critic)** with 3 neural networks:

### 1. Actor Network (The "Driver")
**Job**: Decide what action to take given current state

```
Input: 27 state values
  ↓
Hidden Layer 1: 256 neurons (ReLU activation)
  ↓
Hidden Layer 2: 256 neurons (ReLU activation)
  ↓
Output: 6 values (mean and log_std for 3 actions)
  ↓
Sampling: Use Gaussian distribution to add exploration noise
  ↓
Final Action: [throttle, steering, brake] via tanh activation
```

**Total Parameters**: ~67,000 trainable weights

**Key Feature**: Stochastic policy
- Outputs a probability distribution, not a single action
- During training: samples from distribution (exploration)
- During deployment: uses mean of distribution (exploitation)

### 2. Critic Networks (The "Coach" - 2 networks)
**Job**: Estimate "how good is this state-action pair?"

```
Input: 27 state values + 3 action values = 30 total
  ↓
Hidden Layer 1: 256 neurons (ReLU)
  ↓
Hidden Layer 2: 256 neurons (ReLU)
  ↓
Output: 1 value (Q-value = expected future reward)
```

**Why 2 critics?** Reduces overestimation bias (Double Q-Learning trick)

**Total Parameters**: ~67,000 each = 134,000 total

### Target Networks
- Slow-moving copies of critic networks
- Updated gradually (τ=0.005 per step)
- Provides stable learning targets

### Temperature Parameter (α)
- Auto-tuned during training
- Balances exploration vs exploitation
- Higher α = more random exploration
- Lower α = more deterministic behavior

---

## How It Learns (Training Process)

### Phase 1: Random Exploration (First 1000 steps)
```python
# AI doesn't use its brain yet - pure random actions
action = [
    random(0.7, 1.0),   # High throttle
    random(-0.4, 0.4),  # Random steering
    random(0, 0.02)     # Minimal brake
]
```

**Purpose**: Fill experience replay buffer with diverse scenarios
**Current Issue**: Too aggressive - just accelerates into first obstacle!

### Phase 2: Learning from Experience (After 1000 steps)

Every 2 steps:
1. Sample 256 random experiences from buffer
2. Calculate TD error: `reward + γ * Q(next_state, next_action) - Q(state, action)`
3. Update critics to minimize TD error
4. Update actor to maximize Q-value
5. Update temperature to maintain entropy

### Experience Replay Buffer
- Capacity: 100,000 experiences
- Each experience: `(state, action, reward, next_state, done)`
- Random sampling breaks correlation between consecutive steps

### Reward Function (The "Report Card")

```python
reward = 0.0

# PRIMARY GOAL: Distance traveled
if made_progress:
    reward += distance_traveled * 0.5    # 0.5 points per meter

# SPEED BONUS: Go fast while making progress
if moving_forward and speed > 1.0:
    reward += speed * 0.1                # 0.1 points per m/s

# CRASH PENALTY: Damage detection
if damage_increased > 0.05:
    reward -= 50.0                       # Big penalty

# CRASH PENALTY: Orientation detection
if flipped_or_tilted:
    reward -= 30.0                       # Medium penalty

# STATIONARY PENALTY: Don't just sit there
if speed < 0.5:
    reward -= 0.5 per step               # Small continuous penalty
    if stationary > 3 seconds:
        reward -= 20.0                   # Bigger penalty + force reset
```

**Example Scenarios**:
- Drive straight 10m at 15 m/s = +5 + 1.5 = **+6.5 points**
- Crash into wall = **-50 points**
- Sit stationary 5 seconds = -0.5 * 15 steps - 20 = **-27.5 points**

---

## Why It Crashes (Current Behavior Analysis)

### Root Cause: The AI is Blind

**What's happening**:
1. AI starts episode, sees state: `[x, y, z, vx, vy, vz, ...]`
2. Random exploration says: "Throttle = 0.9, Steering = 0.1"
3. Car accelerates forward, state updates
4. AI gets small positive reward for distance traveled
5. **AI has no idea there's a wall/obstacle ahead**
6. Keeps accelerating because distance reward is positive
7. CRASH - gets -50 reward
8. Too late - already in a bad state

**The fundamental problem**: 
- Sensors tell AI where it IS, not where it's GOING
- Reward is based on outcome, not intent
- By the time AI gets negative reward, it's already crashed

### Why It Gets Stuck

**Scenario**: Car crashes, rolls, lands on wheels in grass
1. `damage = 0.8` (high)
2. `speed = 0.2` (barely moving)
3. AI chooses action: `[0.9, 0.0, 0.0]` (full throttle, no steering)
4. Wheels spin in grass, no movement
5. Stationary timer increases
6. After 3 seconds: should trigger recovery
7. **BUG**: If `speed = 0.2` (not quite 0), doesn't trigger timeout!

### Why It Doesn't Explore Other Actions

**Current random exploration**:
```python
throttle: uniform(0.7, 1.0)   # ALWAYS high!
steering: uniform(-0.4, 0.4)  # Limited turning
brake: uniform(0, 0.02)       # Almost never brakes
```

**Problem**: Not truly random - heavily biased toward "gas pedal down"
- Never learns: "What if I brake before crashing?"
- Never learns: "What if I slow down in grass?"
- Never learns: "What if I reverse out of trouble?"

---

## Data Science Decision Points

### Decision 1: State Representation (Inputs)
**Question**: What should the AI "see"?

**Options**:
| Option | Pros | Cons | Our Choice |
|--------|------|------|------------|
| Raw telemetry (current) | Simple, fast, precise numbers | Blind to environment | ✅ Phase 4 |
| Camera images | Sees road/obstacles | Slow, requires CNNs | ❌ Too complex |
| Hybrid (sensors + vision) | Best of both worlds | Complex architecture | 🎯 Phase 5 goal |

**Recommendation**: Add engineered features:
- Distance to road edges (raycast sensors)
- Heading error (angle to road direction)
- Curvature ahead (road geometry)
- Time-to-collision estimates

### Decision 2: Action Space (Outputs)
**Question**: How much control does AI get?

**Our choice**: 3D continuous (throttle, steering, brake)
- ✅ Simple enough to learn quickly
- ✅ Matches human intuition
- ❌ Missing emergency actions (reverse, parking brake)

**Alternative**: 
- Discrete actions: ["accelerate", "brake", "turn_left", "turn_right"]
- Simpler but less smooth control

### Decision 3: Reward Function
**Question**: How do we measure "good driving"?

**Current design**:
```python
reward = distance * 0.5 + speed * 0.1 - crashes * 50 - stationary * 0.5
```

**Issues**:
- ❌ Encourages reckless speed
- ❌ No penalty for jerky steering
- ❌ No reward for staying on road

**Better design**:
```python
reward = (
    distance * 0.5                    # Primary goal
    + speed_on_road * 0.2             # Speed bonus only on pavement
    - distance_from_center * 0.1      # Stay in lane
    - abs(steering_delta) * 0.05      # Smooth steering
    - crashes * 100                   # Bigger crash penalty
    + time_without_crash * 0.1        # Reward survival
)
```

### Decision 4: Exploration Strategy
**Question**: How does AI discover new behaviors?

**Current**: Random uniform actions (0-1000 steps)
- Problem: Too biased toward full throttle

**Better options**:
1. **Gaussian noise on policy**: `action = policy(state) + noise`
2. **Curriculum learning**: Start with slow speeds, gradually increase
3. **Entropy regularization**: Reward trying diverse actions (SAC does this!)

### Decision 5: Training Hyperparameters

| Parameter | Current Value | What It Controls | Tuning Impact |
|-----------|---------------|------------------|---------------|
| `batch_size` | 256 | Experiences per update | Higher = more stable, slower |
| `learning_rate` | 3e-4 | How fast brain changes | Higher = faster but unstable |
| `gamma` | 0.99 | Future reward discount | Higher = long-term thinking |
| `tau` | 0.005 | Target network update rate | Higher = less stable targets |
| `replay_start_size` | 1000 | Random exploration steps | Higher = more diverse data |
| `buffer_capacity` | 100,000 | Experience memory size | Higher = more diverse sampling |

**Current bottleneck**: Only 1000 random exploration steps
- Not enough diversity before training starts
- Increase to 5000-10000 for better coverage

---

## Next Steps & Improvements

### Immediate Fixes (This Session)

1. **Better Damage Reset**
```python
# Add to environment
def reset_vehicle_damage(self):
    """Reset damage without moving vehicle"""
    self.vehicle.set_part_config(...)  # BeamNG API call
```

2. **Faster Scenario Management**
```python
# Don't reload scenario between episodes
def soft_reset(self):
    """Reset vehicle state only, not entire scenario"""
    self.vehicle.teleport(self.checkpoint_position)
    self.vehicle.set_velocity([0, 0, 0])
    # No scenario.restart() needed!
```

3. **Improved Stuck Detection**
```python
# More sensitive timeout
if speed < 1.0:  # Increased from 0.5
    stationary_timer += dt
    if stationary_timer > 2.0:  # Faster timeout (was 3.0)
        recover()
```

4. **Balanced Random Exploration**
```python
# Better distribution
throttle = random(0.0, 1.0)    # Full range, not biased
steering = random(-1.0, 1.0)   # Full steering range
brake = random(0.0, 0.3)       # Sometimes brake!
```

### Phase 5 Goals (Feature Engineering)

Add these sensors to state vector:
- **Road detection**: Distance to left/right edges (raycasts)
- **Heading alignment**: Angle between car and road direction
- **Curvature lookahead**: Is there a turn coming?
- **Surface type**: Pavement vs grass/dirt
- **Obstacle detection**: Something in front? (if applicable)

Expand state from 27 → 40+ dimensions

### Phase 6 Goals (Advanced Training)

- **Curriculum learning**: Start on straight roads, progress to curves
- **Multi-scenario training**: Train on different maps
- **Transfer learning**: Pre-train on simple tasks
- **Hierarchical RL**: High-level (path planning) + low-level (control)

---

## Quick Reference: Current Model Stats

```
INPUTS:  27 dimensions (position, velocity, dynamics, metadata)
OUTPUTS: 3 dimensions (throttle, steering, brake)

NEURAL NETWORKS:
  - Actor: 27 → 256 → 256 → 6 (mean + std for 3 actions)
  - Critic1: 30 → 256 → 256 → 1 (Q-value)
  - Critic2: 30 → 256 → 256 → 1 (Q-value)
  - Total Trainable Parameters: ~268,000

TRAINING:
  - Algorithm: Soft Actor-Critic (SAC)
  - Experience Replay: 100,000 capacity
  - Batch Size: 256
  - Learning Rate: 0.0003
  - Update Frequency: Every 2 steps (after 1000 random steps)
  
REWARD FUNCTION:
  + Distance traveled * 0.5
  + Speed (when moving forward) * 0.1
  - Crash damage * 50
  - Flip/tilt * 30
  - Stationary * 0.5/step (+ 20 after 3s timeout)

CONTROL LOOP:
  - Frequency: 3.3 Hz (0.3s per step)
  - Episode length: 200 steps max (~60 seconds)
  - Reset time: 1 second (scenario.restart)
```

---

## Learning Resources

**Reinforcement Learning Fundamentals**:
- [Spinning Up in Deep RL](https://spinningup.openai.com/) - Best RL intro
- [SAC Paper](https://arxiv.org/abs/1801.01290) - Original algorithm

**Driving-Specific RL**:
- [Learning to Drive in a Day](https://arxiv.org/abs/1807.00412)
- [End-to-End Deep Learning for Self-Driving](https://arxiv.org/abs/1604.07316)

**BeamNG for ML**:
- [BeamNGpy Documentation](https://beamngpy.readthedocs.io/)
- [BeamNG Research Examples](https://github.com/BeamNG/BeamNGpy/tree/master/examples)

---

**Questions? Ask about**:
- Why we chose SAC over other algorithms (PPO, DQN, A3C)
- How to visualize what the AI is "thinking"
- Trade-offs between sample efficiency and final performance
- How to debug reward function issues
- When to use curriculum learning vs end-to-end training
