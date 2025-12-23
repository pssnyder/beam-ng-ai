# BeamNG AI Driver: Phase 4 Training Configuration Guide

**Date**: October 19, 2025  
**Phase**: 4 - Imitation Learning & Behavior Cloning  
**Focus**: Training constraints, AI style, and neural network configuration  

---

## 📊 **INPUT CHANNELS SUMMARY** (142 Total Numeric + Advanced Sensors)

### **Core Input Categories**

#### **🚗 Vehicle State Channels (7 channels)**
- **Position**: World coordinates (x, y, z) 
- **Velocity**: 3D velocity vector (vx, vy, vz)
- **Speed**: Scalar speed magnitude

#### **🎮 Control Input Channels (3 channels)**  
- **Throttle**: 0.0 → 1.0 (acceleration input)
- **Steering**: -1.0 → 1.0 (left/right steering)
- **Brake**: 0.0 → 1.0 (braking force)

#### **⚙️ Vehicle Dynamics Channels (126 channels)**
- **Engine**: RPM, gear, fuel level, throttle response
- **Transmission**: Clutch ratio, gear selection
- **Wheels**: Individual wheel speeds, tire slip, grip
- **Suspension**: Compression, extension, spring forces
- **Safety Systems**: ABS, ESC, TCS activation states
- **Electrical**: Battery, alternator, lighting systems
- **Temperature**: Engine, transmission, brake temps
- **Aerodynamics**: Downforce, drag coefficients

#### **🔬 Physics Data Channels (6 channels)**
- **G-Forces**: 3-axis acceleration (gx, gy, gz)
- **Acceleration**: Longitudinal acceleration magnitude
- **Damage**: Overall damage level + critical components

#### **🎥 Advanced Sensor Channels (Ready for Integration)**
- **Camera**: 2,764,800 channels (1280×720×3 RGB) → CNN processed to feature vector
- **LiDAR**: Variable point cloud → Grid-based representation
- **GPS**: 6 channels (lat, lon, alt, accuracy, velocity, heading)
- **IMU**: 12 channels (3D accel, gyro, magnetometer, temperature)

---

## 🎯 **CORE/CRITICAL REWARDS SYSTEM**

### **Primary Reward Components (6 Core)**

#### **1. Progress Reward (+)** - *Primary Objective*
```python
progress_reward = distance_along_track * speed_efficiency_multiplier
weight = 1.0  # Highest priority
```
- Encourages forward movement along intended path
- Speed efficiency bonus for maintaining optimal velocity
- **Training Impact**: Drives primary goal achievement

#### **2. Path Deviation Penalty (-)** - *Safety Critical*
```python
path_penalty = (lateral_deviation² + heading_error²) * severity_multiplier  
weight = 2.0  # Heavy penalty
```
- Quadratic penalty increases sharply for larger deviations
- Heading error prevents wrong-direction driving
- **Training Impact**: Keeps AI on intended trajectory

#### **3. Control Smoothness Reward (+)** - *Driving Quality*
```python
smoothness_reward = 1.0 / (1.0 + steering_jerk + throttle_rate_change)
weight = 0.6  # Quality enhancement
```
- Penalizes jerky, erratic control inputs
- Rewards human-like smooth driving behavior
- **Training Impact**: Produces comfortable, realistic driving

#### **4. Speed Optimization (+/-)** - *Performance Balance*
```python
speed_reward = optimal_speed - abs(current_speed - target_speed_for_section)
weight = 0.8  # High importance
```
- Context-aware speed targets (corners vs. straights)
- Balances speed with safety considerations
- **Training Impact**: Teaches appropriate speed for conditions

#### **5. Damage Avoidance Penalty (-)** - *Safety Critical*
```python
damage_penalty = damage_rate * collision_severity * safety_multiplier
weight = 5.0  # Maximum penalty
```
- Severe penalty for any vehicle damage
- Immediate negative feedback for risky behavior
- **Training Impact**: Strong safety-first behavior

#### **6. Efficiency Reward (+)** - *Performance Optimization*
```python
efficiency_reward = (baseline_time - current_time) / baseline_time
weight = 0.4  # Performance enhancement
```
- Rewards faster completion times
- Normalized against baseline performance
- **Training Impact**: Encourages optimal time performance

### **Reward Function Architecture**
```python
total_reward = (
    progress_reward * 1.0 +           # Primary driving objective
    speed_reward * 0.8 +              # Speed optimization
    smoothness_reward * 0.6 +         # Control quality  
    efficiency_reward * 0.4 -         # Time performance
    path_penalty * 2.0 -              # Stay on track (critical)
    damage_penalty * 5.0              # Avoid crashes (critical)
)

# Reward bounds: [-10.0, +3.8] typical range
# Positive rewards: Good driving behavior
# Negative rewards: Unsafe/poor driving behavior
```

---

## 🏗️ **TRAINING PIPELINE ARCHITECTURE**

### **Neural Network Architecture (Behavior Cloning → RL)**

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER (142 channels)               │
├─────────────────────────────────────────────────────────────┤
│  Vehicle State (7)  │  Controls (3)  │  Dynamics (126)  │  Physics (6)  │
│        ↓            │       ↓        │        ↓         │      ↓        │
│   Dense(32)         │  Dense(16)     │   Dense(64)      │  Dense(16)    │
│        ↓            │       ↓        │        ↓         │      ↓        │  
│   Dense(64)         │  Dense(32)     │   Dense(128)     │  Dense(32)    │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE FUSION LAYER                       │
│              Concatenated Features (256) → Dense(128)       │
└─────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    ACTION HEAD (3 outputs)                  │
│  Throttle (Sigmoid)  │  Steering (Tanh)  │  Brake (Sigmoid) │
│      [0.0, 1.0]      │    [-1.0, 1.0]    │    [0.0, 1.0]    │
└─────────────────────────────────────────────────────────────┘
```

### **Training Framework Configuration**
```python
# Phase 4: Behavior Cloning (Supervised Learning)
behavior_cloning_config = {
    "algorithm": "Supervised Learning",
    "loss_function": "MSE + L2 Regularization",
    "optimizer": "Adam (lr=0.001)",
    "batch_size": 32,
    "epochs": 50-100,
    "validation_split": 0.2,
    "l2_weight": 0.001
}

# Phase 5: Reinforcement Learning (PPO)
rl_training_config = {
    "algorithm": "Proximal Policy Optimization (PPO)", 
    "framework": "Stable Baselines 3",
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2
}
```

---

## 🔄 **TRAINING PIPELINE WORKFLOW**

### **Phase 4: Imitation Learning Pipeline**

```
Step 1: Human Demonstration Collection
├── Scenario Setup: west_coast_usa highway + gridmap_v2 urban
├── Recording: 40,000+ samples across 3 driving scenarios  
├── Quality Control: Smoothness validation, trajectory verification
└── Data Storage: HDF5 format with metadata

Step 2: Data Preprocessing  
├── Normalization: Min-max scaling for all input channels
├── Feature Engineering: Speed derivatives, steering rates
├── Augmentation: Lateral shifts (±0.5m), speed scaling (±10%)
└── Train/Validation Split: 80/20 random split

Step 3: Behavior Cloning Training
├── Network Initialization: Xavier uniform weight initialization
├── Training Loop: 50-100 epochs with early stopping
├── Validation: Real-time loss monitoring, overfitting detection
└── Model Checkpoints: Best validation loss + final epoch

Step 4: Model Validation
├── Simulation Testing: 5+ minute autonomous driving
├── Behavioral Metrics: >90% correlation with human demo
├── Performance Metrics: <0.05 validation loss, <10ms inference
└── Safety Validation: No collision during test runs
```

### **Phase 5: Reinforcement Learning Pipeline**

```
Step 1: Environment Integration
├── Gymnasium Wrapper: BeamNG → RL environment interface
├── Action Space: Continuous control (throttle, steering, brake)
├── Observation Space: 142-channel state vector
└── Reward Function: 6-component reward system

Step 2: PPO Training Loop
├── Policy Initialization: Pre-trained behavior cloning weights
├── Experience Collection: 2048 steps per training iteration
├── Policy Update: PPO algorithm with clipped objective
└── Performance Monitoring: Reward tracking, loss metrics

Step 3: Advanced Training
├── Curriculum Learning: Simple → Complex scenarios
├── Domain Randomization: Weather, lighting, traffic variations
├── Multi-Environment: Different maps and driving challenges
└── Hyperparameter Tuning: Learning rate scheduling, batch sizes
```

---

## 🎮 **TRAINING CONSTRAINTS & AI STYLE**

### **Behavioral Constraints**

#### **Safety-First Training Style**
- **Collision Avoidance**: Maximum priority (-5.0 reward weight)
- **Speed Limits**: Context-appropriate speed management
- **Conservative Decision Making**: Prefer safe over aggressive choices
- **Smooth Control**: Human-like input smoothness (jerk penalties)

#### **Performance-Oriented Style** 
- **Track Optimization**: Learns optimal racing lines
- **Speed Management**: Balances safety with performance
- **Predictive Behavior**: Anticipates upcoming track sections
- **Efficiency Focus**: Minimizes lap times while maintaining safety

#### **Robustness Requirements**
- **Generalization**: Performs across different maps/scenarios
- **Disturbance Handling**: Recovers from unexpected situations
- **Real-time Performance**: <10ms inference latency
- **Stability**: Consistent behavior across training sessions

### **Training Methodology**

#### **Phase 4: Conservative Imitation**
- Train on smooth, safe human demonstrations
- Emphasize trajectory following and speed control
- Build foundational driving skills
- Establish safety baselines

#### **Phase 5: Performance Optimization** 
- Use RL to optimize beyond human demonstration
- Explore speed/safety trade-offs within constraints
- Learn advanced techniques (trail braking, optimal lines)
- Maximize performance while maintaining safety margins

---

## 📈 **SUCCESS METRICS & VALIDATION**

### **Training Success Criteria**

#### **Phase 4 Targets**
- [ ] **Data Collection**: 40,000+ demonstration samples
- [ ] **Model Performance**: <0.05 validation loss
- [ ] **Behavioral Similarity**: >90% correlation with human
- [ ] **Real-time Operation**: 5+ minutes autonomous driving
- [ ] **Control Quality**: Smooth inputs (jerk < 2.0 rad/s³)

#### **Phase 5 Targets**
- [ ] **RL Performance**: Consistent reward improvement
- [ ] **Lap Time Optimization**: 10%+ improvement over baseline
- [ ] **Safety Maintenance**: Zero collisions during training
- [ ] **Generalization**: Success across multiple environments
- [ ] **Robustness**: Handles disturbances and edge cases

### **Evaluation Framework**
```python
evaluation_metrics = {
    "safety": ["collision_rate", "damage_frequency", "unsafe_maneuvers"],
    "performance": ["lap_time", "speed_efficiency", "trajectory_accuracy"], 
    "control_quality": ["steering_smoothness", "throttle_consistency", "brake_efficiency"],
    "behavioral": ["human_similarity", "decision_confidence", "reaction_time"]
}
```

---

**Status**: Ready for Phase 4 Implementation  
**Next Action**: Begin human demonstration collection system development  
**Target**: Complete behavior cloning foundation within 2 weeks