# **BeamNG AI Driver - Phase 3: Neural Network Training Blueprint**

**Project:** Data-Driven Driver (Maximum Telemetry Access)  
**Phase:** 3 - Comprehensive Telemetry and State Capture  
**Date:** October 19, 2025  
**Status:** Planning & Implementation  

---

## **📊 Input Channels Configuration**

### **Vehicle State Channels (6 primary)**
- **Position**: `pos` - (x, y, z) world coordinates [3 channels]
- **Velocity**: `vel` - Linear velocity vector [3 channels] 
- **Rotation**: `rotation` - Quaternion (x, y, z, w) [4 channels]
- **Angular Velocity**: `angular_vel` - Rotational rates [3 channels]
- **Direction Vector**: `dir` - Forward direction [3 channels]
- **Up Vector**: `up` - Vehicle up direction [3 channels]

**Subtotal: 19 channels**

### **Vehicle Dynamics Channels (Electrics - 126 available)**
- **Core Dynamics** (12 channels):
  - `throttle`, `brake`, `steering` - Control inputs [3]
  - `wheelspeed`, `airspeed` - Speed metrics [2] 
  - `rpm`, `gear` - Engine state [2]
  - `fuel` - Fuel level [1]
  - `clutch_ratio` - Transmission [1]
  - `abs_active`, `esc_active`, `tcs_active` - Safety systems [3]

- **Physics Forces** (9 channels):
  - G-Forces: `gx`, `gy`, `gz` - Acceleration forces [3]
  - Wheel forces: Front/rear force distribution [4]
  - Suspension: Compression/extension states [2]

- **Vehicle Health** (4 channels):
  - Overall damage level [1]
  - Critical component status [3]

**Subtotal: 25 high-priority channels (101 additional available)**

### **Advanced Sensor Channels**
- **Camera Vision**: RGB image data [1280x720x3 = 2,764,800 values → CNN processed]
- **LiDAR Point Cloud**: Distance + intensity data [~100,000 points → processed to grid]
- **GPS Positioning**: Precise location + heading [4 channels]
- **Advanced IMU**: High-frequency acceleration + gyroscope [6 channels]

**Subtotal: CNN-processed visual + 10 numeric channels**

### **Derived/Calculated Channels (Phase 5)**
- **Time-to-Collision (TTC)**: Collision prediction [1 channel]
- **Path Deviation**: Distance/angle from ideal line [2 channels] 
- **Steering Smoothness**: Rate of change metrics [2 channels]
- **Road Friction Estimate**: Calculated from wheel slip [1 channel]

**Subtotal: 6 channels**

---

## **🎯 Core/Critical Rewards System**

### **Primary Reward Components**

#### **1. Progress Reward (+)**
```python
progress_reward = distance_along_track * speed_bonus_multiplier
# Encourages forward movement and maintaining speed
```

#### **2. Path Deviation Penalty (-)**
```python
path_penalty = path_deviation_distance² * deviation_weight
# Quadratic penalty for leaving optimal racing line
```

#### **3. Speed Optimization (+/-)**
```python
speed_reward = optimal_speed_for_corner - abs(current_speed - target_speed)
# Rewards maintaining appropriate speed for track section
```

#### **4. Smoothness Reward (+)**
```python
smoothness_reward = 1.0 / (1.0 + steering_rate_change + throttle_jerk)
# Rewards smooth, human-like control inputs
```

#### **5. Damage Penalty (-)**
```python
damage_penalty = damage_rate * severity_multiplier
# Heavy penalty for collisions and damage
```

#### **6. Time Efficiency (+)**
```python
time_reward = baseline_time - current_time
# Rewards faster lap times compared to baseline
```

### **Reward Function Architecture**
```python
total_reward = (
    progress_reward * 1.0 +           # Primary objective
    speed_reward * 0.8 +              # Speed optimization  
    smoothness_reward * 0.6 +         # Control quality
    time_reward * 0.4 -               # Efficiency
    path_penalty * 2.0 -              # Stay on track
    damage_penalty * 5.0              # Avoid crashes
)
```

---

## **🏗️ Training Pipeline Architecture**

### **Neural Network Architecture (Planned)**

```
Input Layer:
├── Vehicle State (19 channels) → Dense Layer (64 neurons)
├── Vehicle Dynamics (25 channels) → Dense Layer (32 neurons)  
├── Sensor Data (10 channels) → Dense Layer (16 neurons)
├── Camera Feed (720p RGB) → CNN Backbone → Feature Vector (128)
└── LiDAR Cloud (processed) → Point Cloud Net → Feature Vector (64)

Feature Fusion:
└── Concatenated Features (304 total) → Dense (256) → Dense (128)

Action Head:
└── Output Layer (3 neurons): [throttle, steering, brake]
    ├── Throttle: Sigmoid activation (0.0 → 1.0)
    ├── Steering: Tanh activation (-1.0 → 1.0) 
    └── Brake: Sigmoid activation (0.0 → 1.0)
```

### **Training Framework**
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: Stable Baselines 3
- **Environment**: Custom BeamNG Gym wrapper
- **Update Frequency**: 2048 steps per update
- **Learning Rate**: 3e-4 with decay

---

## **🔄 Training Pipeline Workflow**

### **Phase 3: Data Collection Setup**
1. **Environment Setup**
   - ✅ BeamNG-Python connection established
   - ✅ Vehicle physics and controls working
   - 🔄 Advanced sensor integration (Camera, LiDAR, GPS, IMU)
   - 🔄 High-frequency telemetry polling (up to 2000Hz)

2. **Data Pipeline Creation**
   - 🔄 Real-time sensor data aggregation
   - 🔄 Data buffering system for high-frequency updates
   - 🔄 Feature preprocessing and normalization
   - 🔄 Data logging for offline analysis

### **Phase 4: Imitation Learning Foundation**
3. **Human Demonstration Collection**
   - Record expert driving on simple track
   - Capture input-output pairs for supervised learning
   - Create baseline performance metrics

4. **Behavior Cloning**
   - Train initial policy on human demonstrations
   - Establish basic driving behavior
   - Validate policy can maintain control

### **Phase 5: Feature Engineering** 
5. **Derived Metrics Implementation**
   - Time-to-collision calculation
   - Path deviation measurement
   - Friction estimation
   - Control smoothness metrics

### **Phase 6: Reinforcement Learning**
6. **RL Environment Integration**
   - Implement Gymnasium wrapper
   - Define action/observation spaces
   - Integrate reward function

7. **PPO Training Loop**
   - Initialize policy from behavior cloning
   - Iterative policy improvement
   - Performance monitoring and logging

---

## **📋 Implementation Checklist - Phase 3**

### **Advanced Sensors** 
- [ ] Camera sensor with high resolution capture
- [ ] LiDAR 360-degree point cloud collection  
- [ ] GPS precise positioning system
- [ ] Advanced IMU high-frequency motion data
- [ ] Sensor data synchronization system

### **Data Infrastructure**
- [ ] High-frequency data buffer (2000Hz capability)
- [ ] Real-time data preprocessing pipeline
- [ ] Feature normalization and scaling
- [ ] Data logging and replay system
- [ ] Memory-efficient data structures

### **Performance Optimization**
- [ ] Shared memory usage for large sensor data
- [ ] Asynchronous data collection
- [ ] CPU/GPU load balancing
- [ ] Data compression for storage

### **Testing and Validation**
- [ ] Sensor calibration verification
- [ ] Data quality monitoring
- [ ] System performance benchmarking
- [ ] Error handling and recovery

---

## **🎯 Success Metrics - Phase 3**

### **Data Collection Metrics**
- [ ] Achieve 2000Hz physics data polling
- [ ] Maintain <10ms sensor data latency
- [ ] Collect 100+ telemetry channels simultaneously
- [ ] Process camera frames at 20+ FPS
- [ ] Generate LiDAR point clouds at 10+ Hz

### **System Performance**
- [ ] <5% CPU overhead for data collection
- [ ] <1GB RAM usage for buffering
- [ ] 99%+ data collection reliability
- [ ] Real-time processing capability

---

## **📝 Notes and Observations**

### **Current Status** (Updated: Oct 19, 2025)
- ✅ Phase 1: BeamNG connection established
- ✅ Phase 2: Basic vehicle control and sensors working
- 🔄 Phase 3: Advanced sensor integration starting

### **Key Insights**
- BeamNGpy provides extensive telemetry access (126+ electrics channels)
- Vehicle physics simulation is highly accurate and responsive
- Sensor data quality is excellent for ML training
- Real-time performance is achievable with proper optimization

### **Technical Decisions**
- Using west_coast_usa map for reliable spawn points and varied terrain
- Implementing shared memory for high-bandwidth sensor data
- Prioritizing data quality over collection frequency initially
- Building modular sensor system for easy expansion

---

**Next Update**: After Phase 3 advanced sensor implementation