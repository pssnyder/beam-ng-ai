# BeamNG AI Driver: Exploratory Self-Learning Configuration

**Date**: October 19, 2025  
**Approach**: Policy-Based Reinforcement Learning with Minimal Constraints  
**Philosophy**: Self-Discovery Through Exploration → Progressive Policy Expansion  

---

## 🧠 **TRAINING PHILOSOPHY: EXPLORATORY SELF-LEARNING**

### **Core Principle: "Keep Moving, Don't Crash"**
- **No human demonstration** - AI discovers driving strategies independently
- **Minimal initial constraints** - Maximum exploration freedom
- **Progressive complexity** - Add heuristics gradually as AI masters basics
- **Open environment** - Infinite simulation with dynamic element rendering
- **Neural network visualization** - Live brain activity monitoring during training

---

## 📊 **ENHANCED INPUT CHANNELS** (160+ Total Numeric)

### **Primary Vehicle State (7 channels)**
- Position (x, y, z), Velocity (vx, vy, vz), Speed (scalar)

### **Control Inputs (3 channels)**
- Throttle [0.0-1.0], Steering [-1.0-1.0], Brake [0.0-1.0]

### **Enhanced Vehicle Dynamics (140+ channels)**
#### **Core Dynamics (12 channels)**
- `throttle`, `brake`, `steering`, `wheelspeed`, `airspeed`, `rpm`, `gear`, `fuel`, `clutch_ratio`, `abs_active`, `esc_active`, `tcs_active`

#### **Advanced Engine Telemetry (15 channels)**
- **Turbo System**: `turbo_pressure`, `turbo_rpm`, `boost_pressure`, `wastegate_position`
- **Engine Health**: `engine_temp`, `oil_pressure`, `oil_temp`, `coolant_temp`
- **Performance**: `torque_output`, `power_output`, `efficiency_rating`
- **Emissions**: `exhaust_temp`, `lambda_sensor`, `catalytic_converter_temp`
- **Timing**: `ignition_timing`, `cam_timing`

#### **Transmission & Drivetrain (12 channels)**
- **Transmission**: `transmission_temp`, `torque_converter_slip`, `shift_pressure`, `gear_ratio`
- **Differential**: `differential_temp`, `diff_lock_status`, `torque_distribution`
- **Drivetrain**: `cv_joint_wear`, `driveshaft_vibration`, `transfer_case_temp`
- **Clutch**: `clutch_temp`, `clutch_wear`, `clutch_engagement`

#### **Advanced Suspension & Handling (18 channels)**
- **Suspension**: Individual wheel `compression`, `rebound`, `spring_force`, `damper_velocity` [4x4=16]
- **Stability**: `roll_angle`, `pitch_angle`

#### **Tire & Grip Dynamics (16 channels)**
- **Tire Status**: Individual tire `pressure`, `temperature`, `wear`, `grip_coefficient` [4x4=16]

#### **Safety & Driver Assistance (8 channels)**
- **Active Safety**: `collision_warning`, `blind_spot_detection`, `lane_departure`, `adaptive_cruise`
- **Braking**: `brake_temp`, `brake_wear`, `brake_pressure`, `brake_balance`

#### **Electrical & Auxiliary (12 channels)**
- **Power Systems**: `battery_voltage`, `alternator_output`, `electrical_load`
- **Climate**: `cabin_temp`, `ac_compressor_load`, `heater_core_temp`
- **Lighting**: `headlight_status`, `brake_light_status`, `turn_signal_status`
- **Auxiliary**: `power_steering_pressure`, `windshield_wiper_status`, `horn_status`

#### **Environmental Interaction (10 channels)**
- **Road Surface**: `surface_type`, `friction_coefficient`, `road_temperature`, `wetness_level`
- **Weather Impact**: `wind_resistance`, `air_density`, `visibility_factor`
- **Traffic**: `nearby_vehicle_count`, `collision_proximity`, `traffic_density`

### **Physics Data (6 channels)**
- G-Forces (gx, gy, gz), Acceleration, Damage level, Critical components

### **Advanced Sensors (Ready for Integration)**
- **Camera**: 2.76M pixels → CNN processed to feature vector
- **LiDAR**: Variable point cloud → Grid-based obstacle detection
- **GPS**: 6 channels (positioning, velocity, heading)
- **IMU**: 12 channels (3D motion data)

**Total Input Channels: ~160+ numeric channels + advanced sensor processing**

---

## 🎯 **SIMPLIFIED REWARD SYSTEM: MINIMAL POLICY**

### **Phase 4A: Basic Survival Policy**
```python
# CORE REWARD FUNCTION - Minimal Constraints
def calculate_reward(state, action, next_state):
    reward = 0.0
    
    # PRIMARY REWARD: Distance Traveled
    distance_reward = calculate_distance_traveled(state, next_state)
    reward += distance_reward * 1.0  # Base reward rate
    
    # PRIMARY PENALTY: Object Collision
    if collision_detected(next_state):
        reward -= 100.0  # Major penalty for crashes
        trigger_vehicle_recovery()  # Auto-reset to safe position
    
    # MOVEMENT PENALTY: Stationary timeout
    if movement_stopped_for(5.0):  # 5 second threshold
        reward -= 10.0  # Penalty for stopping
        trigger_vehicle_recovery()  # Reset and start at 5mph
        set_initial_velocity(5.0)  # Force movement
    
    return reward

# Reward bounds: [-100, +∞] with primary focus on exploration
```

### **Expandable Reward Framework**
```python
class ExpandableRewardSystem:
    def __init__(self):
        self.reward_components = {
            'distance': {'weight': 1.0, 'active': True},
            'collision': {'weight': -100.0, 'active': True},
            'movement': {'weight': -10.0, 'active': True}
        }
        self.heuristic_rewards = {}  # For future expansion
    
    def add_heuristic(self, name, function, weight, activation_phase):
        """Add new reward/penalty heuristics during training"""
        self.heuristic_rewards[name] = {
            'function': function,
            'weight': weight,
            'phase': activation_phase,
            'active': False
        }
    
    def activate_phase(self, phase_number):
        """Progressively activate new reward components"""
        for heuristic in self.heuristic_rewards.values():
            if heuristic['phase'] <= phase_number:
                heuristic['active'] = True
```

### **Future Expansion Examples**
```python
# Phase 4B: Proximity Awareness (Add Later)
proximity_reward = lambda: slow_down_bonus_near_objects()

# Phase 4C: Speed Management (Add Later) 
speed_reward = lambda: appropriate_speed_for_conditions()

# Phase 5: Traffic Laws (Future)
traffic_reward = lambda: follow_traffic_signals_and_signs()

# Phase 6: Task-Specific (Future)
parking_reward = lambda: successful_parking_maneuvers()
```

---

## 🚗 **AUTO-RECOVERY SYSTEM**

### **Vehicle Recovery Mechanics**
```python
class AutoRecoverySystem:
    def __init__(self, vehicle, bng):
        self.vehicle = vehicle
        self.bng = bng
        self.stationary_timer = 0.0
        self.collision_detector = CollisionDetector()
        
    def check_recovery_conditions(self, state):
        # Condition 1: Collision Detection
        if self.collision_detector.detect_collision(state):
            self.perform_recovery("collision")
            return True
            
        # Condition 2: Stationary Timeout (5 seconds)
        if self.is_stationary(state):
            self.stationary_timer += self.dt
            if self.stationary_timer >= 5.0:
                self.perform_recovery("stationary")
                return True
        else:
            self.stationary_timer = 0.0
            
        return False
    
    def perform_recovery(self, reason):
        """Use BeamNG's built-in vehicle recovery system"""
        print(f"🚗 Auto-recovery triggered: {reason}")
        
        # BeamNG built-in recovery to safe position
        self.vehicle.recover()
        
        # Set initial forward velocity (5 mph = ~2.24 m/s)
        self.vehicle.control(throttle=0.3, steering=0.0, brake=0.0)
        
        # Reset timers and states
        self.stationary_timer = 0.0
        
        # Log recovery event for analysis
        self.log_recovery_event(reason)
```

---

## 🖥️ **LIVE NEURAL NETWORK VISUALIZATION DASHBOARD**

### **Real-Time Brain Monitoring System**
```python
class NeuralNetworkVisualizationDashboard:
    """Live visualization of AI brain activity during training"""
    
    def __init__(self, model, update_frequency=10):  # 10 Hz updates
        self.model = model
        self.update_freq = update_frequency
        self.activation_history = []
        self.heatmap_data = {}
        
        # Initialize visualization components
        self.setup_dashboard_ui()
        self.setup_layer_monitors()
        self.setup_activation_tracking()
    
    def setup_dashboard_ui(self):
        """Create real-time dashboard interface"""
        self.dashboard = {
            'input_layer': InputLayerVisualizer(160),  # 160+ input channels
            'hidden_layers': [HiddenLayerVisualizer(size) for size in [256, 128, 64]],
            'output_layer': OutputLayerVisualizer(3),  # throttle, steering, brake
            'activation_heatmap': ActivationHeatmap(),
            'synapse_connections': SynapseVisualizer(),
            'real_time_metrics': MetricsPanel()
        }
    
    def monitor_forward_pass(self, input_data):
        """Track neural activity during inference"""
        activations = {}
        
        # Hook into each layer to capture activations
        with torch.no_grad():
            x = input_data
            for i, layer in enumerate(self.model.layers):
                x = layer(x)
                activations[f'layer_{i}'] = x.clone()
                
                # Update visualization
                self.update_layer_visualization(i, x)
                self.update_heatmap(i, x)
        
        return activations
    
    def update_layer_visualization(self, layer_idx, activations):
        """Update real-time layer activity display"""
        # Calculate activation intensity (0-1 scale)
        intensity = torch.abs(activations).mean(dim=0)
        
        # Update dashboard display
        self.dashboard['hidden_layers'][layer_idx].update(intensity)
        
        # Track highly active neurons
        hot_neurons = (intensity > 0.7).nonzero().flatten()
        self.track_hot_neurons(layer_idx, hot_neurons)
    
    def update_heatmap(self, layer_idx, activations):
        """Generate real-time heatmap of brain activity"""
        # Calculate firing frequency and intensity
        firing_rate = self.calculate_firing_rate(activations)
        
        # Update heatmap display
        self.dashboard['activation_heatmap'].update(layer_idx, firing_rate)
        
        # Identify overused/underused regions
        self.analyze_utilization_patterns(layer_idx, firing_rate)
    
    def track_input_utilization(self, input_channels):
        """Monitor which input channels are being utilized"""
        channel_importance = self.calculate_channel_importance()
        
        # Categorize channel usage
        categories = {
            'vehicle_state': channel_importance[:7],
            'controls': channel_importance[7:10], 
            'engine_telemetry': channel_importance[10:25],
            'transmission': channel_importance[25:37],
            'suspension': channel_importance[37:55],
            'tires': channel_importance[55:71],
            'safety_systems': channel_importance[71:79],
            'environmental': channel_importance[79:89]
        }
        
        # Visualize category utilization
        self.dashboard['input_layer'].update_categories(categories)
    
    def generate_training_insights(self):
        """Analyze neural network learning patterns"""
        insights = {
            'overutilized_inputs': self.find_overused_channels(),
            'underutilized_inputs': self.find_underused_channels(),
            'learning_bottlenecks': self.identify_bottlenecks(),
            'policy_evolution': self.track_policy_changes(),
            'exploration_patterns': self.analyze_exploration_behavior()
        }
        
        return insights
```

### **Dashboard UI Layout**
```
┌─────────────────────────────────────────────────────────────────┐
│                  🧠 LIVE AI BRAIN MONITOR                       │
├─────────────────────────────────────────────────────────────────┤
│  INPUT CHANNELS (160+)           │  HIDDEN LAYER 1 (256)        │
│  ██ Vehicle State    [||||||||]  │  ████████████████████████     │
│  ██ Engine Telemetry [██████||||] │  ████████████████████████     │
│  ██ Transmission     [████||||||] │  ████████████████████████     │
│  ██ Suspension       [||||||||||] │  ████████████████████████     │
│  ██ Tires           [████████||] │  ████████████████████████     │
│  ██ Safety Systems   [||||||||] │  ████████████████████████     │
├─────────────────────────────────────────────────────────────────┤
│  HIDDEN LAYER 2 (128)           │  HIDDEN LAYER 3 (64)         │
│  ████████████████████████        │  ████████████████             │
│  ████████████████████████        │  ████████████████             │
├─────────────────────────────────────────────────────────────────┤
│  OUTPUT ACTIONS (3)              │  REAL-TIME METRICS            │
│  🚗 Throttle: [██████████] 0.85  │  Reward: +15.3               │
│  🎯 Steering: [████      ] 0.42  │  Distance: 1.2km             │
│  🛑 Brake:    [|         ] 0.05  │  Collisions: 2               │
├─────────────────────────────────────────────────────────────────┤
│  🔥 ACTIVATION HEATMAP           │  📊 LEARNING INSIGHTS         │
│  [Neuron firing intensity map]   │  • Engine telemetry: HIGH     │
│                                  │  • Suspension data: LOW       │
│                                  │  • Policy stability: GOOD     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ **POLICY-BASED RL TRAINING PIPELINE**

### **Algorithm Selection: Soft Actor-Critic (SAC)**
```python
# SAC Configuration for Continuous Control + Exploration
sac_config = {
    "algorithm": "Soft Actor-Critic (SAC)",
    "exploration": "Maximum entropy policy",
    "action_space": "Continuous [throttle, steering, brake]",
    "observation_space": "160+ dimensional state vector",
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "batch_size": 256,
    "tau": 0.005,  # Soft update coefficient
    "gamma": 0.99,  # Discount factor
    "alpha": "auto",  # Automatic entropy tuning
}
```

### **Training Loop with Live Visualization**
```python
def exploratory_training_loop():
    while training_active:
        # Environment step
        observation = env.get_observation()
        action = policy.select_action(observation)
        
        # Live neural network monitoring
        brain_activity = dashboard.monitor_forward_pass(observation)
        dashboard.update_real_time_display()
        
        # Execute action in BeamNG
        next_observation, reward, done, info = env.step(action)
        
        # Auto-recovery check
        if recovery_system.check_recovery_conditions(next_observation):
            done = True  # Reset episode
        
        # Store experience
        replay_buffer.add(observation, action, reward, next_observation, done)
        
        # Train policy
        if len(replay_buffer) > batch_size:
            policy_loss, q_loss = sac_agent.train_step(replay_buffer.sample())
            dashboard.log_training_metrics(policy_loss, q_loss)
        
        # Progressive reward activation
        if episode % reward_expansion_interval == 0:
            reward_system.maybe_activate_next_phase()
```

---

## 📈 **PROGRESSIVE TRAINING PHASES**

### **Phase 4A: Survival Basics (Weeks 1-2)**
- **Goal**: Learn basic vehicle control, avoid immediate crashes
- **Rewards**: Distance traveled (+), Collisions (-100), Stationary (-10)
- **Expected Behavior**: Erratic driving, frequent crashes, gradual improvement
- **Neural Activity**: Heavy firing in control outputs, basic sensor inputs

### **Phase 4B: Spatial Awareness (Weeks 3-4)**
- **Goal**: Develop obstacle detection and avoidance
- **Added Rewards**: Proximity awareness, smooth steering
- **Expected Behavior**: Better collision avoidance, smoother control
- **Neural Activity**: Increased utilization of LiDAR/camera inputs

### **Phase 4C: Efficiency Learning (Weeks 5-6)**
- **Goal**: Optimize speed and path efficiency
- **Added Rewards**: Speed optimization, path efficiency
- **Expected Behavior**: Faster, more direct routes
- **Neural Activity**: Balancing speed vs. safety neurons

### **Phase 5+: Complex Behaviors (Future)**
- Traffic law adherence, parking, multi-vehicle scenarios
- Advanced heuristics and task-specific rewards
- Real-world driving behavior emergence

---

## 🎯 **SUCCESS METRICS**

### **Phase 4A Targets**
- [ ] **Survival Rate**: >60% episodes without crash/timeout
- [ ] **Distance Progress**: Average 500m+ per episode
- [ ] **Control Learning**: Smooth action sequences (low jerk)
- [ ] **Recovery Resilience**: Successful recovery from >90% of resets

### **Neural Network Insights**
- [ ] **Input Utilization**: >80% of channels showing some activity
- [ ] **Layer Efficiency**: No dead neurons (>5% activation rate)
- [ ] **Learning Stability**: Consistent policy improvement over episodes
- [ ] **Exploration Balance**: Healthy mix of exploitation vs. exploration

---

**Status**: Ready for Phase 4A Implementation  
**Next Action**: Build exploratory RL environment with live brain visualization  
**Timeline**: 2 weeks for basic survival policy + dashboard development