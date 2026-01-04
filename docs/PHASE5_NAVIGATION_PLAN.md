# Phase 5: Navigation Awareness & Feature Engineering

**Date Started**: January 3, 2026  
**Status**: 🚀 ACTIVE  
**Prerequisites**: Phase 4C Complete (SAC neural network training, smoothness penalties)

---

## 🎯 Phase Overview

Phase 5 transitions the AI from blind distance maximization to **goal-directed navigation** with environmental awareness. The AI will learn to navigate toward specific waypoints, follow routes, detect road surfaces, and optimize speed based on upcoming geometry.

### Core Philosophy
- **No camera dependency** - Use GPS, electrics, and physics sensors only
- **Incremental complexity** - Start with single waypoint, progress to multi-waypoint routes
- **Feature engineering focus** - Derive intelligent features from raw telemetry
- **Real-world constraints** - Friction estimation, road surface detection, trajectory prediction

---

## 📍 Training Environment: West Coast USA Highway

### Map Characteristics
**Location**: west_coast_usa map  
**Spawn Point**: (-717.121, 101, 118.675)  

**Route Progression**:
1. **Urban Start** (0-500m)
   - Straight initial section
   - Buildings and obstacles
   - Tight maneuvering required
   - Good for basic waypoint navigation

2. **Transition Zone** (500-1500m)
   - Opening up from city
   - Gentle curves
   - Increasing speed zones
   - Grace runoffs to grass (off-road detection testing)

3. **Open Highway** (1500m+)
   - Smooth flowing road
   - Ideal for high-speed training
   - Long sight lines
   - Perfect for trajectory prediction and speed optimization

**Training Strategy**: Use environment progression as natural curriculum learning

---

## 🗺️ Phase 5 Roadmap

### **Part A: Basic Navigation (Foundation)**
**Timeline**: Week 1-2  
**Goal**: AI can navigate to single waypoint with heading awareness

#### A1. GPS Sensor Integration
- **File**: `phase5a_gps_integration.py`
- Add GPS sensor to vehicle
- Expose: latitude, longitude, altitude, heading
- Verify coordinate accuracy
- Test GPS update rate

#### A2. Single Waypoint System
- **File**: `phase5a_single_waypoint.py`
- Define target coordinates (500m down highway)
- Calculate distance to waypoint
- Calculate bearing to waypoint
- Add to state space: `distance_to_waypoint`, `bearing_to_waypoint`

#### A3. Heading Alignment Reward
- **File**: `phase5a_heading_reward.py`
- Calculate heading error: `vehicle_heading - bearing_to_waypoint`
- Reward: `cos(heading_error) * distance_progress`
- Penalty for driving away from waypoint
- Test with straight road section

#### A4. Road Surface Detection
- **File**: `phase5a_road_detection.py`
- Use wheel slip ratio (wheelspeed vs actual speed)
- Detect grass runoffs using damage accumulation rate
- Add state feature: `estimated_surface_type`
- Penalty: -5 points/second for off-road

**Success Criteria**:
- ✅ AI reaches single waypoint 90%+ of attempts
- ✅ Average heading error < 15 degrees
- ✅ Detects off-road within 0.5 seconds
- ✅ State space expanded to 31D (27 + 4 navigation)

---

### **Part B: Multi-Waypoint Pathfinding (Enhancement)**
**Timeline**: Week 3-4  
**Goal**: AI follows predefined routes through multiple waypoints

#### B1. Route Definition System
- **File**: `phase5b_route_system.py`
- Define waypoint lists for west_coast_usa
- Route 1: 3 waypoints (urban section)
- Route 2: 5 waypoints (transition zone)
- Route 3: 8 waypoints (full highway loop)
- Checkpoint progression logic

#### B2. Path Deviation Metric
- **File**: `phase5b_path_deviation.py`
- Calculate perpendicular distance from ideal line
- Ideal line: straight segment between consecutive waypoints
- Cross product method: `cross(pos - wpA, wpB - wpA) / ||wpB - wpA||`
- Add to state: `path_deviation_distance`
- Penalty: -2 points per meter of deviation

#### B3. Route Completion Tracking
- **File**: `phase5b_route_tracking.py`
- Track: current waypoint index, waypoints completed, route progress %
- Reward scaling: early waypoints (lower reward) → later waypoints (higher reward)
- Route completion bonus: +50 points for finishing route
- Episode ends: route complete OR crash OR timeout

**Success Criteria**:
- ✅ Completes 3-waypoint urban route 70%+ of time
- ✅ Path deviation < 10m average
- ✅ Progression through waypoints sequential (no skipping)
- ✅ State space: 37D (31 + 6 route features)

---

### **Part C: Feature Engineering (Advanced)**
**Timeline**: Week 5-6  
**Goal**: Intelligent speed control and trajectory optimization

#### C1. Road Friction Estimation
- **File**: `phase5c_friction_estimation.py`
- Slip ratio: `(wheelspeed - actual_speed) / wheelspeed`
- Lateral grip: G-forces during cornering
- Brake effectiveness: deceleration rate vs brake input
- Add to state: `estimated_friction_coefficient`
- Use for: dynamic speed limits in curves

#### C2. Trajectory Prediction
- **File**: `phase5c_trajectory_prediction.py`
- Predict position in 1s, 2s, 3s using velocity vector
- Check if trajectory intersects waypoint
- Calculate required steering to hit waypoint
- Reward: anticipatory corrections (before error grows)
- Add to state: `predicted_waypoint_miss_distance`

#### C3. Speed Optimization Zones
- **File**: `phase5c_speed_zones.py`
- Straight sections: max speed (30 m/s)
- Approaching waypoints: calculate required braking distance
- Curves: estimate safe speed from path curvature
- Dynamic speed target based on upcoming geometry
- Add to state: `optimal_speed_for_section`

#### C4. Control Smoothness Enhancements
- **File**: `phase5c_control_smoothness.py`
- Jerk penalty: rate of change of acceleration
- Steering rate vs speed: tighter limits at high speed
- Brake-throttle transition smoothness
- Reward smooth control sequences over multiple steps

**Success Criteria**:
- ✅ Friction estimate correlates with surface type (±0.2 coefficient)
- ✅ Trajectory prediction within 2m of actual path
- ✅ Speed adjusted based on upcoming waypoint distance
- ✅ Acceleration jerk < 5 m/s³ average
- ✅ State space: 41D (37 + 4 advanced features)

---

### **Part D: Integration & Training (Polish)**
**Timeline**: Week 7-8  
**Goal**: Production-ready navigation system with curriculum learning

#### D1. Curriculum Learning Implementation
- **File**: `phase5d_curriculum_learning.py`
- Stage 1: Single waypoint, 100m straight (episodes 1-50)
- Stage 2: Two waypoints, 300m gentle curve (episodes 51-150)
- Stage 3: Five waypoints, 1km highway section (episodes 151-300)
- Stage 4: Eight waypoints, 2km complex route (episodes 301+)
- Auto-advance on 80% success rate

#### D2. Reward Function Rebalancing
- **File**: `phase5d_reward_rebalancing.py`
- Integrate all reward components
- Tune weights based on telemetry analysis
- Test different reward scales for navigation vs smoothness
- Document final reward formula

**Current (Phase 4C)**:
```python
distance_progress: 0.5/meter
speed_bonus: 0.1/m/s
crash: -30
smoothness penalties: -0.2 to -5.0
```

**Proposed (Phase 5)**:
```python
waypoint_progress: 1.0/meter        # Higher than raw distance
heading_alignment: 0.3 * cos(error) # Bonus for pointing at target
path_deviation: -2.0/meter          # Stay on path
off_road: -5.0/second               # Avoid grass
route_completion: +50               # Finish route bonus
time_efficiency: +20/waypoint       # Faster = better
```

#### D3. Performance Benchmarks
- **File**: `phase5d_benchmarks.py`
- Benchmark 1: Single waypoint success rate
- Benchmark 2: Three-waypoint completion rate
- Benchmark 3: Average speed on route
- Benchmark 4: Path deviation metrics
- Benchmark 5: Smoothness scores
- Save benchmark results to CSV for comparison

#### D4. Long Training Runs
- **File**: `phase5d_extended_training.py`
- 500 episode runs with curriculum learning
- Best model tracking per stage
- Telemetry logging for all stages
- Performance graphs: success rate, distance, smoothness over time

**Success Criteria**:
- ✅ 90%+ success on Stage 1 (single waypoint)
- ✅ 80%+ success on Stage 2 (two waypoints)
- ✅ 70%+ success on Stage 3 (five waypoints)
- ✅ 60%+ success on Stage 4 (complex route)
- ✅ Average speed > 20 m/s on open highway
- ✅ Path deviation < 5m on straight sections

---

## 📊 State Space Evolution

### Phase 4C Final State (27D)
```python
# Position (3D)
position_x, position_y, position_z

# Velocity (3D)
velocity_x, velocity_y, velocity_z

# Distance (3D)
speed, distance_from_origin, distance_from_checkpoint

# Dynamics (15D)
throttle, steering, brake, rpm, gear, wheelspeed,
gx, gy, gz, damage, abs_active, esc_active, tcs_active, fuel

# Episode (3D)
episode_time, crash_count, stationary_time
```

### Phase 5 Final State (41D)
```python
# Phase 4C (27D) +

# Navigation (6D)
distance_to_waypoint          # Meters to current waypoint
bearing_to_waypoint           # Angle in radians
heading_error                 # Vehicle heading - bearing
waypoint_index                # Which checkpoint (0-N)
path_deviation                # Perpendicular distance from ideal line
route_completion_percent      # 0.0 to 1.0

# Road Awareness (4D)
estimated_friction            # 0.0 (ice) to 1.0 (dry asphalt)
wheel_slip_ratio              # 0.0 (no slip) to 1.0 (full slip)
off_road_duration             # Seconds off pavement
surface_type_estimate         # 0=road, 1=grass, 2=dirt

# Advanced Features (4D)
predicted_miss_distance       # Trajectory prediction error
optimal_speed_for_section     # Recommended speed (m/s)
braking_distance_to_waypoint  # Meters needed to stop
acceleration_jerk             # m/s³ (smoothness metric)
```

---

## 🎯 Success Criteria Summary

### Minimum (Phase 5 Complete)
- ✅ AI navigates to single waypoint 90%+ success
- ✅ Multi-waypoint route completion 70%+ (3 waypoints)
- ✅ Path deviation < 10m average
- ✅ Off-road detection working (grass vs pavement)
- ✅ Friction estimation implemented
- ✅ All state features integrated and functional

### Target (Excellent Performance)
- 🎯 5-waypoint route completion 80%+
- 🎯 Path deviation < 5m average
- 🎯 Maintains 20+ m/s average speed
- 🎯 Zero crashes on straight sections
- 🎯 Smooth control (jerk < 3 m/s³)

### Stretch (Superhuman)
- 🏆 8-waypoint complex route 90%+ success
- 🏆 Optimal racing line adherence
- 🏆 Adaptive speed control (fast straights, slow curves)
- 🏆 Predictable, human-like driving behavior
- 🏆 Beats hand-tuned PD controller

---

## 🚫 Out of Scope (Save for Phase 6/7)

- ❌ Lane detection (requires camera)
- ❌ Traffic interaction (requires AI traffic)
- ❌ Complex intersections (Phase 6)
- ❌ Weather/lighting variations (Phase 7)
- ❌ Multi-vehicle scenarios (Phase 7)
- ❌ Pedestrian detection (Phase 7)

---

## 📁 File Organization

```
src/phase5/
├── phase5a_gps_integration.py          # GPS sensor setup
├── phase5a_single_waypoint.py          # Basic waypoint navigation
├── phase5a_heading_reward.py           # Heading alignment
├── phase5a_road_detection.py           # Surface type detection
├── phase5b_route_system.py             # Multi-waypoint routes
├── phase5b_path_deviation.py           # Path following metrics
├── phase5b_route_tracking.py           # Route progress tracking
├── phase5c_friction_estimation.py      # Grip/surface analysis
├── phase5c_trajectory_prediction.py    # Look-ahead prediction
├── phase5c_speed_zones.py              # Dynamic speed limits
├── phase5c_control_smoothness.py       # Jerk penalties
├── phase5d_curriculum_learning.py      # Progressive difficulty
├── phase5d_reward_rebalancing.py       # Final reward tuning
├── phase5d_benchmarks.py               # Performance testing
└── phase5d_extended_training.py        # Long training runs

docs/
├── PHASE5_NAVIGATION_PLAN.md           # This document
├── PHASE5_WAYPOINT_COORDINATES.md      # west_coast_usa waypoint definitions
├── PHASE5_REWARD_ANALYSIS.md           # Reward component tuning
└── PHASE5_SUCCESS_REPORT.md            # Final results (created at end)
```

---

## 🔄 Integration with Phase 4C

Phase 5 builds directly on Phase 4C infrastructure:

**Reused Components**:
- ✅ SAC neural network architecture
- ✅ Experience replay buffer
- ✅ Training loop and metrics
- ✅ Crash detection and recovery
- ✅ Smoothness penalties
- ✅ Telemetry logging system
- ✅ Model checkpointing

**Enhanced Components**:
- 🔄 State space: 27D → 41D
- 🔄 Reward function: distance-based → waypoint-based
- 🔄 Episode termination: add route completion
- 🔄 Sensors: add GPS
- 🔄 Debug output: add waypoint progress

**New Components**:
- ✨ Waypoint management system
- ✨ Path deviation calculation
- ✨ Route definition framework
- ✨ Friction estimation module
- ✨ Trajectory prediction
- ✨ Curriculum learning scheduler

---

## 📈 Expected Training Timeline

**Phase 5A (Weeks 1-2)**: Basic Navigation
- Implementation: 3-4 days
- Testing & debugging: 2-3 days
- Training runs: 3-4 days
- **Milestone**: 90% single waypoint success

**Phase 5B (Weeks 3-4)**: Multi-Waypoint Routes
- Implementation: 4-5 days
- Route definition: 1-2 days
- Training runs: 4-5 days
- **Milestone**: 70% three-waypoint success

**Phase 5C (Weeks 5-6)**: Feature Engineering
- Friction estimation: 2-3 days
- Trajectory prediction: 2-3 days
- Speed optimization: 2-3 days
- Integration testing: 2-3 days
- **Milestone**: Adaptive speed control working

**Phase 5D (Weeks 7-8)**: Polish & Extended Training
- Curriculum implementation: 2-3 days
- Reward tuning: 2-3 days
- Benchmarking: 1-2 days
- Long training (500+ episodes): 4-5 days
- **Milestone**: Phase 5 complete

**Total**: ~8 weeks (flexible based on results)

---

## 🎓 Learning Objectives

By end of Phase 5, the AI should demonstrate:

1. **Goal-Directed Behavior**
   - Drives toward specific coordinates
   - Follows multi-waypoint routes
   - Completes predefined paths

2. **Environmental Awareness**
   - Detects road vs off-road surfaces
   - Estimates friction/grip levels
   - Adapts to changing conditions

3. **Anticipatory Control**
   - Predicts future trajectory
   - Adjusts speed before waypoints
   - Plans ahead for curves

4. **Smooth Execution**
   - Low acceleration jerk
   - Gradual control inputs
   - Natural driving feel

5. **Robust Performance**
   - Works across different routes
   - Generalizes to new waypoints
   - Handles curriculum progression

---

## 🛠️ Technical Risks & Mitigation

### Risk 1: GPS Coordinate System Confusion
- **Risk**: BeamNG coordinates may not align with lat/lon
- **Mitigation**: Use GPS for relative bearing only, keep Cartesian for distance
- **Fallback**: Pure Cartesian waypoint system without GPS

### Risk 2: Path Deviation Calculation Complexity
- **Risk**: Cross product math errors or edge cases
- **Mitigation**: Extensive unit testing with known geometries
- **Fallback**: Simple distance-to-line approximation

### Risk 3: Friction Estimation Inaccuracy
- **Risk**: Wheel slip may not correlate well with surface
- **Mitigation**: Combine multiple signals (slip, g-forces, damage)
- **Fallback**: Binary road/off-road detection only

### Risk 4: State Space Too Large
- **Risk**: 41D may be too complex for current network
- **Mitigation**: Incremental addition, monitor training stability
- **Fallback**: Prune least-used features based on gradient analysis

### Risk 5: Curriculum Learning Instability
- **Risk**: AI may overfit to early stages
- **Mitigation**: Mix in harder examples even in early stages
- **Fallback**: Manual stage progression based on performance

---

## 📝 Documentation Requirements

Each sub-phase file should include:

1. **Header Comment**
   - Phase/subphase identifier
   - Purpose and goals
   - Dependencies on previous phases
   - Expected outcomes

2. **Configuration Section**
   - Hyperparameters
   - Reward weights
   - Waypoint definitions
   - Threshold values

3. **Inline Comments**
   - Algorithm explanations
   - Math formulas
   - Edge case handling
   - Performance notes

4. **Output Logging**
   - Progress indicators
   - Debug information
   - Performance metrics
   - Error messages with context

5. **Testing Section**
   - Unit tests for key functions
   - Integration tests
   - Validation checks

---

## 🎯 Phase 5 Completion Checklist

- [ ] Part A: Basic Navigation complete
  - [ ] GPS sensor integrated
  - [ ] Single waypoint navigation working
  - [ ] Heading alignment reward tuned
  - [ ] Road surface detection functional
  
- [ ] Part B: Multi-Waypoint Routes complete
  - [ ] Route definition system implemented
  - [ ] Path deviation calculation working
  - [ ] Route completion tracking functional
  - [ ] 3-waypoint route 70%+ success

- [ ] Part C: Feature Engineering complete
  - [ ] Friction estimation implemented
  - [ ] Trajectory prediction working
  - [ ] Speed optimization zones functional
  - [ ] Control smoothness enhanced

- [ ] Part D: Integration & Training complete
  - [ ] Curriculum learning implemented
  - [ ] Reward function rebalanced
  - [ ] Benchmarks established
  - [ ] Extended training completed (500+ episodes)

- [ ] Documentation complete
  - [ ] All code files documented
  - [ ] Waypoint coordinates defined
  - [ ] Reward analysis documented
  - [ ] Success report created

- [ ] Success criteria met
  - [ ] Minimum criteria: 90% single waypoint, 70% multi-waypoint
  - [ ] Target criteria: 80% 5-waypoint, <5m deviation, 20+ m/s speed
  - [ ] All state features functional

---

**Phase 5 Status**: 🚀 READY TO BEGIN  
**Next Action**: Implement Part A1 - GPS Sensor Integration
