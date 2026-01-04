#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5A: Single Waypoint Navigation
BeamNG AI Driver - Goal-Directed Navigation Foundation

This file implements basic waypoint navigation using Cartesian coordinates.
The AI learns to drive toward a specific target position using:
- Distance to waypoint
- Bearing to waypoint (atan2 calculation)
- Heading error (vehicle heading vs bearing)
- Heading alignment reward

Builds on: Phase 4C (SAC neural network, smoothness penalties)
New features: Waypoint-based rewards, heading alignment, goal-directed behavior

State space: 31D (27D from Phase 4C + 4D navigation)
"""

import time
import numpy as np
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import csv
from pathlib import Path
from datetime import datetime

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage, GForces

# PyTorch for neural networks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("PyTorch available - neural network training enabled")
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available")

# Import Phase 4C components
import sys
sys.path.append('../phase4')
from phase4c_neural_highway_training import (
    SACActorNetwork, SACCriticNetwork, SACAgent,
    ExperienceReplay, TrainingMetrics
)

# ============================================================================
# WAYPOINT DEFINITIONS
# ============================================================================

# West Coast USA - Route 1 (Urban Basics)
ROUTE_1_WAYPOINTS = [
    {'id': 'spawn', 'pos': (-717.121, 101.0, 118.675), 'name': 'Start'},
    {'id': 'wp1', 'pos': (-730.0, 85.0, 118.5), 'name': 'Urban Straight'},
]

# For Phase 5A, we'll use just the first waypoint
TARGET_WAYPOINT = ROUTE_1_WAYPOINTS[1]
SPAWN_POS = ROUTE_1_WAYPOINTS[0]['pos']
SPAWN_ROT = (0, 0, 0.3826834, 0.9238795)

# Waypoint tolerance
WAYPOINT_REACHED_RADIUS = 15.0  # meters (generous for learning)

# ============================================================================
# NAVIGATION STATE (Extended from Phase 4C)
# ============================================================================

@dataclass
class NavigationState:
    """
    Extended state representation with navigation features
    Phase 4C (27D) + Navigation (4D) = 31D total
    """
    # Phase 4C base state (27D)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    speed: float = 0.0
    distance_from_origin: float = 0.0
    distance_from_checkpoint: float = 0.0
    throttle: float = 0.0
    steering: float = 0.0
    brake: float = 0.0
    rpm: float = 0.0
    gear: float = 0.0
    wheelspeed: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    damage: float = 0.0
    abs_active: float = 0.0
    esc_active: float = 0.0
    tcs_active: float = 0.0
    fuel: float = 1.0
    episode_time: float = 0.0
    crash_count: float = 0.0
    stationary_time: float = 0.0
    
    # NEW: Navigation features (4D)
    distance_to_waypoint: float = 0.0      # Meters to target
    bearing_to_waypoint: float = 0.0       # Radians (0 = north)
    heading_error: float = 0.0             # Radians (bearing - vehicle_heading)
    waypoint_reached: float = 0.0          # 1.0 if reached, 0.0 otherwise
    
    def to_vector(self) -> np.ndarray:
        """Convert to 31D numpy array for neural network"""
        return np.array([
            # Position (3)
            self.position[0], self.position[1], self.position[2],
            # Velocity (3)
            self.velocity[0], self.velocity[1], self.velocity[2],
            # Distance (3)
            self.speed, self.distance_from_origin, self.distance_from_checkpoint,
            # Dynamics (15)
            self.throttle, self.steering, self.brake, self.rpm, self.gear,
            self.wheelspeed, self.gx, self.gy, self.gz,
            self.damage, self.abs_active, self.esc_active, self.tcs_active, self.fuel,
            # Episode (3)
            self.episode_time, self.crash_count, self.stationary_time,
            # Navigation (4) - NEW
            self.distance_to_waypoint, self.bearing_to_waypoint,
            self.heading_error, self.waypoint_reached
        ], dtype=np.float32)

# ============================================================================
# SINGLE WAYPOINT ENVIRONMENT
# ============================================================================

class SingleWaypointEnvironment:
    """
    Phase 5A: Single waypoint navigation environment
    Extends Phase 4C with waypoint-based rewards
    """
    
    def __init__(self, host='localhost', port=64256):
        self.host = host
        self.port = port
        self.bng = None
        self.vehicle = None
        self.scenario = None
        self.metrics = None
        self.bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        
        # Waypoint tracking
        self.target_waypoint = np.array(TARGET_WAYPOINT['pos'], dtype=np.float32)
        self.waypoint_name = TARGET_WAYPOINT['name']
        self.waypoint_reached = False
        self.closest_approach = float('inf')
        
        # Episode tracking (from Phase 4C)
        self.episode_origin = None
        self.current_checkpoint = None
        self.episode_start_time = 0.0
        self.last_position = None
        self.last_damage = 0.0
        self.steps_since_checkpoint = 0
        
        # Statistics
        self.episode_count = 0
        self.total_steps = 0
        self.crash_count = 0
        self.stationary_timer = 0.0
        self.already_recovered = False
        
        # Control smoothness tracking
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.last_brake = 0.0
    
    def connect(self, auto_launch=False):
        """Connect to running BeamNG instance"""
        print("=" * 60)
        print("Phase 5A: Single Waypoint Navigation")
        print("=" * 60)
        print(f"Target: {self.waypoint_name} at {self.target_waypoint}")
        print()
        
        try:
            set_up_simple_logging()
            self.bng = BeamNGpy(self.host, self.port, home=self.bng_home)
            self.bng.open(launch=auto_launch)
            print("[OK] Connected to BeamNG")
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False
    
    def setup_scenario(self, map_name='west_coast_usa', fast_attach=False):
        """Setup scenario with waypoint visualization"""
        if fast_attach:
            print("\n[FAST ATTACH] Using existing scenario...")
            return self._attach_to_existing()
        
        print(f"\nSetting up scenario on {map_name}...")
        
        try:
            self.scenario = Scenario(map_name, 'waypoint_navigation',
                                    description='Phase 5A: Single Waypoint Navigation')
            
            # Random vehicle selection (from Phase 4C)
            vehicles = ['etk800', 'etkc', 'etki', 'vivace', 'moonhawk']
            selected_vehicle = np.random.choice(vehicles)
            print(f"[VEHICLE] {selected_vehicle}")
            
            self.vehicle = Vehicle('ai_waypoint', model=selected_vehicle, license='PHASE5A')
            
            # Attach sensors
            self.vehicle.sensors.attach('electrics', Electrics())
            self.vehicle.sensors.attach('damage', Damage())
            self.vehicle.sensors.attach('gforces', GForces())
            
            # Spawn at start position
            self.scenario.add_vehicle(self.vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
            
            # Build and load
            self.scenario.make(self.bng)
            self.bng.settings.set_deterministic(60)
            self.bng.scenario.load(self.scenario)
            self.bng.scenario.start()
            
            time.sleep(3)
            
            # Initialize tracking
            self.vehicle.sensors.poll()
            pos = self.vehicle.state['pos']
            self.episode_origin = np.array(pos, dtype=np.float32)
            self.current_checkpoint = self.episode_origin.copy()
            self.last_position = self.episode_origin.copy()
            self.episode_start_time = time.time()
            
            print("[OK] Scenario ready")
            print(f"[WAYPOINT] Navigate to: {self.waypoint_name}")
            print(f"[DISTANCE] {np.linalg.norm(self.target_waypoint - self.episode_origin):.1f}m away")
            return True
            
        except Exception as e:
            print(f"[ERROR] Scenario setup failed: {e}")
            return False
    
    def get_vehicle_heading(self):
        """Extract vehicle heading from direction vector"""
        # vehicle.state['dir'] gives forward direction vector
        dir_vec = self.vehicle.state.get('dir', None)
        if dir_vec is None:
            return 0.0
        
        # Convert direction vector to heading (radians, 0 = north)
        # dir[0] = x component (east), dir[1] = y component (north)
        heading = np.arctan2(dir_vec[0], dir_vec[1])
        return heading
    
    def calculate_navigation_features(self, vehicle_pos):
        """Calculate distance, bearing, and heading error to waypoint"""
        # Distance to waypoint (2D, ignore Z)
        dx = self.target_waypoint[0] - vehicle_pos[0]
        dy = self.target_waypoint[1] - vehicle_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Bearing to waypoint (0 = north, clockwise)
        bearing = np.arctan2(dx, dy)
        
        # Vehicle heading
        vehicle_heading = self.get_vehicle_heading()
        
        # Heading error (how far off we're pointing)
        heading_error = bearing - vehicle_heading
        
        # Normalize heading error to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Check if waypoint reached
        reached = 1.0 if distance < WAYPOINT_REACHED_RADIUS else 0.0
        
        # Track closest approach
        if distance < self.closest_approach:
            self.closest_approach = distance
        
        return distance, bearing, heading_error, reached
    
    def get_state(self) -> NavigationState:
        """Get current state with navigation features"""
        self.vehicle.sensors.poll()
        
        # Vehicle state
        pos = np.array(self.vehicle.state['pos'], dtype=np.float32)
        vel = np.array(self.vehicle.state['vel'], dtype=np.float32)
        speed = float(np.linalg.norm(vel))
        
        # Calculate navigation features
        dist_to_wp, bearing_to_wp, heading_err, wp_reached = self.calculate_navigation_features(pos)
        
        # Electrics and sensors
        electrics = self.vehicle.sensors['electrics']
        damage_data = self.vehicle.sensors['damage']
        gforces = self.vehicle.sensors['gforces']
        
        # Helper function to safely extract float values
        def safe_float(value, default=0.0):
            if isinstance(value, (int, float)):
                return float(value)
            return default
        
        # Parse damage
        damage_values = []
        if damage_data and isinstance(damage_data, dict):
            for value in damage_data.values():
                if isinstance(value, (int, float)):
                    damage_values.append(value)
                elif isinstance(value, dict):
                    damage_values.extend([v for v in value.values() if isinstance(v, (int, float))])
        total_damage = sum(damage_values) if damage_values else 0.0
        
        # Distances
        distance_from_origin = float(np.linalg.norm(pos - self.episode_origin))
        distance_from_checkpoint = float(np.linalg.norm(pos - self.current_checkpoint))
        
        return NavigationState(
            position=pos,
            velocity=vel,
            speed=speed,
            distance_from_origin=distance_from_origin,
            distance_from_checkpoint=distance_from_checkpoint,
            throttle=safe_float(electrics.get('throttle', 0.0)),
            steering=safe_float(electrics.get('steering', 0.0)),
            brake=safe_float(electrics.get('brake', 0.0)),
            rpm=safe_float(electrics.get('rpm', 0.0)),
            gear=safe_float(electrics.get('gear', 0.0)),
            wheelspeed=safe_float(electrics.get('wheelspeed', 0.0)),
            gx=safe_float(gforces.get('gx', 0.0)),
            gy=safe_float(gforces.get('gy', 0.0)),
            gz=safe_float(gforces.get('gz', -9.81), -9.81),
            damage=total_damage,
            abs_active=safe_float(electrics.get('abs_active', 0.0)),
            esc_active=safe_float(electrics.get('esc_active', 0.0)),
            tcs_active=safe_float(electrics.get('tcs_active', 0.0)),
            fuel=safe_float(electrics.get('fuel', 1.0), 1.0),
            episode_time=time.time() - self.episode_start_time,
            crash_count=float(self.crash_count),
            stationary_time=self.stationary_timer,
            # Navigation features
            distance_to_waypoint=dist_to_wp,
            bearing_to_waypoint=bearing_to_wp,
            heading_error=heading_err,
            waypoint_reached=wp_reached
        )
    
    def step(self, action: np.ndarray):
        """Execute action and return (state, reward, done, info)"""
        current_state = self.get_state()
        
        # Apply action
        throttle = float(np.clip(action[0], 0, 1))
        steering = float(np.clip(action[1], -1, 1))
        brake = float(np.clip(action[2], 0, 1))
        
        # Calculate control smoothness
        throttle_delta = abs(throttle - self.last_throttle)
        steering_delta = abs(steering - self.last_steering)
        brake_delta = abs(brake - self.last_brake)
        
        self.last_throttle = throttle
        self.last_steering = steering
        self.last_brake = brake
        
        # Control vehicle
        self.vehicle.control(throttle=throttle, steering=steering, brake=brake, parkingbrake=0)
        time.sleep(0.3)
        
        # Get next state
        next_state = self.get_state()
        
        # Calculate reward
        reward, info = self._calculate_reward(
            current_state, next_state,
            throttle, steering, brake,
            throttle_delta, steering_delta, brake_delta
        )
        
        # Add navigation info
        info['distance_to_waypoint'] = next_state.distance_to_waypoint
        info['heading_error_deg'] = np.degrees(next_state.heading_error)
        info['waypoint_reached'] = next_state.waypoint_reached > 0.5
        
        # Log telemetry BEFORE recovery
        position_delta = np.linalg.norm(next_state.position - current_state.position)
        damage_increase = next_state.damage - self.last_damage
        info['position_delta'] = position_delta
        info['damage_increase'] = damage_increase
        
        if hasattr(self, 'metrics') and self.metrics:
            event_type = ''
            if info.get('crash_detected'): event_type = 'CRASH'
            elif info.get('stationary_timeout'): event_type = 'STUCK'
            elif info.get('waypoint_reached'): event_type = 'WAYPOINT'
            elif info.get('speeding'): event_type = 'SPEEDING'
            
            self.metrics.log_telemetry(
                episode=self.episode_count, step=self.total_steps, state=next_state,
                damage_increase=damage_increase, position_delta=position_delta,
                stationary_timer=self.stationary_timer, info=info,
                reward=reward, event_type=event_type
            )
        
        # Check for episode end
        done = info['crash_detected'] or info['stationary_timeout'] or info['waypoint_reached']
        
        # Handle crash/timeout recovery
        if info['crash_detected'] or info['stationary_timeout']:
            self.crash_count += 1
            self._recover_to_spawn()
            self.already_recovered = True
            done = True
        
        # Handle waypoint reached (success!)
        if info['waypoint_reached'] and not self.waypoint_reached:
            self.waypoint_reached = True
            print(f"\n  [WAYPOINT] REACHED {self.waypoint_name}! (+100 bonus)")
            done = True
        
        self.total_steps += 1
        
        return next_state.to_vector(), reward, done, info
    
    def _calculate_reward(self, current_state, next_state,
                         throttle, steering, brake,
                         throttle_delta, steering_delta, brake_delta):
        """Calculate reward with waypoint-based navigation"""
        reward = 0.0
        info = {
            'crash_detected': False,
            'stationary_timeout': False,
            'waypoint_reached': False,
            'speeding': False,
            'speed': next_state.speed
        }
        
        # ========== WAYPOINT PROGRESS REWARD (PRIMARY) ==========
        # Progress toward waypoint is MORE valuable than raw distance
        waypoint_progress = current_state.distance_to_waypoint - next_state.distance_to_waypoint
        if waypoint_progress > 0:
            reward += waypoint_progress * 1.0  # 1.0 points per meter (higher than Phase 4C)
            info['waypoint_progress'] = waypoint_progress
        
        # ========== HEADING ALIGNMENT BONUS ==========
        # Reward for pointing toward waypoint
        # cos(heading_error): 1.0 when perfect, 0.0 when perpendicular, -1.0 when backwards
        heading_alignment = np.cos(next_state.heading_error)
        if waypoint_progress > 0:  # Only when making progress
            reward += heading_alignment * 0.3  # Bonus up to +0.3 per step
        
        # ========== WAYPOINT REACHED BONUS ==========
        if next_state.waypoint_reached > 0.5:
            reward += 100.0  # Huge bonus for success!
            info['waypoint_reached'] = True
        
        # ========== SPEED OPTIMIZATION ==========
        speed_limit_ms = 30.0
        if waypoint_progress > 0 and next_state.speed > 1.0:
            if next_state.speed <= speed_limit_ms:
                reward += next_state.speed * 0.1
            else:
                overspeed = next_state.speed - speed_limit_ms
                reward -= overspeed * 0.5
                info['speeding'] = True
        
        # ========== SMOOTHNESS PENALTIES (from Phase 4C) ==========
        if steering_delta > 0.3:
            reward -= steering_delta * 2.0
        if throttle > 0.3 and brake > 0.3:
            reward -= (throttle * brake) * 5.0
            info['brake_riding'] = True
        if abs(steering) > 0.7 and next_state.speed > 10.0:
            reward -= abs(steering) * 0.5
        
        # ========== CRASH DETECTION (from Phase 4C, adjusted thresholds) ==========
        damage_increase = next_state.damage - self.last_damage
        
        if next_state.damage > 100.0:
            reward -= 30.0
            info['crash_detected'] = True
            self.last_damage = next_state.damage
        elif damage_increase > 50.0:
            reward -= 30.0
            info['crash_detected'] = True
            self.last_damage = next_state.damage
        elif damage_increase > 10.0:
            reward -= 10.0
            self.last_damage = next_state.damage
        elif damage_increase > 0.5:
            reward -= 2.0
            self.last_damage = next_state.damage
        
        # ========== STUCK DETECTION ==========
        position_delta = np.linalg.norm(next_state.position - current_state.position)
        if position_delta < 0.2:
            self.stationary_timer += 0.3
            reward -= 0.2
            if next_state.speed > 10.0 and position_delta < 0.05:
                reward -= 1.0
                info['stuck'] = True
            if self.stationary_timer > 6.0:
                info['stationary_timeout'] = True
                reward -= 20.0
        else:
            self.stationary_timer = 0.0
        
        info['reward'] = reward
        return reward, info
    
    def _recover_to_spawn(self):
        """Teleport back to spawn point"""
        try:
            self.vehicle.teleport(pos=SPAWN_POS, rot_quat=SPAWN_ROT, reset=True)
            time.sleep(2.0)
            self.vehicle.sensors.poll()
            self.last_damage = 0.0
            self.stationary_timer = 0.0
            self.last_throttle = 0.0
            self.last_steering = 0.0
            self.last_brake = 0.0
        except Exception as e:
            print(f"[WARN] Recovery failed: {e}")
    
    def reset_episode(self):
        """Reset for new episode"""
        print("\n--- Episode Reset ---")
        
        try:
            self.bng.scenario.restart()
            time.sleep(1)
            
            self.vehicle.sensors.poll()
            pos = self.vehicle.state['pos']
            self.episode_origin = np.array(pos, dtype=np.float32)
            self.current_checkpoint = self.episode_origin.copy()
            self.last_position = self.episode_origin.copy()
            self.last_damage = 0.0
            self.episode_start_time = time.time()
            self.crash_count = 0
            self.stationary_timer = 0.0
            self.episode_count += 1
            self.waypoint_reached = False
            self.closest_approach = float('inf')
            
            # Reset control tracking
            self.last_throttle = 0.0
            self.last_steering = 0.0
            self.last_brake = 0.0
            
            print(f"Episode {self.episode_count} started")
            print(f"Target: {self.waypoint_name} ({np.linalg.norm(self.target_waypoint - self.episode_origin):.1f}m away)")
            
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
    
    def _attach_to_existing(self):
        """Attach to already-running scenario and vehicle (FAST!)"""
        try:
            # Get current vehicles in scenario
            vehicles = self.bng.vehicles.get_current()
            if not vehicles:
                print("[ERROR] No vehicles in running scenario")
                return False
            
            # Use first vehicle
            vehicle_id = list(vehicles.keys())[0]
            self.vehicle = vehicles[vehicle_id]
            print(f"[VEHICLE] Found existing: {vehicle_id}")
            
            # CRITICAL: Connect to the vehicle
            self.vehicle.connect(self.bng)
            print("[CONNECT] Connected to vehicle")
            
            # Wait for connection to stabilize
            time.sleep(1)
            
            # Sensors should already be attached from previous session
            # Just verify they exist, don't re-attach
            try:
                self.vehicle.sensors.poll()
                print("[SENSORS] Using existing sensors")
            except Exception as sensor_err:
                print(f"[SENSORS] Re-attaching sensors: {sensor_err}")
                self.vehicle.sensors.attach('electrics', Electrics())
                self.vehicle.sensors.attach('damage', Damage())
                self.vehicle.sensors.attach('gforces', GForces())
                time.sleep(1)
                self.vehicle.sensors.poll()
            
            # Initialize tracking
            pos = self.vehicle.state['pos']
            self.episode_origin = np.array(pos, dtype=np.float32)
            self.current_checkpoint = self.episode_origin.copy()
            self.last_position = self.episode_origin.copy()
            self.episode_start_time = time.time()
            self.last_damage = 0.0  # Reset damage tracking
            
            print("[OK] Fast attach complete (no scenario reload!)")
            print(f"[WAYPOINT] Navigate to: {self.waypoint_name}")
            print(f"[DISTANCE] {np.linalg.norm(self.target_waypoint - self.episode_origin):.1f}m away")
            return True
            
        except Exception as e:
            print(f"[ERROR] Fast attach failed: {e}")
            import traceback
            traceback.print_exc()
            print("[FALLBACK] Will create new scenario...")
            return False
    
    def close(self):
        """Close connection"""
        if self.bng:
            self.bng.disconnect()

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_single_waypoint(episodes=100, batch_size=64, replay_start_size=1000, auto_launch=False, fast_attach=True):
    """Train AI to navigate to single waypoint"""
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required")
        return
    
    # Initialize environment
    env = SingleWaypointEnvironment()
    
    if not env.connect(auto_launch=auto_launch):  # Connect or launch BeamNG
        return
    
    if not env.setup_scenario(fast_attach=fast_attach):
        return
    
    # Initialize agent (31D state now)
    state_dim = 31  # Extended from 27D
    action_dim = 3
    
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    replay_buffer = ExperienceReplay(capacity=100000)
    
    # Metrics
    metrics = TrainingMetrics()
    env.metrics = metrics
    
    # NOTE: Phase 4C models are 27D - incompatible with Phase 5A (31D)
    # Starting fresh training for waypoint navigation
    # TODO Phase 5B: Implement transfer learning from Phase 4C base
    
    best_model_path = metrics.model_dir / 'waypoint_best.pth'
    if best_model_path.exists():
        print("\n[RESUME] Loading existing Phase 5A model...")
        try:
            agent.load(str(best_model_path))
        except Exception as e:
            print(f"[WARN] Model load failed (dimension mismatch?): {e}")
            print("[INFO] Starting fresh training")
    
    print("\n" + "=" * 60)
    print("TRAINING START - Single Waypoint Navigation")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"State space: 31D (27D base + 4D navigation)")
    print(f"Target: {TARGET_WAYPOINT['name']} at {TARGET_WAYPOINT['pos']}")
    print("=" * 60 + "\n")
    
    total_reward_history = []
    waypoint_success_count = 0
    
    try:
        for episode in range(episodes):
            if episode > 0 and not env.already_recovered:
                env.reset_episode()
            
            env.already_recovered = False
            
            state = env.get_state().to_vector()
            episode_reward = 0
            episode_steps = 0
            episode_start = time.time()
            
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            if len(replay_buffer) < replay_start_size:
                print(f"[EXPLORE] Collecting samples ({len(replay_buffer)}/{replay_start_size})")
            else:
                print(f"[TRAIN] Neural training mode")
            
            while True:
                # Get action
                if len(replay_buffer) < replay_start_size:
                    # Better exploration: bias toward forward motion
                    action = np.array([
                        np.random.uniform(0.5, 0.9),   # More throttle
                        np.random.uniform(-0.5, 0.5),  # Moderate steering
                        np.random.uniform(0, 0.1)       # Minimal brake
                    ])
                else:
                    action = agent.get_action(state, deterministic=False)
                
                # Execute step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                replay_buffer.push(state, action, reward, next_state, float(done))
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Print progress every 5 steps
                if episode_steps % 5 == 0 or done:
                    flags = []
                    if info.get('waypoint_reached'): flags.append("[WAYPOINT!]")
                    if info.get('crash_detected'): flags.append("[CRASH]")
                    if info.get('stationary_timeout'): flags.append("[STUCK]")
                    if info.get('brake_riding'): flags.append("[BRAKE+THROTTLE]")
                    flags_str = " ".join(flags) if flags else ""
                    
                    print(f"  Step {episode_steps}: Speed={info.get('speed', 0):.1f}m/s, "
                          f"Dist={info.get('distance_to_waypoint', 0):.1f}m, "
                          f"HeadErr={info.get('heading_error_deg', 0):.1f}°, "
                          f"Reward={reward:.2f} {flags_str}")
                
                # Train agent
                if len(replay_buffer) >= replay_start_size and episode_steps % 2 == 0:
                    batch = replay_buffer.sample(batch_size)
                    losses = agent.update(batch)
                
                if done or episode_steps > 200:
                    break
            
            # Episode summary
            episode_time = time.time() - episode_start
            avg_reward = episode_reward / episode_steps if episode_steps > 0 else 0
            final_state = env.get_state()
            
            # Check waypoint success
            if final_state.waypoint_reached > 0.5:
                waypoint_success_count += 1
            
            print(f"\n--- Episode {episode + 1} Complete ---")
            print(f"  Steps: {episode_steps}, Time: {episode_time:.1f}s")
            print(f"  Total Reward: {episode_reward:.1f}, Avg: {avg_reward:.2f}")
            print(f"  Final Distance: {final_state.distance_to_waypoint:.1f}m")
            print(f"  Closest Approach: {env.closest_approach:.1f}m")
            print(f"  Waypoint Reached: {'YES!' if final_state.waypoint_reached > 0.5 else 'No'}")
            print(f"  Success Rate: {waypoint_success_count}/{episode+1} ({100*waypoint_success_count/(episode+1):.1f}%)")
            
            # Save best model (closest approach)
            if env.closest_approach < metrics.best_distance or final_state.waypoint_reached > 0.5:
                metrics.best_distance = env.closest_approach
                agent.save(str(metrics.model_dir / 'waypoint_best.pth'))
                print(f"  [SAVE] New best: {env.closest_approach:.1f}m")
            
            total_reward_history.append(episode_reward)
            
            # Checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                checkpoint_path = metrics.model_dir / f'waypoint_checkpoint_ep{episode+1}.pth'
                agent.save(str(checkpoint_path))
        
        # Training complete
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Episodes: {episodes}")
        print(f"Waypoint Success: {waypoint_success_count}/{episodes} ({100*waypoint_success_count/episodes:.1f}%)")
        print(f"Best Approach: {metrics.best_distance:.1f}m")
        print("=" * 60)
        
        agent.save(str(metrics.model_dir / 'waypoint_final.pth'))
        
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training stopped by user")
        agent.save(str(metrics.model_dir / 'waypoint_interrupted.pth'))
    
    finally:
        env.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # auto_launch=True: Launch new BeamNG (30s startup)
    # auto_launch=False: Connect to running instance
    # fast_attach=True: Reuse running vehicle (instant start!)
    # fast_attach=False: Create new scenario (15s load)
    train_single_waypoint(episodes=100, auto_launch=False, fast_attach=True)
