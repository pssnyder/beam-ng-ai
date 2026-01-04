#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5A: Single Waypoint Navigation (Built on Phase 4C)
BeamNG AI Driver - Goal-Directed Navigation

Built on working Phase 4C code with additions:
- Waypoint coordinates and navigation
- 31D state space (27D + 4D navigation)
- Waypoint progress rewards
- Heading alignment bonus

Usage:
1. Launch BeamNG.drive manually
2. Run this script - it will connect to the running instance
3. Training begins with waypoint navigation goal
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

# ============================================================================
# WAYPOINT NAVIGATION (Phase 5A Addition)
# ============================================================================

# Route 1: Two waypoints along the road (Phase 5A)
WAYPOINT_ROUTE = [
    {'pos': np.array([-730.0, 85.0, 118.5], dtype=np.float32), 'name': 'WP1: First Turn', 'radius': 20.0},
    {'pos': np.array([-750.0, 65.0, 118.0], dtype=np.float32), 'name': 'WP2: Road Curve', 'radius': 20.0},
]

# Spawn position
SPAWN_POS = (-717.121, 101.0, 118.675)
SPAWN_ROT = (0, 0, 0.3826834, 0.9238795)

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
    print("WARNING: PyTorch not available - install with: pip install torch")
    print("Neural network training disabled - using random policy")

# ============================================================================
# NEURAL NETWORK COMPONENTS (from Phase 4B)
# ============================================================================

class SACActorNetwork(nn.Module):
    """SAC Actor Network - outputs continuous actions"""
    
    def __init__(self, state_dim=60, action_dim=3, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob

class SACCriticNetwork(nn.Module):
    """SAC Critic Network - estimates Q-values"""
    
    def __init__(self, state_dim=60, action_dim=3, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_value = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_value(x)

class SACAgent:
    """Soft Actor-Critic Agent for highway driving"""
    
    def __init__(self, state_dim=60, action_dim=3, hidden_dim=256, lr=3e-4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural network training")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Networks
        self.actor = SACActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = SACCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = SACCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic1 = SACCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = SACCriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Temperature parameter
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim
        
        # Training stats
        self.training_steps = 0
        
    def get_action(self, state, deterministic=False):
        """Get action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic)
        return action.cpu().numpy().flatten()
    
    def update(self, batch, gamma=0.99, tau=0.005):
        """Update networks with batch of experiences"""
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_value = reward + (1 - done) * gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_prob - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1, tau)
        self._soft_update(self.target_critic2, self.critic2, tau)
        
        self.training_steps += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha': self.log_alpha.exp().item()
        }
    
    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    
    def save(self, path, metadata=None):
        """Save model checkpoint with optional metadata"""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'training_steps': self.training_steps
        }
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.training_steps = checkpoint.get('training_steps', 0)
        print(f"Model loaded from {path}")
        return checkpoint.get('metadata', {})

class TrainingMetrics:
    """Persistent training metrics and logging"""
    
    def __init__(self, log_dir='training_logs', model_dir='models'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Create timestamped session
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.log_dir / f'training_session_{self.session_id}.csv'
        self.log_path = self.log_dir / f'training_log_{self.session_id}.txt'
        self.telemetry_path = self.log_dir / f'telemetry_{self.session_id}.csv'
        
        # Initialize episode CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Steps', 'Time_s', 'Total_Reward', 'Avg_Reward', 
                           'Final_Distance_m', 'Max_Speed_mph', 'Crashes', 'Final_Damage',
                           'Buffer_Size', 'Timestamp'])
        
        # Initialize telemetry CSV (detailed step data for threshold analysis)
        with open(self.telemetry_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Step', 'Timestamp',
                           'Speed_ms', 'Throttle', 'Steering', 'Brake',
                           'Damage', 'Damage_Increase', 'Total_Damage',
                           'Gx', 'Gy', 'Gz',
                           'Distance_From_Origin', 'Distance_From_Checkpoint',
                           'Position_X', 'Position_Y', 'Position_Z',
                           'Position_Delta', 'Stationary_Timer',
                           'Crash_Detected', 'Stationary_Timeout', 'Flipped',
                           'Reward', 'Event_Type'])
        
        # Best model tracking
        self.best_distance = 0.0
        self.best_reward = float('-inf')
        self.best_episode = 0
        
        print(f"[CHART] Episode metrics: {self.csv_path}")
        print(f"[CHART] Telemetry data: {self.telemetry_path}")
        print(f"[LOG] Text logs: {self.log_path}")
        print(f"[SAVE] Models directory: {self.model_dir}")
    
    def log_episode(self, episode, steps, time_s, total_reward, avg_reward,
                   final_distance, max_speed_mph, crashes, final_damage, buffer_size):
        """Log episode metrics to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, steps, f'{time_s:.1f}', f'{total_reward:.2f}', 
                           f'{avg_reward:.2f}', f'{final_distance:.1f}', f'{max_speed_mph:.1f}',
                           crashes, f'{final_damage:.2f}', buffer_size, 
                           datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    
    def log_telemetry(self, episode, step, state, damage_increase, position_delta,
                     stationary_timer, info, reward, event_type=''):
        """Log detailed telemetry for threshold analysis"""
        with open(self.telemetry_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, step, datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                f'{state.speed:.2f}', f'{state.throttle:.2f}', f'{state.steering:.2f}', f'{state.brake:.2f}',
                f'{state.damage:.3f}', f'{damage_increase:.3f}', f'{state.damage:.3f}',
                f'{state.gx:.2f}', f'{state.gy:.2f}', f'{state.gz:.2f}',
                f'{state.distance_from_origin:.1f}', f'{state.distance_from_checkpoint:.1f}',
                f'{state.position[0]:.1f}', f'{state.position[1]:.1f}', f'{state.position[2]:.1f}',
                f'{position_delta:.3f}', f'{stationary_timer:.1f}',
                info.get('crash_detected', False), info.get('stationary_timeout', False),
                info.get('flipped', False),
                f'{reward:.2f}', event_type
            ])
    
    def log_text(self, message):
        """Append text to log file"""
        with open(self.log_path, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{timestamp}] {message}\n')
    
    def check_best_model(self, episode, distance, reward):
        """Check if this is a new best model"""
        is_best_distance = distance > self.best_distance
        is_best_reward = reward > self.best_reward
        
        if is_best_distance:
            self.best_distance = distance
            self.best_episode = episode
        
        if is_best_reward:
            self.best_reward = reward
        
        return is_best_distance, is_best_reward
    
    def get_summary(self):
        """Get training summary"""
        return {
            'best_distance': self.best_distance,
            'best_reward': self.best_reward,
            'best_episode': self.best_episode,
            'session_id': self.session_id
        }

class ExperienceReplay:
    """Experience replay buffer"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return {
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(action),
            'reward': torch.FloatTensor(reward).unsqueeze(1),
            'next_state': torch.FloatTensor(next_state),
            'done': torch.FloatTensor(done).unsqueeze(1)
        }
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# PERSISTENT BEAMNG ENVIRONMENT
# ============================================================================

@dataclass
class TrainingState:
    """Compact state representation for neural network - Phase 5A (31D)"""
    # Position and velocity (6)
    position: np.ndarray  # x, y, z
    velocity: np.ndarray  # vx, vy, vz
    
    # Distance tracking (3)
    distance_from_origin: float
    distance_from_checkpoint: float
    distance_delta: float
    
    # Vehicle dynamics (15)
    speed: float
    throttle: float
    steering: float
    brake: float
    rpm: float
    gear: float
    wheelspeed: float
    gx: float  # G-forces
    gy: float
    gz: float
    damage: float
    abs_active: float
    esc_active: float
    tcs_active: float
    fuel: float
    
    # Episode info (3)
    episode_time: float
    crash_count: float
    stationary_time: float
    
    # PHASE 5A: Navigation features (4D) - NEW
    distance_to_waypoint: float = 0.0
    bearing_to_waypoint: float = 0.0
    heading_error: float = 0.0
    waypoint_reached: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert to 31D numpy array for neural network (Phase 5A)"""
        return np.array([
            # Position (3)
            self.position[0], self.position[1], self.position[2],
            # Velocity (3)
            self.velocity[0], self.velocity[1], self.velocity[2],
            # Distance (3)
            self.distance_from_origin, self.distance_from_checkpoint, self.distance_delta,
            # Dynamics (15)
            self.speed, self.throttle, self.steering, self.brake, self.rpm,
            self.gear, self.wheelspeed, self.gx, self.gy, self.gz,
            self.damage, self.abs_active, self.esc_active, self.tcs_active, self.fuel,
            # Episode (3)
            self.episode_time, self.crash_count, self.stationary_time,
            # PHASE 5A: Navigation (4)
            self.distance_to_waypoint, self.bearing_to_waypoint,
            self.heading_error, self.waypoint_reached
        ], dtype=np.float32)

class PersistentHighwayEnvironment:
    """
    Highway training environment that connects to running BeamNG instance
    Uses scenario restart instead of full reload for speed
    """
    
    def __init__(self, host='localhost', port=25252):
        self.host = host
        self.port = port
        self.bng = None
        self.vehicle = None
        self.scenario = None
        self.metrics = None  # Will be set by training loop for telemetry logging
        self.bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        
        # Episode tracking
        self.episode_origin = None
        self.current_checkpoint = None
        self.last_safe_checkpoint = None  # Last known good position on road
        self.last_safe_orientation = None  # Orientation at last safe checkpoint
        self.episode_start_time = 0.0
        self.last_position = None
        self.last_damage = 0.0
        self.steps_since_checkpoint = 0
        
        # Statistics
        self.episode_count = 0
        self.total_steps = 0
        self.crash_count = 0
        self.max_distance = 0.0
        self.stationary_timer = 0.0
        self.already_recovered = False  # Flag to skip reset_episode() after crash recovery
        
        # Control smoothness tracking (for penalty calculation)
        self.last_throttle = 0.0
        self.last_steering = 0.0
        self.last_brake = 0.0
    
    def _calculate_waypoint_features(self, vehicle_pos):
        """Calculate navigation features to current waypoint (Phase 5A)"""
        # Get current waypoint from route
        if self.current_waypoint_index >= len(WAYPOINT_ROUTE):
            # All waypoints reached - navigate to last waypoint
            self.current_waypoint_index = len(WAYPOINT_ROUTE) - 1
        
        waypoint = WAYPOINT_ROUTE[self.current_waypoint_index]
        target_pos = waypoint['pos']
        radius = waypoint['radius']
        
        # Distance to waypoint (2D, ignore Z)
        dx = target_pos[0] - vehicle_pos[0]
        dy = target_pos[1] - vehicle_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Bearing to waypoint (radians, 0 = north)
        bearing = np.arctan2(dx, dy)
        
        # Vehicle heading from direction vector
        dir_vec = self.vehicle.state.get('dir', None)
        if dir_vec is not None:
            vehicle_heading = np.arctan2(dir_vec[0], dir_vec[1])
        else:
            vehicle_heading = 0.0
        
        # Heading error (normalized to [-pi, pi])
        heading_error = bearing - vehicle_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Check if waypoint reached
        reached = 1.0 if distance < radius else 0.0
        
        return distance, bearing, heading_error, reached
        
    def connect(self, auto_launch=True):
        """Connect to already-running BeamNG instance (or launch if needed)"""
        print("=" * 60)
        print("Phase 4C: Neural Highway Training (Persistent Instance)")
        print("=" * 60)
        print("Searching for BeamNG instance...")
        print()
        
        try:
            set_up_simple_logging()
            
            # Try common ports in order
            common_ports = [64256, 25252, self.port] if self.port not in [64256, 25252] else [64256, 25252]
            
            for port in common_ports:
                try:
                    print(f"Trying port {port}...", end=" ")
                    self.bng = BeamNGpy(self.host, port, home=self.bng_home)
                    self.bng.open(launch=False)
                    self.port = port  # Update to working port
                    print(f"[OK] Connected on port {port}!")
                    return True
                except:
                    print("[X]")
                    continue
            
            # No running instance found, launch new one
            if auto_launch:
                print("\nNo running instance found on any port.")
                print("Launching BeamNG.drive (this will take ~30 seconds)...")
                self.bng = BeamNGpy(self.host, 64256, home=self.bng_home)
                self.bng.open(launch=True)
                self.port = 64256
                print("[OK] BeamNG launched successfully on port 64256!")
                return True
            else:
                raise Exception("No running BeamNG instance found")
                    
        except Exception as e:
            print(f"\nERROR: Could not connect to BeamNG: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure BeamNG.drive is installed at:", self.bng_home)
            print("2. Try running BeamNG manually first")
            print("3. Ports tried:", common_ports)
            return False
    
    def setup_scenario(self, map_name='west_coast_usa'):
        """Setup highway scenario (only once)"""
        print(f"\nSetting up scenario on map: {map_name}")
        
        try:
            self.scenario = Scenario(map_name, 'neural_highway_training',
                                    description='Phase 4C: Neural Training')
            
            # Randomize vehicle model for varied training (different handling/acceleration)
            vehicles = ['etk800', 'etkc', 'etki', 'etk800_premium', 'vivace', 
                       'moonhawk', 'bluebuck', 'hopper', 'pessima', 'covet']
            selected_vehicle = np.random.choice(vehicles)
            print(f"[RAND] Selected vehicle: {selected_vehicle} (randomized for variety)")
            
            self.vehicle = Vehicle('ai_neural', model=selected_vehicle, license='NEURAL')
            
            # Attach sensors (basic sensors only - no BeamNG.tech license required)
            self.vehicle.sensors.attach('electrics', Electrics())
            self.vehicle.sensors.attach('damage', Damage())
            self.vehicle.sensors.attach('gforces', GForces())
            print("[OK] Basic sensors attached (Electrics, Damage, GForces)")
            
            # Use proven spawn coordinates from Phase 2 (west_coast_usa)
            print(f"Using validated spawn coordinates for {map_name}")
            spawn_pos = (-717.121, 101, 118.675)
            spawn_rot = (0, 0, 0.3826834, 0.9238795)
            self.scenario.add_vehicle(self.vehicle, pos=spawn_pos, rot_quat=spawn_rot)
            
            print("Building scenario...")
            self.scenario.make(self.bng)
            
            print("Setting performance optimizations...")
            # Speed up physics and reduce graphics load
            self.bng.settings.set_deterministic(60)  # 60Hz physics (consistent)
            
            # Disable UI overlays for performance
            try:
                self.bng.settings.set_particles_enabled(False)
                self.bng.settings.set_shadows_enabled(False)
            except:
                pass  # Older BeamNGpy versions may not have these
            
            print("Loading scenario (this may take 30-60 seconds on first load)...")
            self.bng.scenario.load(self.scenario)
            print("[OK] Scenario loaded")
            
            print("Starting scenario...")
            self.bng.scenario.start()
            print("[OK] Scenario started")
            
            print("Physics stabilization (5 seconds)...")
            time.sleep(5)  # Increased wait for physics to settle
            print("[OK] Physics stabilized")
            
            # Release parking brake and give initial throttle burst
            self.vehicle.control(parkingbrake=0, throttle=1.0, steering=0, brake=0)
            print("[CAR] Parking brake released + initial throttle burst")
            time.sleep(0.5)
            
            # Initialize tracking
            self.vehicle.sensors.poll()
            pos = self.vehicle.state['pos']
            self.episode_origin = np.array(pos, dtype=np.float32)
            self.current_checkpoint = self.episode_origin.copy()
            self.last_safe_checkpoint = self.episode_origin.copy()
            self.last_safe_orientation = (0, 0, 0.3826834, 0.9238795)  # Original spawn orientation
            self.last_position = self.episode_origin.copy()
            self.episode_start_time = time.time()
            self.steps_since_checkpoint = 0
            
            print(f"\nScenario ready!")
            print(f"Starting position: {pos}")
            return True
            
        except Exception as e:
            print(f"ERROR: Scenario setup failed: {e}")
            print(f"\nTrying fallback map: west_coast_usa")
            if map_name != 'west_coast_usa':
                return self.setup_scenario('west_coast_usa')
            return False
    
    def soft_reset(self, position=None):
        """Ultra-fast reset: teleport vehicle without reloading scenario
        
        Args:
            position: Optional position to reset to. If None, uses highway spawn.
        """
        try:
            # Use validated west_coast_usa spawn coordinates
            self.vehicle.teleport(
                pos=(-717.121, 101, 118.675),
                rot_quat=(0, 0, 0.3826834, 0.9238795),
                reset=True)
            time.sleep(0.3)  # Minimal settling time
            
            # Release parking brake and throttle burst
            self.vehicle.control(throttle=0.7, steering=0, brake=0, parkingbrake=0)
            time.sleep(0.3)
            
            # Reset tracking variables
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
            
            return True
            
        except Exception as e:
            print(f"[WARN] Soft reset failed: {e}, falling back to scenario restart")
            return False
    
    def reset_episode(self):
        """Fast episode reset using scenario restart"""
        print("\n--- Episode Reset ---")
        
        try:
            # Use BeamNG's restart function (faster than full reload)
            self.bng.scenario.restart()
            time.sleep(1)  # Reduced wait time - physics settles quickly
            
            # Release parking brake (automatic)
            self.vehicle.control(parkingbrake=0)
            time.sleep(0.2)
            
            # Reset tracking
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
            
            # Reset control smoothness tracking
            self.last_throttle = 0.0
            self.last_steering = 0.0
            self.last_brake = 0.0
            
            # PHASE 5A: Reset waypoint progression
            self.current_waypoint_index = 0
            self.waypoints_reached = 0
            
            # Initial throttle burst to overcome inertia
            self.vehicle.control(throttle=1.0, steering=0, brake=0, parkingbrake=0)
            time.sleep(0.5)
            wp_name = WAYPOINT_ROUTE[0]['name']
            print(f"Episode {self.episode_count} started - Target: {wp_name}")
            
        except Exception as e:
            print(f"WARNING: Reset failed: {e}")
            print("Attempting manual recovery...")
            self.vehicle.recover()
            time.sleep(1)
    
    def get_state(self) -> TrainingState:
        """Get current state for neural network"""
        self.vehicle.sensors.poll()
        
        # Vehicle state
        pos = np.array(self.vehicle.state['pos'], dtype=np.float32)
        vel = np.array(self.vehicle.state['vel'], dtype=np.float32)
        speed = float(np.linalg.norm(vel))
        
        # Electrics
        electrics = self.vehicle.sensors['electrics']
        damage_data = self.vehicle.sensors['damage']
        gforces = self.vehicle.sensors['gforces']
        
        # Parse damage (can be nested dict or direct dict of floats)
        try:
            if damage_data and isinstance(damage_data, dict):
                # Extract all numeric damage values, handling nested dicts
                damage_values = []
                for value in damage_data.values():
                    if isinstance(value, (int, float)):
                        damage_values.append(value)
                    elif isinstance(value, dict):
                        # Nested dict - extract numeric values
                        damage_values.extend([v for v in value.values() if isinstance(v, (int, float))])
                damage = max(damage_values) if damage_values else 0.0
            else:
                damage = 0.0
        except Exception as e:
            print(f"Warning: Damage parsing failed: {e}, using 0.0")
            damage = 0.0
        
        # Distance calculations
        distance_from_origin = float(np.linalg.norm(pos - self.episode_origin))
        distance_from_checkpoint = float(np.linalg.norm(pos - self.current_checkpoint))
        distance_delta = float(np.linalg.norm(pos - self.last_position))
        
        self.last_position = pos.copy()
        
        # Update max distance
        if distance_from_origin > self.max_distance:
            self.max_distance = distance_from_origin
        
        # Parse gear (can be string like 'P', 'R', 'N', 'D', '1', '2', etc.)
        gear_value = electrics.get('gear', 0)
        if isinstance(gear_value, str):
            # Convert string gears to numeric: P=-1, R=-2, N=0, D=1, numbers as-is
            gear_map = {'P': -1, 'R': -2, 'N': 0, 'D': 1, 'M': 1}
            try:
                gear_float = float(gear_map.get(gear_value, float(gear_value)))
            except (ValueError, TypeError):
                gear_float = 0.0
        else:
            gear_float = float(gear_value) if gear_value is not None else 0.0
        
        # PHASE 5A: Calculate waypoint navigation features
        dist_to_wp, bearing_to_wp, heading_err, wp_reached = self._calculate_waypoint_features(pos)
        
        return TrainingState(
            position=pos,
            velocity=vel,
            distance_from_origin=distance_from_origin,
            distance_from_checkpoint=distance_from_checkpoint,
            distance_delta=distance_delta,
            speed=speed,
            throttle=electrics.get('throttle', 0.0),
            steering=electrics.get('steering', 0.0),
            brake=electrics.get('brake', 0.0),
            rpm=electrics.get('rpm', 0.0),
            gear=gear_float,
            wheelspeed=electrics.get('wheelspeed', 0.0),
            gx=gforces.get('gx', 0.0),
            gy=gforces.get('gy', 0.0),
            gz=gforces.get('gz', 0.0),
            damage=damage,
            abs_active=electrics.get('abs_active', 0.0),
            esc_active=electrics.get('esc_active', 0.0),
            tcs_active=electrics.get('tcs_active', 0.0),
            fuel=electrics.get('fuel', 1.0),
            episode_time=time.time() - self.episode_start_time,
            crash_count=float(self.crash_count),
            stationary_time=self.stationary_timer,
            # PHASE 5A: Navigation features
            distance_to_waypoint=dist_to_wp,
            bearing_to_waypoint=bearing_to_wp,
            heading_error=heading_err,
            waypoint_reached=wp_reached
        )
    
    def step(self, action: np.ndarray):
        """Execute action and return (state, reward, done, info)"""
        # Get current state before action
        current_state = self.get_state()
        
        # Apply action (throttle, steering, brake)
        throttle = float(np.clip(action[0], 0, 1))
        steering = float(np.clip(action[1], -1, 1))
        brake = float(np.clip(action[2], 0, 1))
        
        # ENFORCE: Prevent simultaneous throttle+brake (mutual exclusivity)
        if throttle > 0.1 and brake > 0.1:
            # If both are active, zero out the smaller one
            if throttle > brake:
                brake = 0.0  # Keep throttle, cancel brake
            else:
                throttle = 0.0  # Keep brake, cancel throttle
        
        # Calculate control smoothness (penalize jerky inputs)
        throttle_delta = abs(throttle - self.last_throttle)
        steering_delta = abs(steering - self.last_steering)
        brake_delta = abs(brake - self.last_brake)
        
        # Store for next step
        self.last_throttle = throttle
        self.last_steering = steering
        self.last_brake = brake
        
        # Control vehicle (parking brake always off - AI doesn't control)
        self.vehicle.control(throttle=throttle, steering=steering, brake=brake, parkingbrake=0)
        time.sleep(0.3)  # 3.3 Hz control (faster than 2 Hz, still stable)
        
        # Track safe checkpoints (update every 10m of progress without damage)
        self.steps_since_checkpoint += 1
        if self.steps_since_checkpoint > 30:  # ~10 seconds of driving
            current_state_check = self.get_state()
            # Only checkpoint if low damage and making progress
            if current_state_check.damage < 0.1 and current_state_check.speed > 2.0:
                self.last_safe_checkpoint = current_state_check.position.copy()
                # Get current orientation from vehicle state
                if 'dir' in self.vehicle.state:
                    # dir gives us forward direction vector, convert to quaternion
                    # For now, use a forward-facing orientation
                    self.last_safe_orientation = (0, 0, 0, 1)  # Upright, will adjust based on velocity
                self.steps_since_checkpoint = 0
        
        # Get next state
        next_state = self.get_state()
        
        # PHASE 5A: Check if waypoint reached and advance
        if next_state.waypoint_reached > 0.5 and self.current_waypoint_index < len(WAYPOINT_ROUTE):
            self.waypoints_reached += 1
            waypoint_name = WAYPOINT_ROUTE[self.current_waypoint_index]['name']
            print(f"\n  ✓ WAYPOINT REACHED: {waypoint_name} ({self.waypoints_reached}/{self.total_waypoints})")
            
            # Advance to next waypoint
            self.current_waypoint_index += 1
            if self.current_waypoint_index < len(WAYPOINT_ROUTE):
                next_wp = WAYPOINT_ROUTE[self.current_waypoint_index]
                print(f"  → Next target: {next_wp['name']}")
        
        # Calculate reward (pass control deltas for smoothness penalty)
        reward, info = self._calculate_reward(
            current_state, next_state, 
            throttle, steering, brake,
            throttle_delta, steering_delta, brake_delta
        )
        
        # Calculate position_delta and damage_increase for telemetry
        position_delta = np.linalg.norm(next_state.position - current_state.position)
        damage_increase = next_state.damage - self.last_damage
        info['position_delta'] = position_delta
        info['damage_increase'] = damage_increase
        
        # Log telemetry BEFORE recovery (captures actual crash state)
        if hasattr(self, 'metrics') and self.metrics:
            event_type = ''
            if info.get('crash_detected'): event_type = 'CRASH'
            elif info.get('stationary_timeout'): event_type = 'STUCK'
            elif info.get('speeding'): event_type = 'SPEEDING'
            
            self.metrics.log_telemetry(
                episode=self.episode_count, step=self.total_steps, state=next_state,
                damage_increase=damage_increase, position_delta=position_delta,
                stationary_timer=self.stationary_timer, info=info,
                reward=reward, event_type=event_type
            )
        
        # Check for episode end
        done = info['crash_detected'] or info['stationary_timeout']
        
        # Handle crash - recover to road center at current position
        if info['crash_detected']:
            self.crash_count += 1
            crash_type = "FLIPPED" if info.get('flipped') else "DAMAGE"
            speed_info = f" (speed: {next_state.speed:.1f} m/s)"
            print(f"  [CRASH] {crash_type} CRASH #{self.crash_count} at {next_state.distance_from_origin:.1f}m{speed_info} - recovering...")
            
            # Quick recovery: use current position but reset orientation to upright
            self._recover_to_road_center(next_state.position)
            
            # Update checkpoint to crash location
            self.current_checkpoint = next_state.position.copy()
            self.already_recovered = True  # Flag to skip reset_episode()
            done = True
        
        # Handle stationary timeout - recover to road center
        if info['stationary_timeout']:
            print(f"  [TIME] Stationary timeout after {self.stationary_timer:.1f}s - recovering...")
            self._recover_to_road_center(current_state.position)
            self.already_recovered = True  # Flag to skip reset_episode()
            done = True
        
        self.total_steps += 1
        
        return next_state.to_vector(), reward, done, info
    
    def _recover_to_road_center(self, position):
        """Full vehicle recovery: always reset to highway spawn point"""
        try:
            print("    [TOOL] Executing full vehicle recovery...")
            print(f"    [PIN] Recovering to spawn point")
            
            # Use validated west_coast_usa spawn coordinates
            self.vehicle.teleport(
                pos=(-717.121, 101, 118.675),
                rot_quat=(0, 0, 0.3826834, 0.9238795),
                reset=True)
            time.sleep(2.0)  # Longer settling time - let physics stabilize
            
            # Poll sensors to confirm new position
            self.vehicle.sensors.poll()
            new_pos = self.vehicle.state['pos']
            
            # Reset all tracking
            self.last_damage = 0.0
            self.stationary_timer = 0.0
            self.steps_since_checkpoint = 0
            
            # Very gentle initial throttle - let AI take over
            self.vehicle.control(throttle=0.2, steering=0, brake=0, parkingbrake=0)
            time.sleep(1.0)  # Extra time to start moving
            
            print(f"    [OK] Vehicle recovered to spawn ({new_pos[0]:.1f}, {new_pos[1]:.1f}, {new_pos[2]:.1f})")
            print(f"    [OK] Damage reset, vehicle upright, ready to continue")
            
        except Exception as e:
            print(f"    [WARN] Recovery failed, using fallback: {e}")
            self.vehicle.recover()  # Fallback to BeamNG's built-in recovery
            time.sleep(1)
    
    def _calculate_reward(self, current_state, next_state, 
                         throttle, steering, brake,
                         throttle_delta, steering_delta, brake_delta):
        """Calculate reward with waypoint navigation + smoothness penalties (Phase 5A)"""
        reward = 0.0
        info = {
            'crash_detected': False, 
            'stationary_timeout': False,
            'flipped': False,
            'speeding': False,
            'stuck': False,
            'speed': next_state.speed,
            'distance_progress': 0.0,
            'waypoint_reached': False  # Phase 5A
        }
        
        # PHASE 5A: Waypoint progress reward (PRIORITY)
        waypoint_progress = current_state.distance_to_waypoint - next_state.distance_to_waypoint
        if waypoint_progress > 0:
            reward += waypoint_progress * 1.0  # 1.0 points per meter toward waypoint
            info['waypoint_progress'] = waypoint_progress
        
        # PHASE 5A: Heading alignment bonus
        heading_alignment = np.cos(next_state.heading_error)
        if waypoint_progress > 0:  # Only when making progress
            reward += heading_alignment * 0.3  # Up to +0.3 for pointing correctly
        
        # PHASE 5A: Waypoint reached bonus (scales with progress)
        if next_state.waypoint_reached > 0.5:
            # Bonus increases for later waypoints (encourages completion)
            waypoint_bonus = 100.0 * (self.current_waypoint_index + 1)
            reward += waypoint_bonus
            info['waypoint_reached'] = True
            info['waypoint_bonus'] = waypoint_bonus
            
            # Route completion bonus (all waypoints)
            if self.waypoints_reached >= self.total_waypoints:
                reward += 500.0  # MASSIVE bonus for full route!
                info['route_completed'] = True
        
        # Distance reward (secondary - kept for exploration)
        distance_progress = next_state.distance_from_checkpoint - current_state.distance_from_checkpoint
        info['distance_progress'] = distance_progress
        
        if distance_progress > 0:
            reward += distance_progress * 0.2  # Reduced from 0.5 (waypoint is priority)
        
        # Speed bonus (when making progress) - BUT penalize excessive speed
        speed_limit_ms = 30.0  # ~67 mph for highway (was 20 m/s / 45 mph - too conservative)
        if distance_progress > 0 and next_state.speed > 1.0:
            if next_state.speed <= speed_limit_ms:
                reward += next_state.speed * 0.1  # Bonus for good speed
            else:
                # Penalty for speeding - linear penalty above limit
                overspeed = next_state.speed - speed_limit_ms
                reward -= overspeed * 0.5  # Penalize excess speed
                info['speeding'] = True
        
        # ========== SMOOTHNESS PENALTIES (NEW) ==========
        
        # 1. Penalize excessive steering changes (jerky steering)
        if steering_delta > 0.3:  # Steering changed by more than 30%
            reward -= steering_delta * 2.0  # Penalty scales with jerkiness
            if steering_delta > 0.5:
                info['jerky_steering'] = True
        
        # 2. Penalize simultaneous brake + throttle (brake riding) - HEAVY PENALTY
        if throttle > 0.3 and brake > 0.3:  # Both inputs active
            brake_riding_penalty = (throttle * brake) * 15.0  # INCREASED from 5.0 - AI rides brakes too much
            reward -= brake_riding_penalty
            info['brake_riding'] = True
        
        # 3. Penalize excessive brake usage when not needed (speed already low)
        if brake > 0.5 and next_state.speed < 5.0:  # Heavy braking at low speed
            reward -= 1.0
            info['unnecessary_braking'] = True
        
        # 4. Reward smooth throttle control
        if throttle_delta < 0.1:  # Smooth throttle changes
            reward += 0.1
        
        # 5. Penalize excessive steering magnitude (oversteering)
        if abs(steering) > 0.7 and next_state.speed > 10.0:  # Hard steering at speed
            reward -= abs(steering) * 0.5
            info['oversteering'] = True
        
        # ========== END SMOOTHNESS PENALTIES ==========
        
        # Crash detection (damage-based) - Check both damage increase AND total damage
        damage_increase = next_state.damage - self.last_damage
        
        # NOTE: Damage scale analysis from telemetry shows:
        # - Normal driving: 0.000-0.001
        # - Minor scrapes: 0.001-0.05
        # - Major crashes: 100-3000+ (massive spikes)
        # Thresholds adjusted based on actual observed values
        
        # Priority 1: High total damage (vehicle is wrecked)
        if next_state.damage > 100.0:  # Total damage threshold (was 0.5, telemetry shows 3000+ for wrecks)
            reward -= 30.0
            info['crash_detected'] = True
            self.last_damage = next_state.damage
            print(f"    [WARN] VEHICLE WRECKED: total damage {next_state.damage:.2f}")
        # Priority 2: Moderate damage increase (significant crash)
        elif damage_increase > 50.0:  # Major damage spike (telemetry shows crashes = 100-3000)
            reward -= 30.0
            info['crash_detected'] = True
            self.last_damage = next_state.damage
            print(f"    [WARN] CRASH DAMAGE: {damage_increase:.2f} increase (total: {next_state.damage:.2f})")
        # Priority 3: Minor damage increase (scrapes/bumps)
        elif damage_increase > 10.0:  # Minor damage (small impacts, was 0.15)
            reward -= 10.0  # Moderate penalty for impacts
            self.last_damage = next_state.damage
            print(f"    [BUMP] Minor damage: {damage_increase:.2f} increase (total: {next_state.damage:.2f})")
        # Priority 4: Tiny damage (curb scrapes - just track, minimal penalty)
        elif damage_increase > 0.5:  # Very minor scrapes (was 0.05)
            reward -= 2.0  # Small penalty, keep driving
            self.last_damage = next_state.damage
        
        # Orientation check (flipped/tilted vehicle) - DISABLED FOR NOW
        # G-forces change during acceleration/braking, causing false positives
        # Only check for extreme cases where car is literally upside down
        # gz: -1 = upright, +1 = upside down, 0 = on side
        
        # Only trigger on EXTREME orientation (nearly impossible during normal driving)
        is_completely_flipped = next_state.gz > 0.7  # Almost completely upside down
        is_sideways = abs(next_state.gx) > 1.5 and abs(next_state.gy) > 1.5  # Extreme multi-axis tilt
        
        if is_completely_flipped and is_sideways:  # Require BOTH conditions
            reward -= 20.0
            info['crash_detected'] = True
            info['flipped'] = True
            print(f"    [WARN] VEHICLE FLIPPED: Gz={next_state.gz:.2f}, Gx={next_state.gx:.2f}, Gy={next_state.gy:.2f}")
        
        # Stationary/Stuck detection - POSITION BASED (not speed!)
        # Check actual position change, not wheel speed (catches wheel-spinning in ditches)
        position_delta = np.linalg.norm(next_state.position - current_state.position)
        info['position_delta'] = position_delta  # For logging
        
        # If barely moved (< 0.2m in 0.3s = < 0.67 m/s actual movement) - RELAXED
        if position_delta < 0.2:  # Was 0.3, now more lenient
            self.stationary_timer += 0.3
            reward -= 0.2  # Reduced penalty (was 0.5)
            
            # Extra penalty for wheel-spinning (high speed but no movement)
            if next_state.speed > 10.0 and position_delta < 0.05:  # Raised threshold from 5.0
                reward -= 1.0  # Reduced from 2.0
                info['stuck'] = True
            
            if self.stationary_timer > 6.0:  # Recover if stuck for 6 seconds
                info['stationary_timeout'] = True
                reward -= 20.0
                print(f"  ! STUCK/STATIONARY after {self.stationary_timer:.1f}s ")
                print(f"    (wheelspeed: {next_state.speed:.1f} m/s, actual movement: {position_delta:.2f}m)")
        else:
            self.stationary_timer = 0.0
        
        info['reward'] = reward
        
        return reward, info
    
    def close(self):
        """Properly close connection (keeps BeamNG running)"""
        if self.bng:
            try:
                print("\n" + "="*60)
                print("Closing training session...")
                print(f"  Total steps executed: {self.total_steps}")
                print(f"  Total episodes: {self.episode_count}")
                print(f"  Max distance achieved: {self.max_distance:.1f}m")
                print("="*60)
                
                # Close the connection properly
                self.bng.disconnect()
                self.bng = None
                print("[OK] Cleanly disconnected from BeamNG")
                print("  (BeamNG instance still running for next session)")
            except Exception as e:
                print(f"Warning during disconnect: {e}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_highway_neural(episodes=100, batch_size=64, replay_start_size=3000):
    """Main training loop with persistent metrics and best model tracking"""
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available. Install with: pip install torch")
        return
    
    # Initialize environment
    env = PersistentHighwayEnvironment()
    
    if not env.connect():
        return
    
    if not env.setup_scenario():
        return
    
    # Initialize agent and replay buffer
    state_dim = 31  # Phase 5A: 27D base + 4D navigation (waypoint features)
    action_dim = 3  # throttle, steering, brake
    
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    replay_buffer = ExperienceReplay(capacity=100000)
    
    # Initialize metrics tracking
    metrics = TrainingMetrics()
    env.metrics = metrics  # Give environment access to metrics for telemetry logging
    
    # Check for existing best model and load it
    best_model_path = metrics.model_dir / 'highway_best.pth'
    
    if best_model_path.exists():
        print("\n[LOOP] Found existing best model - loading to continue training...")
        metadata = agent.load(str(best_model_path))
        if metadata:
            metrics.best_distance = metadata.get('best_distance', 0.0)
            metrics.best_reward = metadata.get('best_reward', float('-inf'))
            metrics.best_episode = metadata.get('best_episode', 0)
            print(f"   Previous best: {metrics.best_distance:.1f}m distance, {metrics.best_reward:.1f} reward")
    
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Replay buffer: {replay_buffer.capacity}")
    print(f"Batch size: {batch_size}")
    print(f"Metrics: {metrics.csv_path.name}")
    print("=" * 60 + "\n")
    
    metrics.log_text(f"Training started: {episodes} episodes")
    total_reward_history = []
    
    try:
        for episode in range(episodes):
            # Only reset if we didn't already recover from crash
            if episode > 0 and not env.already_recovered:
                env.reset_episode()
            
            # Clear recovery flag for next episode
            env.already_recovered = False
            
            state = env.get_state().to_vector()
            episode_reward = 0
            episode_steps = 0
            episode_start = time.time()
            
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            if len(replay_buffer) < replay_start_size:
                remaining = replay_start_size - len(replay_buffer)
                print(f"[RAND] Random Exploration Mode ({len(replay_buffer)}/{replay_start_size} samples, {remaining} to go)")
            else:
                print(f" Neural Network Training Mode (buffer: {len(replay_buffer)}/{replay_buffer.capacity})")
            
            # Episode loop
            while True:
                # Get action
                if len(replay_buffer) < replay_start_size:
                    # Random exploration - BALANCED (not just full throttle!)
                    action = np.array([
                        np.random.uniform(0.3, 1.0),    # Full range throttle
                        np.random.uniform(-0.8, 0.8),   # Wide steering range
                        np.random.uniform(0, 0.3)       # Sometimes brake!
                    ])
                else:
                    # Use policy
                    action = agent.get_action(state, deterministic=False)
                
                # Execute step (telemetry now logged inside step() before recovery)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                replay_buffer.push(state, action, reward, next_state, float(done))
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Print progress every step during exploration
                if len(replay_buffer) < replay_start_size:
                    # Build warning flags
                    flags = []
                    if info.get('speeding'): flags.append("[SPEED]")
                    if info.get('stuck'): flags.append("[STUCK]")
                    if info.get('brake_riding'): flags.append("[BRAKE+THROTTLE]")
                    if info.get('jerky_steering'): flags.append("[JERKY]")
                    if info.get('oversteering'): flags.append("[OVERSTEER]")
                    flags_str = " ".join(flags) if flags else ""
                    
                    print(f"  Explore Step {episode_steps}: Speed={info.get('speed', 0):.1f} m/s {flags_str}, "
                          f"Distance={info.get('distance_progress', 0):.1f}m, "
                          f"Reward={reward:.2f}, Action=[T:{action[0]:.2f} S:{action[1]:.2f} B:{action[2]:.2f}]")
                    
                    # Detailed debug info every 5 steps
                    if episode_steps % 5 == 0:
                        state_obj = env.get_state()
                        pos_delta = info.get('position_delta', 0)
                        print(f"    DEBUG: Pos=({state_obj.position[0]:.1f},{state_obj.position[1]:.1f},{state_obj.position[2]:.1f}), "
                              f"PosDelta={pos_delta:.2f}m, Gz={state_obj.gz:.2f}, "
                              f"Damage={state_obj.damage:.2f}, StationaryTimer={env.stationary_timer:.1f}s")
                
                # Train agent
                if len(replay_buffer) >= replay_start_size and episode_steps % 2 == 0:
                    batch = replay_buffer.sample(batch_size)
                    losses = agent.update(batch)
                    
                    if episode_steps % 20 == 0:  # Update every 20 steps (was 5)
                        print(f"   Neural Training Step {episode_steps}: Reward={reward:.2f}, "
                              f"Actor Loss={losses['actor_loss']:.4f}, Critic Loss={losses['critic1_loss']:.4f}, "
                              f"Distance={info.get('distance_progress', 0):.1f}m")
                
                if done or episode_steps > 200:  # Max 200 steps per episode
                    break
            
            # Episode summary
            episode_time = time.time() - episode_start
            avg_reward = episode_reward / episode_steps if episode_steps > 0 else 0
            final_state = env.get_state()
            max_speed_mph = final_state.speed * 2.237
            
            # Log to CSV
            metrics.log_episode(
                episode=episode + 1,
                steps=episode_steps,
                time_s=episode_time,
                total_reward=episode_reward,
                avg_reward=avg_reward,
                final_distance=final_state.distance_from_origin,
                max_speed_mph=max_speed_mph,
                crashes=env.crash_count,
                final_damage=final_state.damage,
                buffer_size=len(replay_buffer)
            )
            
            # Check for best model
            is_best_dist, is_best_reward = metrics.check_best_model(
                episode + 1, 
                final_state.distance_from_origin,
                episode_reward
            )
            
            print(f"\n--- Episode {episode + 1} Complete ---")
            print(f"  Steps: {episode_steps}, Time: {episode_time:.1f}s")
            print(f"  Total Reward: {episode_reward:.1f}, Avg: {avg_reward:.2f}")
            print(f"  Final Distance: {final_state.distance_from_origin:.1f}m")
            print(f"  Max Speed: {final_state.speed:.1f} m/s ({max_speed_mph:.1f} mph)")
            print(f"  Crashes: {env.crash_count}, Damage: {final_state.damage:.2f}")
            print(f"  Buffer Size: {len(replay_buffer)}/{replay_buffer.capacity}")
            
            # Save best models
            if is_best_dist:
                print(f"  [TROPHY] NEW BEST DISTANCE! ({final_state.distance_from_origin:.1f}m)")
                metadata = {
                    'best_distance': metrics.best_distance,
                    'best_reward': metrics.best_reward,
                    'best_episode': metrics.best_episode,
                    'episode': episode + 1
                }
                agent.save(str(metrics.model_dir / 'highway_best.pth'), metadata=metadata)
                metrics.log_text(f"New best distance: {final_state.distance_from_origin:.1f}m in episode {episode+1}")
            
            if is_best_reward:
                print(f"  [STAR] NEW BEST REWARD! ({episode_reward:.1f})")
                agent.save(str(metrics.model_dir / 'highway_best_reward.pth'))
            
            total_reward_history.append(episode_reward)
            
            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                checkpoint_path = metrics.model_dir / f'highway_checkpoint_ep{episode+1}.pth'
                agent.save(str(checkpoint_path))
                agent.save(checkpoint_path)
                print(f"  [SAVE] Checkpoint saved: {checkpoint_path}")
        
        # Training complete summary
        summary = metrics.get_summary()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total Episodes: {episodes}")
        print(f"Average Reward: {np.mean(total_reward_history):.2f}")
        print(f"Best Distance: {summary['best_distance']:.1f}m (Episode {summary['best_episode']})")
        print(f"Best Reward: {summary['best_reward']:.1f}")
        print(f"Session ID: {summary['session_id']}")
        print(f"Metrics saved: {metrics.csv_path}")
        print("=" * 60)
        
        # Save final model
        agent.save(str(metrics.model_dir / 'highway_final.pth'))
        metrics.log_text(f"Training complete: {episodes} episodes, best distance {summary['best_distance']:.1f}m")
        
    except KeyboardInterrupt:
        print("\n\n" + "!" * 60)
        print("TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("!" * 60)
        print(f"Completed {episode + 1}/{episodes} episodes")
        
        summary = metrics.get_summary()
        print(f"Best distance so far: {summary['best_distance']:.1f}m")
        print(f"Saving interrupted model...")
        
        agent.save(str(metrics.model_dir / 'highway_interrupted.pth'))
        metrics.log_text(f"Training interrupted at episode {episode+1}")
        print("[OK] Model saved")
        
    except Exception as e:
        print("\n\n" + "!" * 60)
        print("ERROR DURING TRAINING")
        print("!" * 60)
        print(f"Error: {e}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        print("\\nAttempting to save model before exit...")
        try:
            agent.save('highway_model_error.pth')
            print("[OK] Emergency model save successful")
        except:
            print("[X] Could not save model")
            
    finally:
        print("\\nCleaning up...")
        env.close()

if __name__ == "__main__":
    train_highway_neural(episodes=100)

