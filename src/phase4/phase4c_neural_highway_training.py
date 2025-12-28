#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4C: Neural Network Highway Training with Persistent BeamNG
BeamNG AI Driver - SAC Neural Network Training on Highway

Features:
- Connects to already-running BeamNG instance
- SAC (Soft Actor-Critic) neural network training
- Distance-based progressive rewards
- Scenario restart for fast iteration (no full reload)
- Experience replay buffer
- Real-time training statistics

Usage:
1. Launch BeamNG.drive manually
2. Run this script - it will connect to the running instance
3. Training begins immediately on highway map
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
from beamngpy.sensors import Electrics, Damage, GForces, Lidar, AdvancedIMU

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
    
    def __init__(self, log_dir='training_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped session
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = self.log_dir / f'training_session_{self.session_id}.csv'
        self.log_path = self.log_dir / f'training_log_{self.session_id}.txt'
        
        # Initialize CSV
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Steps', 'Time_s', 'Total_Reward', 'Avg_Reward', 
                           'Final_Distance_m', 'Max_Speed_mph', 'Crashes', 'Final_Damage',
                           'Buffer_Size', 'Timestamp'])
        
        # Best model tracking
        self.best_distance = 0.0
        self.best_reward = float('-inf')
        self.best_episode = 0
        
        print(f"📊 Metrics logging to: {self.csv_path}")
        print(f"📝 Text logs to: {self.log_path}")
    
    def log_episode(self, episode, steps, time_s, total_reward, avg_reward,
                   final_distance, max_speed_mph, crashes, final_damage, buffer_size):
        """Log episode metrics to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, steps, f'{time_s:.1f}', f'{total_reward:.2f}', 
                           f'{avg_reward:.2f}', f'{final_distance:.1f}', f'{max_speed_mph:.1f}',
                           crashes, f'{final_damage:.2f}', buffer_size, 
                           datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    
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
    """Compact state representation for neural network"""
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
    
    # Spatial awareness - LiDAR (8 directional distances)
    lidar_front: float
    lidar_front_left: float
    lidar_front_right: float
    lidar_left: float
    lidar_right: float
    lidar_rear_left: float
    lidar_rear_right: float
    lidar_rear: float
    
    # IMU motion data (6)
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    
    # Episode info (3)
    episode_time: float
    crash_count: float
    stationary_time: float
    
    def to_vector(self) -> np.ndarray:
        """Convert to flat numpy vector for neural network"""
        return np.array([
            # Position (3)
            self.position[0], self.position[1], self.position[2],
            # Velocity (3)
            self.velocity[0], self.velocity[1], self.velocity[2],
            # Distance (3)
            self.distance_from_origin, self.distance_from_checkpoint, self.distance_delta,
            # Dynamics (15)
            self.speed, self.throttle, self.steering, self.brake,
            self.rpm, self.gear, self.wheelspeed,
            self.gx, self.gy, self.gz,
            self.damage, self.abs_active, self.esc_active, self.tcs_active, self.fuel,
            # LiDAR (8)
            self.lidar_front, self.lidar_front_left, self.lidar_front_right,
            self.lidar_left, self.lidar_right,
            self.lidar_rear_left, self.lidar_rear_right, self.lidar_rear,
            # IMU (6)
            self.accel_x, self.accel_y, self.accel_z,
            self.gyro_x, self.gyro_y, self.gyro_z,
            # Episode (3)
            self.episode_time, self.crash_count, self.stationary_time
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
                    print(f"✓ Connected on port {port}!")
                    return True
                except:
                    print("✗")
                    continue
            
            # No running instance found, launch new one
            if auto_launch:
                print("\nNo running instance found on any port.")
                print("Launching BeamNG.drive (this will take ~30 seconds)...")
                self.bng = BeamNGpy(self.host, 64256, home=self.bng_home)
                self.bng.open(launch=True)
                self.port = 64256
                print("✓ BeamNG launched successfully on port 64256!")
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
            
            self.vehicle = Vehicle('ai_neural', model='etk800', license='NEURAL')
            
            # Attach basic sensors
            self.vehicle.sensors.attach('electrics', Electrics())
            self.vehicle.sensors.attach('damage', Damage())
            self.vehicle.sensors.attach('gforces', GForces())
            
            # Attach spatial awareness sensors
            print("Attaching LiDAR sensor for road/obstacle detection...")
            lidar = Lidar('lidar_road', self.bng, self.vehicle,
                         requested_update_time=0.1,  # 10 Hz
                         pos=(0, 2.0, 1.0),  # Front of vehicle
                         dir=(0, 1, 0),  # Forward facing
                         vertical_resolution=8,  # 8 rays for directional sensing
                         vertical_angle=15,  # Limited vertical spread
                         horizontal_angle=120,  # 120 degree FOV
                         max_distance=100,  # 100m range
                         is_visualised=False)
            self.vehicle.sensors.attach('lidar_road', lidar)
            
            print("Attaching Advanced IMU for motion sensing...")
            imu = AdvancedIMU('imu_motion', self.bng, self.vehicle,
                             pos=(0, 0, 0.5),  # Center of vehicle
                             is_send_immediately=True)
            self.vehicle.sensors.attach('imu_motion', imu)
            
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
            print("✓ Scenario loaded")
            
            print("Starting scenario...")
            self.bng.scenario.start()
            print("✓ Scenario started")
            
            print("Physics stabilization (5 seconds)...")
            time.sleep(5)  # Increased wait for physics to settle
            print("✓ Physics stabilized")
            
            # Test sensors
            print("\nTesting spatial awareness sensors...")
            self.vehicle.sensors.poll()
            lidar_test = self.vehicle.sensors['lidar_road']
            imu_test = self.vehicle.sensors['imu_motion']
            if lidar_test and 'points' in lidar_test:
                print(f"  ✓ LiDAR: {len(lidar_test['points'])} points detected")
            else:
                print("  ⚠️  LiDAR: No data yet (will initialize during training)")
            
            if imu_test:
                print(f"  ✓ IMU: Motion data ready")
            else:
                print("  ⚠️  IMU: No data yet (will initialize during training)")
            
            # Release parking brake and give initial throttle burst
            self.vehicle.control(parkingbrake=0, throttle=1.0, steering=0, brake=0)
            print("\n🚗 Parking brake released + initial throttle burst")
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
            print(f"⚠️ Soft reset failed: {e}, falling back to scenario restart")
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
            
            # Initial throttle burst to overcome inertia
            self.vehicle.control(throttle=1.0, steering=0, brake=0, parkingbrake=0)
            time.sleep(0.5)
            print(f"Episode {self.episode_count} started (1s reset + throttle burst)")
            
        except Exception as e:
            print(f"WARNING: Reset failed: {e}")
            print("Attempting manual recovery...")
            self.vehicle.recover()
            time.sleep(1)
    
    def _process_lidar(self, lidar_data) -> np.ndarray:
        """Process LiDAR point cloud into 8 directional distance measurements"""
        try:
            if not lidar_data or 'points' not in lidar_data:
                return np.full(8, 100.0, dtype=np.float32)  # Max distance if no data
            
            points = lidar_data['points']
            if len(points) == 0:
                return np.full(8, 100.0, dtype=np.float32)
            
            # Convert to numpy array for efficient processing
            points_array = np.array(points, dtype=np.float32)
            
            # Calculate angles for each point (in vehicle reference frame)
            angles = np.arctan2(points_array[:, 1], points_array[:, 0])  # atan2(y, x)
            distances = np.linalg.norm(points_array[:, :2], axis=1)  # xy distance
            
            # Define 8 directional sectors (45 degrees each)
            # 0: front, 1: front-left, 2: front-right, 3: left, 4: right, 5: rear-left, 6: rear-right, 7: rear
            sectors = np.zeros(8, dtype=np.float32) + 100.0  # Initialize with max distance
            
            # Bin points into sectors
            for angle, dist in zip(angles, distances):
                angle_deg = np.degrees(angle) % 360
                
                if 337.5 <= angle_deg or angle_deg < 22.5:
                    sectors[0] = min(sectors[0], dist)  # Front
                elif 22.5 <= angle_deg < 67.5:
                    sectors[1] = min(sectors[1], dist)  # Front-left
                elif 67.5 <= angle_deg < 112.5:
                    sectors[3] = min(sectors[3], dist)  # Left
                elif 112.5 <= angle_deg < 157.5:
                    sectors[5] = min(sectors[5], dist)  # Rear-left
                elif 157.5 <= angle_deg < 202.5:
                    sectors[7] = min(sectors[7], dist)  # Rear
                elif 202.5 <= angle_deg < 247.5:
                    sectors[6] = min(sectors[6], dist)  # Rear-right
                elif 247.5 <= angle_deg < 292.5:
                    sectors[4] = min(sectors[4], dist)  # Right
                elif 292.5 <= angle_deg < 337.5:
                    sectors[2] = min(sectors[2], dist)  # Front-right
            
            # Normalize to 0-1 range (0 = obstacle at vehicle, 1 = clear 100m+)
            return np.clip(sectors / 100.0, 0.0, 1.0)
            
        except Exception as e:
            print(f"Warning: LiDAR processing failed: {e}, using default values")
            return np.full(8, 1.0, dtype=np.float32)  # Assume clear
    
    def _process_imu(self, imu_data) -> np.ndarray:
        """Process IMU data into acceleration and gyro vectors"""
        try:
            if not imu_data:
                return np.zeros(6, dtype=np.float32)
            
            accel = imu_data.get('accel', [0, 0, 0])
            gyro = imu_data.get('gyro', [0, 0, 0])
            
            # Normalize accelerations (typical range -20 to 20 m/s^2) to -1 to 1
            accel_norm = np.array(accel, dtype=np.float32) / 20.0
            accel_norm = np.clip(accel_norm, -1.0, 1.0)
            
            # Normalize gyro (typical range -10 to 10 rad/s) to -1 to 1
            gyro_norm = np.array(gyro, dtype=np.float32) / 10.0
            gyro_norm = np.clip(gyro_norm, -1.0, 1.0)
            
            return np.concatenate([accel_norm, gyro_norm])
            
        except Exception as e:
            print(f"Warning: IMU processing failed: {e}, using default values")
            return np.zeros(6, dtype=np.float32)
    
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
        
        # Process LiDAR data - extract 8 directional distances
        lidar_data = self.vehicle.sensors['lidar_road']
        lidar_distances = self._process_lidar(lidar_data)
        
        # Process IMU data - get acceleration and gyro
        imu_data = self.vehicle.sensors['imu_motion']
        imu_values = self._process_imu(imu_data)
        
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
            # LiDAR distances
            lidar_front=lidar_distances[0],
            lidar_front_left=lidar_distances[1],
            lidar_front_right=lidar_distances[2],
            lidar_left=lidar_distances[3],
            lidar_right=lidar_distances[4],
            lidar_rear_left=lidar_distances[5],
            lidar_rear_right=lidar_distances[6],
            lidar_rear=lidar_distances[7],
            # IMU data
            accel_x=imu_values[0],
            accel_y=imu_values[1],
            accel_z=imu_values[2],
            gyro_x=imu_values[3],
            gyro_y=imu_values[4],
            gyro_z=imu_values[5],
            # Episode info
            episode_time=time.time() - self.episode_start_time,
            crash_count=float(self.crash_count),
            stationary_time=self.stationary_timer
        )
    
    def step(self, action: np.ndarray):
        """Execute action and return (state, reward, done, info)"""
        # Get current state before action
        current_state = self.get_state()
        
        # Apply action (throttle, steering, brake)
        throttle = float(np.clip(action[0], 0, 1))
        steering = float(np.clip(action[1], -1, 1))
        brake = float(np.clip(action[2], 0, 1))
        
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
        
        # Calculate reward
        reward, info = self._calculate_reward(current_state, next_state)
        
        # Check for episode end
        done = info['crash_detected'] or info['stationary_timeout']
        
        # Handle crash - recover to road center at current position
        if info['crash_detected']:
            self.crash_count += 1
            crash_type = "FLIPPED" if info.get('flipped') else "DAMAGE"
            speed_info = f" (speed: {next_state.speed:.1f} m/s)"
            print(f"  💥 {crash_type} CRASH #{self.crash_count} at {next_state.distance_from_origin:.1f}m{speed_info} - recovering...")
            
            # Quick recovery: use current position but reset orientation to upright
            self._recover_to_road_center(next_state.position)
            
            # Update checkpoint to crash location
            self.current_checkpoint = next_state.position.copy()
            done = True
        
        # Handle stationary timeout - recover to road center
        if info['stationary_timeout']:
            print(f"  ⏱️ Stationary timeout after {self.stationary_timer:.1f}s - recovering...")
            self._recover_to_road_center(current_state.position)
            done = True
        
        self.total_steps += 1
        
        return next_state.to_vector(), reward, done, info
    
    def _recover_to_road_center(self, position):
        """Full vehicle recovery: always reset to highway spawn point"""
        try:
            print("    🔧 Executing full vehicle recovery...")
            print(f"    📍 Recovering to spawn point")
            
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
            
            print(f"    ✓ Vehicle recovered to spawn ({new_pos[0]:.1f}, {new_pos[1]:.1f}, {new_pos[2]:.1f})")
            print(f"    ✓ Damage reset, vehicle upright, ready to continue")
            
        except Exception as e:
            print(f"    ⚠️ Recovery failed, using fallback: {e}")
            self.vehicle.recover()  # Fallback to BeamNG's built-in recovery
            time.sleep(1)
    
    def _calculate_reward(self, current_state, next_state):
        """Calculate reward with distance-based progressive system"""
        reward = 0.0
        info = {
            'crash_detected': False, 
            'stationary_timeout': False,
            'flipped': False,
            'speeding': False,
            'stuck': False,
            'speed': next_state.speed,
            'distance_progress': 0.0
        }
        
        # Distance reward (primary)
        distance_progress = next_state.distance_from_checkpoint - current_state.distance_from_checkpoint
        info['distance_progress'] = distance_progress
        
        if distance_progress > 0:
            reward += distance_progress * 0.5  # 0.5 points per meter
        
        # Speed bonus (when making progress) - BUT penalize excessive speed
        speed_limit_ms = 20.0  # 45 mph = ~20 m/s
        if distance_progress > 0 and next_state.speed > 1.0:
            if next_state.speed <= speed_limit_ms:
                reward += next_state.speed * 0.1  # Bonus for good speed
            else:
                # Penalty for speeding - increases quadratically
                overspeed = next_state.speed - speed_limit_ms
                reward -= overspeed * 0.5  # Penalize excess speed
                info['speeding'] = True
        
        # Crash detection (damage-based) - VERY LENIENT, only major crashes
        damage_increase = next_state.damage - self.last_damage
        if damage_increase > 0.4:  # MUCH higher threshold (was 0.1) - only severe crashes
            reward -= 30.0
            info['crash_detected'] = True
            self.last_damage = next_state.damage
            print(f"    ⚠️ SEVERE DAMAGE: {damage_increase:.2f} increase (total: {next_state.damage:.2f})")
        elif damage_increase > 0.05:  # Minor damage (curb scrapes) - penalize but DON'T reset
            reward -= 5.0  # Small penalty, keep driving
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
            print(f"    ⚠️ VEHICLE FLIPPED: Gz={next_state.gz:.2f}, Gx={next_state.gx:.2f}, Gy={next_state.gy:.2f}")
        
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
            
            if self.stationary_timer > 12.0:  # VERY long timeout (was 8.0)
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
                print("✓ Cleanly disconnected from BeamNG")
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
    # State: 27 original + 8 LiDAR + 6 IMU = 41 dimensions
    state_dim = 41  # Updated from 27 with spatial awareness sensors
    action_dim = 3  # throttle, steering, brake
    
    print(f"Neural network input: {state_dim} dimensions (27 base + 8 LiDAR + 6 IMU)")
    
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    replay_buffer = ExperienceReplay(capacity=100000)
    
    # Initialize metrics tracking
    metrics = TrainingMetrics()
    
    # Check for existing best model and load it
    best_model_path = Path('models/highway_best.pth')
    best_model_path.parent.mkdir(exist_ok=True)
    
    if best_model_path.exists():
        print("\n🔄 Found existing best model - loading to continue training...")
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
            if episode > 0:
                env.reset_episode()
            
            state = env.get_state().to_vector()
            episode_reward = 0
            episode_steps = 0
            episode_start = time.time()
            
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            if len(replay_buffer) < replay_start_size:
                remaining = replay_start_size - len(replay_buffer)
                print(f"🎲 Random Exploration Mode ({len(replay_buffer)}/{replay_start_size} samples, {remaining} to go)")
            else:
                print(f"🧠 Neural Network Training Mode (buffer: {len(replay_buffer)}/{replay_buffer.capacity})")
            
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
                
                # Execute step
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                replay_buffer.push(state, action, reward, next_state, float(done))
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                # Print progress every step during exploration
                if len(replay_buffer) < replay_start_size:
                    speeding_flag = " ⚠️SPEEDING" if info.get('speeding', False) else ""
                    stuck_flag = " 🚫STUCK" if info.get('stuck', False) else ""
                    print(f"  Explore Step {episode_steps}: Speed={info.get('speed', 0):.1f} m/s{speeding_flag}{stuck_flag}, "
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
                        print(f"  🧠 Neural Training Step {episode_steps}: Reward={reward:.2f}, "
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
                print(f"  🏆 NEW BEST DISTANCE! ({final_state.distance_from_origin:.1f}m)")
                metadata = {
                    'best_distance': metrics.best_distance,
                    'best_reward': metrics.best_reward,
                    'best_episode': metrics.best_episode,
                    'episode': episode + 1
                }
                agent.save('models/highway_best.pth', metadata=metadata)
                metrics.log_text(f"New best distance: {final_state.distance_from_origin:.1f}m in episode {episode+1}")
            
            if is_best_reward:
                print(f"  🌟 NEW BEST REWARD! ({episode_reward:.1f})")
                agent.save('models/highway_best_reward.pth')
            
            total_reward_history.append(episode_reward)
            
            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                checkpoint_path = f'models/highway_checkpoint_ep{episode+1}.pth'
                agent.save(checkpoint_path)
                print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
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
        agent.save('models/highway_final.pth')
        metrics.log_text(f"Training complete: {episodes} episodes, best distance {summary['best_distance']:.1f}m")
        
    except KeyboardInterrupt:
        print("\n\n" + "!" * 60)
        print("TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("!" * 60)
        print(f"Completed {episode + 1}/{episodes} episodes")
        
        summary = metrics.get_summary()
        print(f"Best distance so far: {summary['best_distance']:.1f}m")
        print(f"Saving interrupted model...")
        
        agent.save('models/highway_interrupted.pth')
        metrics.log_text(f"Training interrupted at episode {episode+1}")
        print("✓ Model saved")
        
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
            print("✓ Emergency model save successful")
        except:
            print("✗ Could not save model")
            
    finally:
        print("\\nCleaning up...")
        env.close()

if __name__ == "__main__":
    train_highway_neural(episodes=100)
