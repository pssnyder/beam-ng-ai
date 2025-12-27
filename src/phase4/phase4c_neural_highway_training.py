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
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'training_steps': self.training_steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.training_steps = checkpoint['training_steps']
        print(f"Model loaded from {path}")

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
        self.episode_start_time = 0.0
        self.last_position = None
        self.last_damage = 0.0
        
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
            
            # Attach sensors
            self.vehicle.sensors.attach('electrics', Electrics())
            self.vehicle.sensors.attach('damage', Damage())
            self.vehicle.sensors.attach('gforces', GForces())
            
            # Use proven spawn positions (won't fall through map)
            if map_name == 'west_coast_usa':
                # Proven coordinates from Phase 2
                spawn_pos = (-717.121, 101, 118.675)
                spawn_rot = (0, 0, 0.3826834, 0.9238795)
            elif map_name == 'automation_test_track':
                # Test track spawn (needs validation - fallback to west_coast if issues)
                spawn_pos = (387.5, -2.5, 40.8)
                spawn_rot = (0, 0, 0.9238795, 0.3826834)
            else:
                # Generic fallback
                spawn_pos = (0, 0, 100)
                spawn_rot = (0, 0, 0, 1)
            
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
            
            print("Loading scenario...")
            self.bng.scenario.load(self.scenario)
            
            print("Starting scenario...")
            self.bng.scenario.start()
            
            print("Physics stabilization (3 seconds)...")
            time.sleep(3)  # Reduced from 5 to 3 seconds
            
            # Release parking brake and give initial throttle burst
            self.vehicle.control(parkingbrake=0, throttle=1.0, steering=0, brake=0)
            print("🚗 Parking brake released + initial throttle burst")
            time.sleep(0.5)
            
            # Initialize tracking
            self.vehicle.sensors.poll()
            pos = self.vehicle.state['pos']
            self.episode_origin = np.array(pos, dtype=np.float32)
            self.current_checkpoint = self.episode_origin.copy()
            self.last_position = self.episode_origin.copy()
            self.episode_start_time = time.time()
            
            print(f"\nScenario ready!")
            print(f"Starting position: {pos}")
            return True
            
        except Exception as e:
            print(f"ERROR: Scenario setup failed: {e}")
            print(f"\nTrying fallback map: west_coast_usa")
            if map_name != 'west_coast_usa':
                return self.setup_scenario('west_coast_usa')
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
        
        damage = max(damage_data.values()) if damage_data else 0.0
        
        # Distance calculations
        distance_from_origin = float(np.linalg.norm(pos - self.episode_origin))
        distance_from_checkpoint = float(np.linalg.norm(pos - self.current_checkpoint))
        distance_delta = float(np.linalg.norm(pos - self.last_position))
        
        self.last_position = pos.copy()
        
        # Update max distance
        if distance_from_origin > self.max_distance:
            self.max_distance = distance_from_origin
        
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
            gear=electrics.get('gear', 0.0),
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
        
        # Get next state
        next_state = self.get_state()
        
        # Calculate reward
        reward, info = self._calculate_reward(current_state, next_state)
        
        # Check for episode end
        done = info['crash_detected'] or info['stationary_timeout']
        
        # Handle crash - set new checkpoint
        if info['crash_detected']:
            self.current_checkpoint = next_state.position.copy()
            self.crash_count += 1
            print(f"  CRASH #{self.crash_count} at {next_state.distance_from_origin:.1f}m - checkpoint updated")
        
        # Handle stationary timeout - recover
        if info['stationary_timeout']:
            print("  Stationary timeout - recovering...")
            self.vehicle.recover()
            time.sleep(2)
            done = True
        
        self.total_steps += 1
        
        return next_state.to_vector(), reward, done, info
    
    def _calculate_reward(self, current_state, next_state):
        """Calculate reward with distance-based progressive system"""
        reward = 0.0
        info = {
            'crash_detected': False, 
            'stationary_timeout': False,
            'speed': next_state.speed,
            'distance_progress': 0.0
        }
        
        # Distance reward (primary)
        distance_progress = next_state.distance_from_checkpoint - current_state.distance_from_checkpoint
        info['distance_progress'] = distance_progress
        
        if distance_progress > 0:
            reward += distance_progress * 0.5  # 0.5 points per meter
        
        # Speed bonus (when making progress)
        if distance_progress > 0 and next_state.speed > 1.0:
            reward += next_state.speed * 0.1
        
        # Crash detection
        damage_increase = next_state.damage - self.last_damage
        if damage_increase > 0.05:
            reward -= 50.0
            info['crash_detected'] = True
            self.last_damage = next_state.damage
        
        # Stationary penalty - MORE AGGRESSIVE
        if next_state.speed < 0.5:
            self.stationary_timer += 0.3  # Accumulates faster
            reward -= 0.5  # Bigger penalty per step
            if self.stationary_timer > 3.0:  # Faster timeout (was 5.0)
                info['stationary_timeout'] = True
                reward -= 20.0
                print(f"  ! STATIONARY TIMEOUT after {self.stationary_timer:.1f}s")
        else:
            self.stationary_timer = 0.0
        
        info['reward'] = reward
        
        return reward, info
    
    def close(self):
        """Close connection (doesn't close BeamNG)"""
        if self.bng:
            print("\nDisconnecting from BeamNG (instance still running)")
            # Don't call bng.close() - leave BeamNG running

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_highway_neural(episodes=100, batch_size=64, replay_start_size=1000):
    """Main training loop"""
    
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
    state_dim = 27  # From TrainingState.to_vector()
    action_dim = 3  # throttle, steering, brake
    
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    replay_buffer = ExperienceReplay(capacity=100000)
    
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Replay buffer: {replay_buffer.capacity}")
    print(f"Batch size: {batch_size}")
    print("=" * 60 + "\n")
    
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
            
            # Episode loop
            while True:
                # Get action
                if len(replay_buffer) < replay_start_size:
                    # Random exploration - MAXIMUM THROTTLE BIAS
                    action = np.array([
                        np.random.uniform(0.7, 1.0),    # MUCH higher throttle (was 0.4-0.9)
                        np.random.uniform(-0.4, 0.4),   # Moderate steering
                        np.random.uniform(0, 0.02)      # Minimal brake (was 0-0.05)
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
                    print(f"  Explore Step {episode_steps}: Speed={info.get('speed', 0):.1f} m/s, "
                          f"Distance={info.get('distance_progress', 0):.1f}m, "
                          f"Reward={reward:.2f}, Action=[T:{action[0]:.2f} S:{action[1]:.2f} B:{action[2]:.2f}]")
                
                # Train agent
                if len(replay_buffer) >= replay_start_size and episode_steps % 2 == 0:
                    batch = replay_buffer.sample(batch_size)
                    losses = agent.update(batch)
                    
                    if episode_steps % 5 == 0:  # More frequent updates (was 10)
                        print(f"  Step {episode_steps}: Reward={reward:.2f}, "
                              f"Actor Loss={losses['actor_loss']:.4f}, "
                              f"Distance={info.get('distance_progress', 0):.1f}m")
                
                if done or episode_steps > 200:  # Max 200 steps per episode
                    break
            
            episode_time = time.time() - episode_start
            total_reward_history.append(episode_reward)
            
            print(f"\nEpisode {episode + 1} Complete:")
            print(f"  Time: {episode_time:.1f}s")
            print(f"  Steps: {episode_steps}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Crashes: {env.crash_count}")
            print(f"  Max Distance: {env.max_distance:.1f}m")
            print(f"  Buffer Size: {len(replay_buffer)}")
            
            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                agent.save(f'highway_model_ep{episode+1}.pth')
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total Episodes: {episodes}")
        print(f"Average Reward: {np.mean(total_reward_history):.2f}")
        print(f"Best Episode: {np.max(total_reward_history):.2f}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        agent.save('highway_model_interrupted.pth')
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    train_highway_neural(episodes=100)
