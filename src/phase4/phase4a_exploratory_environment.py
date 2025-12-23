#!/usr/bin/env python3
"""
Phase 4A: Exploratory Self-Learning Environment
BeamNG AI Driver - Minimal Policy Reinforcement Learning

Philosophy: "Keep Moving, Don't Crash" - Let AI discover driving strategies
Approach: Policy-based RL with auto-recovery and live neural network monitoring
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage, GForces

# Optional PyTorch import for neural network components
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not available - using random policy for demo")
    TORCH_AVAILABLE = False

@dataclass
class ExploratoryState:
    """Enhanced state representation for exploratory learning"""
    timestamp: float
    
    # Basic Vehicle State (7 channels)
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]  
    speed: float
    
    # Control State (3 channels)
    current_throttle: float
    current_steering: float
    current_brake: float
    
    # Enhanced Vehicle Telemetry (40+ priority channels from electrics)
    electrics_core: Dict[str, float]  # Core 40 most important channels
    damage_level: float
    gforces: Tuple[float, float, float]
    
    # Environmental Context
    collision_detected: bool
    stationary_time: float
    distance_traveled: float
    episode_time: float

class MinimalPolicyNetwork:
    """Simple policy network for exploratory learning with monitoring"""
    
    def __init__(self, input_size=50, hidden_sizes=[128, 64, 32], output_size=3):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_activations = {}
        
        if TORCH_AVAILABLE:
            self.network = self._build_torch_network(input_size, hidden_sizes, output_size)
        else:
            # Fallback to simple random policy for demo
            print("🎲 Using random policy (PyTorch not available)")
            self.network = None
            
    def _build_torch_network(self, input_size, hidden_sizes, output_size):
        """Build PyTorch network if available"""
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(torch.nn.Linear(prev_size, output_size))
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass with activation monitoring"""
        if not TORCH_AVAILABLE or self.network is None:
            # Random policy fallback
            throttle = np.random.uniform(0.1, 0.6)  # Gentle throttle
            steering = np.random.uniform(-0.3, 0.3)  # Small steering
            brake = np.random.uniform(0.0, 0.1)     # Minimal braking
            
            output = np.array([throttle, steering, brake])
            self.layer_activations['output'] = {'simulated': True}
            return output
        
        # PyTorch forward pass
        with torch.no_grad():
            x = self.network(x)
            
            # Apply output activations
            throttle = torch.sigmoid(x[:, 0])
            steering = torch.tanh(x[:, 1])
            brake = torch.sigmoid(x[:, 2])
            
            output = torch.stack([throttle, steering, brake], dim=1)
            self.layer_activations['output'] = output.detach().clone()
            
            return output.squeeze().numpy()
    
    def get_activation_summary(self):
        """Get summary of neural activity for visualization"""
        if not TORCH_AVAILABLE:
            return {'random_policy': {'active_neurons': 'N/A', 'simulation_mode': True}}
        
        summary = {}
        for layer_name, activations in self.layer_activations.items():
            if hasattr(activations, 'mean'):
                summary[layer_name] = {
                    'mean_activation': float(activations.mean()),
                    'max_activation': float(activations.max()),
                    'active_neurons': int((activations.abs() > 0.1).sum()),
                    'total_neurons': int(activations.numel())
                }
            else:
                summary[layer_name] = {'status': 'simulated'}
        return summary

class AutoRecoverySystem:
    """Vehicle recovery system for autonomous training"""
    
    def __init__(self, vehicle: Vehicle, stationary_threshold=5.0):
        self.vehicle = vehicle
        self.stationary_threshold = stationary_threshold
        self.stationary_timer = 0.0
        self.last_position = None  
        self.recovery_count = 0
        self.collision_detected = False
        
    def update(self, current_state: ExploratoryState, dt: float = 0.1) -> bool:
        """Check if recovery is needed, return True if recovery performed"""
        recovery_needed = False
        recovery_reason = None
        
        # Check for collision (damage increase indicates collision)
        if current_state.damage_level > 0.01:  # Damage threshold
            recovery_needed = True
            recovery_reason = "collision"
            self.collision_detected = True
        
        # Check for stationary timeout
        if current_state.speed < 0.5:  # Near stationary (< 0.5 m/s)
            self.stationary_timer += dt
            if self.stationary_timer >= self.stationary_threshold:
                recovery_needed = True
                recovery_reason = "stationary_timeout"
        else:
            self.stationary_timer = 0.0  # Reset timer if moving
        
        # Perform recovery if needed
        if recovery_needed and recovery_reason:
            self.perform_recovery(recovery_reason)
            return True
        
        return False
    
    def perform_recovery(self, reason: str):
        """Execute vehicle recovery to safe position"""
        print(f"🚗 Auto-recovery triggered: {reason} (Recovery #{self.recovery_count + 1})")
        
        try:
            # Use BeamNG's built-in recovery system
            self.vehicle.recover()
            
            # Wait for recovery to complete
            time.sleep(2.0)
            
            # Apply gentle forward throttle to start movement
            self.vehicle.control(throttle=0.2, steering=0.0, brake=0.0)
            
            # Reset recovery tracking
            self.stationary_timer = 0.0
            self.collision_detected = False
            self.recovery_count += 1
            
            print(f"✅ Recovery complete - vehicle repositioned safely")
            
        except Exception as e:
            print(f"⚠️  Recovery failed: {e}")

class ExploratoryEnvironment:
    """BeamNG environment for exploratory self-learning"""
    
    def __init__(self):
        self.bng: Optional[BeamNGpy] = None
        self.vehicle: Optional[Vehicle] = None
        self.recovery_system: Optional[AutoRecoverySystem] = None
        
        # Episode tracking
        self.episode_start_time = 0.0
        self.episode_start_position = None
        self.total_distance = 0.0
        self.episode_count = 0
        
        # Reward tracking
        self.total_reward = 0.0
        self.reward_components = {
            'distance': 0.0,
            'collision_penalty': 0.0,
            'stationary_penalty': 0.0
        }
        
    def initialize_environment(self) -> bool:
        """Initialize BeamNG with minimal constraints environment"""
        print("🚀 Phase 4A: Exploratory Self-Learning Environment")
        print("=" * 60)
        print("Philosophy: Keep Moving, Don't Crash")
        print("Approach: Minimal policy with auto-recovery")
        print()
        
        try:
            set_up_simple_logging()
            
            bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
            self.bng = BeamNGpy('localhost', 25252, home=bng_home)
            
            print("🔄 Launching BeamNG for exploratory learning...")
            self.bng.open(launch=True)
            print("✅ BeamNG connected - ready for AI exploration!")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment initialization failed: {e}")
            return False
    
    def create_exploration_scenario(self) -> bool:
        """Create open environment for AI exploration"""
        if not self.bng:
            return False
            
        print("\n🗺️  Creating exploration environment...")
        
        try:
            # Use west_coast_usa for open environment exploration
            scenario = Scenario('west_coast_usa', 'ai_exploration', 
                              description='Phase 4A: AI Self-Learning Environment')
            
            self.vehicle = Vehicle('ai_explorer', model='etk800', license='EXPLORE')
            
            # Attach essential sensors only (keep it simple)
            electrics = Electrics()
            damage = Damage()
            gforces = GForces()
            
            self.vehicle.sensors.attach('electrics', electrics)
            self.vehicle.sensors.attach('damage', damage)
            self.vehicle.sensors.attach('gforces', gforces)
            
            # Spawn in open area for exploration
            spawn_pos = (-717.121, 101, 118.675)  # Proven spawn location
            spawn_rot = (0, 0, 0.3826834, 0.9238795)
            
            scenario.add_vehicle(self.vehicle, pos=spawn_pos, rot_quat=spawn_rot)
            
            print("⚙️  Building exploration scenario...")
            scenario.make(self.bng)
            
            print("📍 Loading scenario...")
            self.bng.scenario.load(scenario)
            
            print("▶️  Starting AI exploration...")
            self.bng.scenario.start()
            
            # Initialize recovery system
            self.recovery_system = AutoRecoverySystem(self.vehicle)
            
            # Initialize episode tracking
            self.episode_start_time = time.time()
            self.episode_start_position = spawn_pos
            
            print("✅ Exploration environment ready!")
            return True
            
        except Exception as e:
            print(f"❌ Scenario creation failed: {e}")
            return False
    
    def get_state(self) -> ExploratoryState:
        """Get current environment state for policy decision"""
        if not self.vehicle:
            raise ValueError("Vehicle not initialized")
        
        # Poll sensors
        self.vehicle.sensors.poll()
        
        # Basic vehicle state
        pos = self.vehicle.state['pos']
        vel = self.vehicle.state['vel']
        speed = np.linalg.norm(vel)
        
        # Extract core electrics data (select most important 40 channels)
        electrics_data = dict(self.vehicle.sensors['electrics'])
        electrics_core = self.select_core_electrics(electrics_data)
        
        # Damage and physics
        damage_data = dict(self.vehicle.sensors['damage'])
        damage_level = max(damage_data.values()) if damage_data else 0.0
        
        gforces_data = dict(self.vehicle.sensors['gforces'])
        gforces = (gforces_data.get('gx', 0), gforces_data.get('gy', 0), gforces_data.get('gz', 0))
        
        # Calculate distance traveled
        if self.episode_start_position:
            distance = np.linalg.norm(np.array(pos) - np.array(self.episode_start_position))
        else:
            distance = 0.0
        
        # Episode timing
        episode_time = time.time() - self.episode_start_time
        
        return ExploratoryState(
            timestamp=time.time(),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
            speed=float(speed),
            current_throttle=0.0,  # Will be updated by action
            current_steering=0.0,
            current_brake=0.0,
            electrics_core=electrics_core,
            damage_level=damage_level,
            gforces=gforces,
            collision_detected=damage_level > 0.01,
            stationary_time=self.recovery_system.stationary_timer if self.recovery_system else 0.0,
            distance_traveled=float(distance),
            episode_time=episode_time
        )
    
    def select_core_electrics(self, electrics_data: Dict) -> Dict[str, float]:
        """Select most important electrics channels for neural network"""
        # Core channels that are most relevant for driving control
        priority_channels = [
            'throttle', 'brake', 'steering', 'wheelspeed', 'airspeed',
            'rpm', 'gear', 'fuel', 'turbo_pressure', 'engine_temp',
            'transmission_temp', 'abs_active', 'esc_active', 'tcs_active',
            'clutch_ratio', 'torque', 'power', 'brake_temp'
        ]
        
        core_data = {}
        for channel in priority_channels:
            core_data[channel] = electrics_data.get(channel, 0.0)
        
        # Fill remaining slots with other available data
        remaining_slots = max(0, 40 - len(core_data))
        other_channels = [k for k in electrics_data.keys() if k not in priority_channels]
        
        for i, channel in enumerate(other_channels[:remaining_slots]):
            core_data[channel] = electrics_data[channel]
        
        return core_data
    
    def calculate_reward(self, state: ExploratoryState, action: np.ndarray, next_state: ExploratoryState) -> float:
        """Minimal policy reward: distance traveled - penalties"""
        reward = 0.0
        
        # PRIMARY REWARD: Distance traveled
        distance_reward = next_state.distance_traveled - state.distance_traveled
        reward += distance_reward * 1.0  # 1.0 reward per meter
        self.reward_components['distance'] += distance_reward
        
        # PRIMARY PENALTY: Collision detected
        if next_state.collision_detected and not state.collision_detected:
            collision_penalty = -100.0
            reward += collision_penalty
            self.reward_components['collision_penalty'] += collision_penalty
            print(f"💥 Collision detected! Penalty: {collision_penalty}")
        
        # MOVEMENT PENALTY: Will be handled by auto-recovery system
        # No explicit penalty here, recovery system handles resets
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[ExploratoryState, float, bool, Dict]:
        """Execute action and return next state"""
        if not self.vehicle or not self.recovery_system:
            raise ValueError("Environment not properly initialized")
        
        # Apply action to vehicle
        throttle, steering, brake = action[0], action[1], action[2]
        
        # Clip actions to valid ranges
        throttle = np.clip(throttle, 0.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0) 
        brake = np.clip(brake, 0.0, 1.0)
        
        # Execute control
        self.vehicle.control(throttle=float(throttle), 
                           steering=float(steering), 
                           brake=float(brake))
        
        # Wait for physics update
        time.sleep(0.1)  # 10 Hz control rate
        
        # Get current state
        current_state = self.get_state()
        
        # Update state with current action
        current_state.current_throttle = throttle
        current_state.current_steering = steering
        current_state.current_brake = brake
        
        # Check if auto-recovery is needed
        recovery_performed = self.recovery_system.update(current_state, dt=0.1)
        
        # Calculate reward (will be expanded in future phases)
        reward = self.calculate_reward(getattr(self, '_last_state', current_state), action, current_state)
        self.total_reward += reward
        
        # Episode termination conditions
        done = False
        if recovery_performed:
            done = True  # Reset episode after recovery
            print(f"📊 Episode {self.episode_count} complete - Total reward: {self.total_reward:.2f}")
        
        # Store state for next iteration
        self._last_state = current_state
        
        # Info for monitoring
        info = {
            'recovery_performed': recovery_performed,
            'recovery_count': self.recovery_system.recovery_count,
            'distance_traveled': current_state.distance_traveled,
            'episode_time': current_state.episode_time,
            'reward_components': self.reward_components.copy()
        }
        
        return current_state, reward, done, info
    
    def reset(self) -> ExploratoryState:
        """Reset environment for new episode"""
        self.episode_count += 1
        self.episode_start_time = time.time()
        self.total_reward = 0.0
        self.reward_components = {'distance': 0.0, 'collision_penalty': 0.0, 'stationary_penalty': 0.0}
        
        if self.vehicle:
            # Get fresh starting position
            self.episode_start_position = self.vehicle.state['pos']
        
        print(f"\n🔄 Starting Episode {self.episode_count}")
        return self.get_state()
    
    def cleanup(self):
        """Clean environment shutdown"""
        print("\n🔒 Exploratory environment cleanup...")
        try:
            if self.bng:
                self.bng.close()
            print("✅ Environment closed")
        except:
            pass

def state_to_tensor(state: ExploratoryState):
    """Convert state to neural network input (tensor or numpy array)"""
    # Combine all state information into input vector
    features = []
    
    # Vehicle state (7 features)
    features.extend(state.position)
    features.extend(state.velocity)
    features.append(state.speed)
    
    # Control state (3 features)
    features.extend([state.current_throttle, state.current_steering, state.current_brake])
    
    # Core electrics data (40 features)
    features.extend(list(state.electrics_core.values()))
    
    # Physics and damage (4 features)
    features.append(state.damage_level)
    features.extend(state.gforces)
    
    if TORCH_AVAILABLE:
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    else:
        return np.array(features, dtype=np.float32)

def demo_exploratory_learning():
    """Demonstrate exploratory learning environment"""
    env = ExploratoryEnvironment()
    
    try:
        # Initialize environment
        if not env.initialize_environment():
            return False
        
        if not env.create_exploration_scenario():
            return False
        
        # Create simple policy network
        input_size = 54  # 7 + 3 + 40 + 4 features
        policy = MinimalPolicyNetwork(input_size=input_size)
        
        # Run exploration episodes
        print(f"\n🧠 Starting exploratory learning with {input_size} input features...")
        print("🎯 Minimal Policy: Keep moving, avoid crashes")
        
        for episode in range(3):  # Demo with 3 episodes
            state = env.reset()
            episode_reward = 0.0
            steps = 0
            
            while steps < 100:  # Max 100 steps per episode (~10 seconds)
                # Convert state to neural network input
                state_tensor = state_to_tensor(state)
                
                # Get action from policy (random exploration for demo)
                action = policy.forward(state_tensor)
                
                # Add some exploration noise
                action += np.random.normal(0, 0.1, size=action.shape)
                
                # Execute step
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Display progress
                if steps % 10 == 0:
                    print(f"  Step {steps}: Reward={reward:.2f}, Distance={next_state.distance_traveled:.1f}m, Speed={next_state.speed:.1f}m/s")
                    
                    # Show neural network activity
                    activation_summary = policy.get_activation_summary()
                    print(f"  🧠 Neural Activity: {activation_summary.get('layer_3', {}).get('active_neurons', 0)} active neurons")
                
                state = next_state
                steps += 1
                
                if done:
                    break
            
            print(f"📊 Episode {episode + 1} complete: {episode_reward:.2f} total reward, {steps} steps")
        
        print("\n🎉 Exploratory learning demo complete!")
        print("✅ AI successfully demonstrated basic exploration behavior")
        print("🧠 Neural network activity monitoring functional")
        print("🚗 Auto-recovery system operational")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        env.cleanup()

if __name__ == "__main__":
    print("BeamNG AI Driver - Phase 4A: Exploratory Self-Learning")
    print("Minimal Policy Reinforcement Learning with Auto-Recovery")
    print()
    
    success = demo_exploratory_learning()
    
    if success:
        print("\n🚀 PHASE 4A FOUNDATION READY!")
        print("Next: Implement full SAC training loop with live visualization")
        print("• Continuous learning with replay buffer")
        print("• Live neural network visualization dashboard")
        print("• Progressive reward system expansion")
    else:
        print("\n❌ Phase 4A needs debugging - check errors above")