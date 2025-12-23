#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4C: Highway Distance Training with Crash Recovery
BeamNG AI Driver - Real-World Environment Training

Goal: Train AI to maximize distance traveled on Automation Test Track highway
Reward System: Distance from reference point, reset on crash, progressive rewards
Environment: automation_test_track map (highway section)

Key Features:
- Distance-based reward (greater distance = higher cumulative reward)
- Crash detection with distance reset to crash location
- Resume rewarding from crash point (not back to zero)
- Highway environment with visual features for learning
- Concrete/road following incentive
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage, GForces, Camera

# Optional PyTorch import for neural network components
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch not available - using random policy for demo")
    TORCH_AVAILABLE = False

@dataclass
class DistanceTrackingState:
    """State representation with distance tracking and crash recovery"""
    timestamp: float
    
    # Vehicle State
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    speed: float
    
    # Control State
    throttle: float
    steering: float
    brake: float
    
    # Distance Tracking (key feature)
    distance_from_origin: float          # Total distance from episode start
    distance_from_last_checkpoint: float # Distance from last crash/checkpoint
    checkpoint_position: Tuple[float, float, float]  # Current reference point
    
    # Crash Detection
    damage_level: float
    crash_detected: bool
    crash_count: int
    
    # Vehicle Telemetry
    electrics_data: Dict[str, float]
    gforces: Tuple[float, float, float]
    
    # Episode Stats
    episode_time: float
    total_reward: float

@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for analysis"""
    distance_reward: float = 0.0
    speed_bonus: float = 0.0
    crash_penalty: float = 0.0
    stationary_penalty: float = 0.0
    total: float = 0.0

class DistanceRewardCalculator:
    """
    Reward calculator based on distance traveled with crash recovery
    
    Philosophy:
    - Reward distance from current checkpoint (crash location or origin)
    - On crash: Reset checkpoint to crash location, lose accumulated rewards
    - Continue rewarding from new checkpoint
    - Higher distance = exponentially higher rewards
    """
    
    def __init__(self, distance_scale=0.5, speed_bonus_scale=0.1, crash_penalty=-50.0):
        self.distance_scale = distance_scale
        self.speed_bonus_scale = speed_bonus_scale
        self.crash_penalty = crash_penalty
        
        # Tracking
        self.last_damage = 0.0
        self.accumulated_reward = 0.0
        self.best_distance = 0.0
        
    def calculate_reward(self, state: DistanceTrackingState, 
                        next_state: DistanceTrackingState) -> RewardBreakdown:
        """Calculate reward with crash-aware distance tracking"""
        breakdown = RewardBreakdown()
        
        # 1. DISTANCE REWARD - Progressive reward for distance from checkpoint
        # Higher distance = better rewards (encourages exploration)
        distance_delta = next_state.distance_from_last_checkpoint - state.distance_from_last_checkpoint
        
        if distance_delta > 0:
            # Reward positive distance progress
            breakdown.distance_reward = distance_delta * self.distance_scale
            
            # Bonus for achieving new distance milestones
            if next_state.distance_from_last_checkpoint > self.best_distance:
                milestone_bonus = (next_state.distance_from_last_checkpoint - self.best_distance) * 0.1
                breakdown.distance_reward += milestone_bonus
                self.best_distance = next_state.distance_from_last_checkpoint
        
        # 2. SPEED BONUS - Encourage maintaining speed while traveling
        # Only reward speed if actually making forward progress
        if distance_delta > 0 and next_state.speed > 1.0:
            breakdown.speed_bonus = next_state.speed * self.speed_bonus_scale
        
        # 3. CRASH PENALTY - Heavy penalty for crashes (reset occurs separately)
        # Check for new damage (indicates crash)
        damage_increase = next_state.damage_level - state.damage_level
        if damage_increase > 0.05:  # Threshold for crash detection
            breakdown.crash_penalty = self.crash_penalty
            self.accumulated_reward = 0.0  # Reset accumulated rewards on crash
            print(f"💥 CRASH DETECTED! Distance from checkpoint: {state.distance_from_last_checkpoint:.1f}m")
            print(f"   All accumulated rewards lost. New checkpoint set.")
        
        # 4. STATIONARY PENALTY - Small penalty for not moving
        if next_state.speed < 0.5:
            breakdown.stationary_penalty = -0.1
        
        # Calculate total
        breakdown.total = (breakdown.distance_reward + 
                          breakdown.speed_bonus + 
                          breakdown.crash_penalty + 
                          breakdown.stationary_penalty)
        
        # Track accumulated reward
        if breakdown.crash_penalty == 0:
            self.accumulated_reward += breakdown.total
        
        return breakdown

class CrashRecoverySystem:
    """
    Manages crash detection and checkpoint updates
    
    On crash:
    1. Detect collision via damage increase
    2. Set new checkpoint to crash location
    3. Reset distance_from_checkpoint to 0
    4. Continue training from crash point
    """
    
    def __init__(self, vehicle: Vehicle, damage_threshold=0.05):
        self.vehicle = vehicle
        self.damage_threshold = damage_threshold
        self.last_damage = 0.0
        self.crash_count = 0
        self.checkpoint_position = None
        self.stationary_timer = 0.0
        
    def check_crash(self, current_state: DistanceTrackingState) -> bool:
        """Check if crash occurred based on damage increase"""
        damage_increase = current_state.damage_level - self.last_damage
        
        if damage_increase > self.damage_threshold:
            print(f"🚗 Crash #{self.crash_count + 1} - Setting new checkpoint")
            self.crash_count += 1
            self.last_damage = current_state.damage_level
            
            # Update checkpoint to current position
            self.checkpoint_position = current_state.position
            
            return True
        
        return False
    
    def check_stationary_timeout(self, speed: float, dt: float = 0.5) -> bool:
        """Check if vehicle has been stationary too long"""
        if speed < 0.3:
            self.stationary_timer += dt
            if self.stationary_timer > 5.0:  # 5 second timeout
                print("⏱️  Vehicle stationary timeout - recovery needed")
                return True
        else:
            self.stationary_timer = 0.0
        
        return False
    
    def perform_recovery(self):
        """Recover vehicle to safe position"""
        print("🔄 Performing auto-recovery...")
        try:
            self.vehicle.recover()
            time.sleep(2.0)
            
            # Apply gentle forward motion
            self.vehicle.control(throttle=0.3, steering=0.0, brake=0.0)
            time.sleep(0.5)
            
            # Update checkpoint after recovery
            self.vehicle.sensors.poll()
            pos = self.vehicle.state['pos']
            self.checkpoint_position = (pos[0], pos[1], pos[2])
            self.stationary_timer = 0.0
            
            print("✅ Recovery complete - new checkpoint set")
        except Exception as e:
            print(f"❌ Recovery failed: {e}")

class HighwayDistanceEnvironment:
    """
    BeamNG environment for highway distance training
    Uses Automation Test Track map with highway section
    """
    
    def __init__(self, use_camera=False):
        self.bng: Optional[BeamNGpy] = None
        self.vehicle: Optional[Vehicle] = None
        self.reward_calculator: Optional[DistanceRewardCalculator] = None
        self.recovery_system: Optional[CrashRecoverySystem] = None
        self.use_camera = use_camera
        
        # Episode tracking
        self.episode_origin = None
        self.current_checkpoint = None
        self.episode_start_time = 0.0
        self.episode_count = 0
        
        # Statistics
        self.total_distance = 0.0
        self.max_distance_achieved = 0.0
        self.total_crashes = 0
        
    def initialize_environment(self) -> bool:
        """Initialize BeamNG with Automation Test Track"""
        print("🚀 Phase 4C: Highway Distance Training")
        print("=" * 60)
        print("Map: automation_test_track (highway section)")
        print("Goal: Maximize distance traveled from origin/crash points")
        print("Reward: Progressive distance-based with crash recovery")
        print()
        
        try:
            set_up_simple_logging()
            
            bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
            self.bng = BeamNGpy('localhost', 25252, home=bng_home)
            
            print("🔄 Launching BeamNG for highway training...")
            self.bng.open(launch=True)
            print("✅ BeamNG connected!")
            
            return True
            
        except Exception as e:
            print(f"❌ Environment initialization failed: {e}")
            return False
    
    def create_highway_scenario(self) -> bool:
        """Create scenario on Automation Test Track highway"""
        if not self.bng:
            return False
            
        print("\n🗺️  Creating highway training scenario...")
        
        try:
            # Use automation_test_track for realistic highway environment
            scenario = Scenario('automation_test_track', 'highway_distance_training',
                              description='Phase 4C: Highway Distance Training with Visual Input')
            
            self.vehicle = Vehicle('ai_highway_driver', model='etk800', license='DIST4C')
            
            # Attach essential sensors
            electrics = Electrics()
            damage = Damage()
            gforces = GForces()
            
            self.vehicle.sensors.attach('electrics', electrics)
            self.vehicle.sensors.attach('damage', damage)
            self.vehicle.sensors.attach('gforces', gforces)
            
            # Optional: Attach camera for visual input
            if self.use_camera:
                camera = Camera('front_camera', self.bng,
                              pos=(0, 1.5, 0.5), dir=(0, 1, 0),
                              resolution=(640, 480), fov=90)
                self.vehicle.sensors.attach('camera', camera)
                print("📷 Front camera attached for visual input")
            
            # Highway spawn position on automation_test_track
            # Note: These coordinates need to be validated for the highway section
            # Using a safe starting position (may need adjustment)
            spawn_pos = (0, 0, 0.5)  # Will be updated once map is tested
            spawn_rot = (0, 0, 0, 1)
            
            scenario.add_vehicle(self.vehicle, pos=spawn_pos, rot_quat=spawn_rot)
            
            print("⚙️  Building scenario...")
            scenario.make(self.bng)
            
            print("⚡ Setting physics (60Hz)...")
            self.bng.settings.set_deterministic(60)
            
            print("📍 Loading scenario...")
            self.bng.scenario.load(scenario)
            
            print("▶️  Starting highway training...")
            self.bng.scenario.start()
            
            print("⏱️  Physics stabilization...")
            time.sleep(5)
            
            # Initialize systems
            self.vehicle.sensors.poll()
            pos = self.vehicle.state['pos']
            self.episode_origin = (pos[0], pos[1], pos[2])
            self.current_checkpoint = self.episode_origin
            self.episode_start_time = time.time()
            
            self.reward_calculator = DistanceRewardCalculator()
            self.recovery_system = CrashRecoverySystem(self.vehicle)
            self.recovery_system.checkpoint_position = self.episode_origin
            
            print(f"✅ Highway environment ready!")
            print(f"📍 Starting position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
            print(f"🎯 Origin checkpoint set - begin distance accumulation!")
            
            return True
            
        except Exception as e:
            print(f"❌ Scenario creation failed: {e}")
            print("⚠️  Note: automation_test_track map must be available")
            print("   Alternative: Use 'west_coast_usa' or 'italy' with highway sections")
            return False
    
    def get_state(self) -> DistanceTrackingState:
        """Get current environment state with distance tracking"""
        if not self.vehicle or not self.recovery_system:
            raise ValueError("Environment not initialized")
        
        # Poll sensors
        self.vehicle.sensors.poll()
        
        # Basic vehicle state
        pos = self.vehicle.state['pos']
        vel = self.vehicle.state['vel']
        speed = np.linalg.norm(vel)
        
        # Electrics and damage
        electrics_data = dict(self.vehicle.sensors['electrics'])
        damage_data = dict(self.vehicle.sensors['damage'])
        damage_level = max(damage_data.values()) if damage_data else 0.0
        
        gforces_data = dict(self.vehicle.sensors['gforces'])
        gforces = (gforces_data.get('gx', 0), 
                  gforces_data.get('gy', 0), 
                  gforces_data.get('gz', 0))
        
        # Calculate distances
        distance_from_origin = np.linalg.norm(
            np.array(pos) - np.array(self.episode_origin)
        )
        
        distance_from_checkpoint = np.linalg.norm(
            np.array(pos) - np.array(self.current_checkpoint)
        )
        
        # Update max distance
        if distance_from_origin > self.max_distance_achieved:
            self.max_distance_achieved = distance_from_origin
        
        # Check for crash
        crash_detected = self.recovery_system.check_crash(
            DistanceTrackingState(
                timestamp=time.time(),
                position=(pos[0], pos[1], pos[2]),
                velocity=(vel[0], vel[1], vel[2]),
                speed=speed,
                throttle=electrics_data.get('throttle', 0),
                steering=electrics_data.get('steering', 0),
                brake=electrics_data.get('brake', 0),
                distance_from_origin=distance_from_origin,
                distance_from_last_checkpoint=distance_from_checkpoint,
                checkpoint_position=self.current_checkpoint,
                damage_level=damage_level,
                crash_detected=False,
                crash_count=self.recovery_system.crash_count,
                electrics_data=electrics_data,
                gforces=gforces,
                episode_time=time.time() - self.episode_start_time,
                total_reward=self.reward_calculator.accumulated_reward if self.reward_calculator else 0
            )
        )
        
        # Update checkpoint if crash occurred
        if crash_detected:
            self.current_checkpoint = (pos[0], pos[1], pos[2])
            self.total_crashes += 1
            # Recalculate distance from new checkpoint
            distance_from_checkpoint = 0.0
        
        return DistanceTrackingState(
            timestamp=time.time(),
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
            speed=float(speed),
            throttle=electrics_data.get('throttle', 0.0),
            steering=electrics_data.get('steering', 0.0),
            brake=electrics_data.get('brake', 0.0),
            distance_from_origin=float(distance_from_origin),
            distance_from_last_checkpoint=float(distance_from_checkpoint),
            checkpoint_position=self.current_checkpoint,
            damage_level=damage_level,
            crash_detected=crash_detected,
            crash_count=self.recovery_system.crash_count,
            electrics_data=electrics_data,
            gforces=gforces,
            episode_time=time.time() - self.episode_start_time,
            total_reward=self.reward_calculator.accumulated_reward if self.reward_calculator else 0
        )
    
    def step(self, action: np.ndarray) -> Tuple[DistanceTrackingState, RewardBreakdown, bool, Dict]:
        """Execute action and return next state with reward breakdown"""
        if not self.vehicle or not self.recovery_system or not self.reward_calculator:
            raise ValueError("Environment not properly initialized")
        
        # Get current state
        current_state = self.get_state()
        
        # Apply action
        throttle = np.clip(action[0], 0.0, 1.0)
        steering = np.clip(action[1], -1.0, 1.0)
        brake = np.clip(action[2], 0.0, 1.0)
        
        self.vehicle.control(throttle=float(throttle),
                           steering=float(steering),
                           brake=float(brake))
        
        # Wait for physics update
        time.sleep(0.5)  # 2 Hz control for stability
        
        # Get next state
        next_state = self.get_state()
        
        # Calculate reward
        reward_breakdown = self.reward_calculator.calculate_reward(current_state, next_state)
        
        # Check for recovery needs
        needs_recovery = self.recovery_system.check_stationary_timeout(next_state.speed)
        if needs_recovery:
            self.recovery_system.perform_recovery()
        
        # Episode termination (for now, never terminate - continuous training)
        done = False
        
        # Additional info
        info = {
            'distance_from_origin': next_state.distance_from_origin,
            'distance_from_checkpoint': next_state.distance_from_last_checkpoint,
            'crash_count': next_state.crash_count,
            'max_distance': self.max_distance_achieved,
            'accumulated_reward': next_state.total_reward
        }
        
        return next_state, reward_breakdown, done, info
    
    def print_stats(self, state: DistanceTrackingState, reward: RewardBreakdown):
        """Print training statistics"""
        print(f"\n{'='*60}")
        print(f"📊 Episode Stats (Time: {state.episode_time:.1f}s)")
        print(f"{'='*60}")
        print(f"🎯 Distance from Origin:     {state.distance_from_origin:.1f}m")
        print(f"📍 Distance from Checkpoint: {state.distance_from_last_checkpoint:.1f}m")
        print(f"🏆 Max Distance Achieved:    {self.max_distance_achieved:.1f}m")
        print(f"💥 Crashes: {state.crash_count}")
        print(f"")
        print(f"🎁 Reward Breakdown:")
        print(f"   Distance:    +{reward.distance_reward:.2f}")
        print(f"   Speed Bonus: +{reward.speed_bonus:.2f}")
        print(f"   Crash:       {reward.crash_penalty:.2f}")
        print(f"   Stationary:  {reward.stationary_penalty:.2f}")
        print(f"   TOTAL:       {reward.total:.2f}")
        print(f"")
        print(f"💰 Accumulated Reward: {state.total_reward:.2f}")
        print(f"{'='*60}\n")
    
    def close(self):
        """Cleanup environment"""
        if self.bng:
            try:
                self.bng.close()
                print("🔒 Environment closed")
            except:
                pass

def main():
    """Test highway distance training environment"""
    env = HighwayDistanceEnvironment(use_camera=False)
    
    try:
        # Initialize
        if not env.initialize_environment():
            return False
        
        if not env.create_highway_scenario():
            return False
        
        print("\n🎮 Starting distance training test...")
        print("   Using random policy for demonstration")
        print("   Duration: 60 seconds\n")
        
        start_time = time.time()
        step_count = 0
        
        while time.time() - start_time < 60:
            # Random policy for testing
            action = np.array([
                np.random.uniform(0.2, 0.7),   # Throttle
                np.random.uniform(-0.3, 0.3),  # Steering
                np.random.uniform(0.0, 0.1)    # Brake
            ])
            
            # Execute step
            state, reward, done, info = env.step(action)
            step_count += 1
            
            # Print stats every 10 steps
            if step_count % 10 == 0:
                env.print_stats(state, reward)
        
        print("\n🎉 Test complete!")
        print(f"📊 Final Stats:")
        print(f"   Steps: {step_count}")
        print(f"   Max Distance: {env.max_distance_achieved:.1f}m")
        print(f"   Total Crashes: {env.total_crashes}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()
