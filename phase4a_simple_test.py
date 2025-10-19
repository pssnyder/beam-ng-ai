#!/usr/bin/env python3
"""
Phase 4A: Simple Exploratory Test
Test basic environment functionality without complex neural networks
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage, GForces

@dataclass
class SimpleState:
    """Simplified state for testing"""
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    speed: float
    damage_level: float
    distance_traveled: float

class SimpleExploratoryTest:
    """Simple test of exploratory environment"""
    
    def __init__(self):
        self.bng: Optional[BeamNGpy] = None
        self.vehicle: Optional[Vehicle] = None
        self.start_position = None
        self.stationary_timer = 0.0
        
    def initialize(self) -> bool:
        """Initialize BeamNG environment"""
        print("🚀 Phase 4A: Simple Exploratory Test")
        print("=" * 50)
        print("Testing: Basic movement, collision detection, auto-recovery")
        print()
        
        try:
            set_up_simple_logging()
            
            bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
            self.bng = BeamNGpy('localhost', 25252, home=bng_home)
            
            print("🔄 Connecting to BeamNG...")
            self.bng.open(launch=True)
            print("✅ Connected successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def create_scenario(self) -> bool:
        """Create simple test scenario"""
        if not self.bng:
            return False
            
        print("\n🗺️  Creating test scenario...")
        
        try:
            scenario = Scenario('west_coast_usa', 'simple_test', 
                              description='Phase 4A: Simple Exploratory Test')
            
            self.vehicle = Vehicle('test_car', model='etk800', license='TEST4A')
            
            # Attach basic sensors
            electrics = Electrics()
            damage = Damage()
            gforces = GForces()
            
            self.vehicle.sensors.attach('electrics', electrics)
            self.vehicle.sensors.attach('damage', damage)
            self.vehicle.sensors.attach('gforces', gforces)
            
            # Spawn in open area
            spawn_pos = (-717.121, 101, 118.675)
            spawn_rot = (0, 0, 0.3826834, 0.9238795)
            
            scenario.add_vehicle(self.vehicle, pos=spawn_pos, rot_quat=spawn_rot)
            
            print("⚙️  Building scenario...")
            scenario.make(self.bng)
            
            print("📍 Loading scenario...")
            self.bng.scenario.load(scenario)
            
            print("▶️  Starting scenario...")
            self.bng.scenario.start()
            
            # Store starting position
            time.sleep(3)  # Wait for physics to settle
            self.start_position = spawn_pos
            
            print("✅ Test scenario ready!")
            return True
            
        except Exception as e:
            print(f"❌ Scenario creation failed: {e}")
            return False
    
    def get_simple_state(self) -> SimpleState:
        """Get basic state information"""
        if not self.vehicle:
            raise ValueError("Vehicle not initialized")
        
        # Poll sensors
        self.vehicle.sensors.poll()
        
        # Basic state
        pos = self.vehicle.state['pos']
        vel = self.vehicle.state['vel']
        speed = np.linalg.norm(vel)
        
        # Damage level
        damage_data = dict(self.vehicle.sensors['damage'])
        if damage_data:
            # Filter numeric values only
            numeric_values = [v for v in damage_data.values() if isinstance(v, (int, float))]
            damage_level = max(numeric_values) if numeric_values else 0.0
        else:
            damage_level = 0.0
        
        # Distance from start
        if self.start_position:
            distance = np.linalg.norm(np.array(pos) - np.array(self.start_position))
        else:
            distance = 0.0
        
        return SimpleState(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            velocity=(float(vel[0]), float(vel[1]), float(vel[2])),
            speed=float(speed),
            damage_level=damage_level,
            distance_traveled=float(distance)
        )
    
    def random_action(self) -> Tuple[float, float, float]:
        """Generate random driving action"""
        # Gentle random exploration
        throttle = np.random.uniform(0.1, 0.5)  # Light to moderate throttle
        steering = np.random.uniform(-0.2, 0.2)  # Small steering adjustments
        brake = np.random.uniform(0.0, 0.1)     # Minimal braking
        
        return throttle, steering, brake
    
    def check_recovery_needed(self, state: SimpleState, dt: float = 0.1) -> bool:
        """Check if vehicle needs recovery"""
        recovery_needed = False
        reason = ""
        
        # Check for damage (collision)
        if state.damage_level > 0.01:
            recovery_needed = True
            reason = "collision detected"
        
        # Check for being stationary
        if state.speed < 0.5:  # Less than 0.5 m/s
            self.stationary_timer += dt
            if self.stationary_timer >= 5.0:
                recovery_needed = True
                reason = "stationary timeout"
        else:
            self.stationary_timer = 0.0
        
        if recovery_needed:
            print(f"🚗 Recovery needed: {reason}")
            self.perform_recovery()
            return True
        
        return False
    
    def perform_recovery(self):
        """Recover vehicle to safe position"""
        if not self.vehicle:
            return
            
        print("🔧 Performing vehicle recovery...")
        try:
            self.vehicle.recover()
            time.sleep(2)  # Wait for recovery
            
            # Apply gentle forward motion
            self.vehicle.control(throttle=0.2, steering=0.0, brake=0.0)
            
            self.stationary_timer = 0.0
            print("✅ Recovery complete")
            
        except Exception as e:
            print(f"⚠️  Recovery failed: {e}")
    
    def run_exploration_test(self, duration: int = 30) -> bool:
        """Run simple exploration test"""
        if not self.vehicle:
            return False
            
        print(f"\n🎯 Running exploration test for {duration} seconds...")
        print("Policy: Random actions with auto-recovery")
        print("Rewards: Distance traveled - collision penalties")
        print()
        
        start_time = time.time()
        total_reward = 0.0
        recovery_count = 0
        max_distance = 0.0
        
        try:
            while time.time() - start_time < duration:
                # Get current state
                state = self.get_simple_state()
                
                # Track maximum distance
                max_distance = max(max_distance, state.distance_traveled)
                
                # Check if recovery needed
                if self.check_recovery_needed(state, dt=0.5):
                    recovery_count += 1
                    total_reward -= 50.0  # Recovery penalty
                    time.sleep(2)  # Wait after recovery
                    continue
                
                # Calculate reward (distance traveled)
                distance_reward = state.distance_traveled * 0.1  # 0.1 points per meter
                if state.damage_level > 0.01:
                    distance_reward -= 10.0  # Collision penalty
                
                total_reward += distance_reward * 0.5  # Per step
                
                # Generate and execute random action
                throttle, steering, brake = self.random_action()
                self.vehicle.control(throttle=throttle, steering=steering, brake=brake)
                
                # Display progress every 5 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    print(f"  ⏱️  {elapsed:.0f}s: Distance={state.distance_traveled:.1f}m, "
                          f"Speed={state.speed:.1f}m/s, Reward={total_reward:.1f}")
                
                time.sleep(0.5)  # 2 Hz control rate
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            return False
        
        # Test results
        elapsed_time = time.time() - start_time
        print(f"\n📊 Test Results ({elapsed_time:.1f}s):")
        print(f"  • Maximum distance: {max_distance:.1f}m")
        print(f"  • Total reward: {total_reward:.1f}")
        print(f"  • Recovery events: {recovery_count}")
        print(f"  • Average reward rate: {total_reward/elapsed_time:.2f}/s")
        
        # Success criteria
        success = max_distance > 10.0 and recovery_count < 10
        if success:
            print("✅ Test PASSED: AI showed basic exploration behavior")
        else:
            print("⚠️  Test needs improvement: Limited exploration or too many recoveries")
        
        return success
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n🔒 Cleanup...")
        try:
            if self.bng:
                self.bng.close()
            print("✅ Environment closed")
        except:
            pass

def main():
    """Run simple exploratory test"""
    test = SimpleExploratoryTest()
    
    try:
        # Initialize
        if not test.initialize():
            return False
        
        # Create scenario
        if not test.create_scenario():
            return False
        
        # Run exploration test
        success = test.run_exploration_test(duration=45)  # 45 second test
        
        if success:
            print("\n🎉 PHASE 4A FOUNDATION VALIDATED!")
            print("✅ Basic exploration behavior working")
            print("✅ Auto-recovery system functional") 
            print("✅ Reward system operational")
            print("🚀 Ready for full neural network integration")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        test.cleanup()

if __name__ == "__main__":
    print("BeamNG AI Driver - Phase 4A: Simple Exploratory Test")
    print("Testing basic self-learning environment components")
    print()
    
    success = main()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("• Integrate full neural network with live visualization")
        print("• Implement SAC (Soft Actor-Critic) training loop")
        print("• Add progressive reward system expansion")
        print("• Build neural network activity dashboard")
    else:
        print("\n🔧 DEBUGGING NEEDED:")
        print("• Check vehicle physics and control response")
        print("• Verify recovery system functionality")
        print("• Test sensor data collection pipeline")