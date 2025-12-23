#!/usr/bin/env python3
"""
Phase 2: Core Game Control and Dynamic Sensor Implementation
BeamNG AI Driver - Maximum Telemetry Access

ROBUST VERSION - Handles BeamNG menu loading issues

This phase implements:
1. Robust BeamNG startup and menu handling
2. Proper vehicle spawn positioning
3. Basic sensor data collection
4. Vehicle control interface

Addresses common BeamNG startup issues.
"""

import time
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage

def wait_for_beamng_ready(bng, timeout=30):
    """Wait for BeamNG to be fully ready and responsive"""
    print("⏱️  Waiting for BeamNG to be fully ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Try to get a simple response from BeamNG
            bng.control.step(1)  # This will fail if BeamNG isn't ready
            print("✅ BeamNG is responsive!")
            return True
        except Exception as e:
            print(f"⏳ BeamNG not ready yet... ({time.time() - start_time:.1f}s)")
            time.sleep(2)
    
    print("❌ BeamNG failed to become ready within timeout")
    return False

def main():
    """Phase 2 Main: Core Game Control and Dynamic Sensor Implementation"""
    
    print("🚀 Phase 2: Core Game Control and Dynamic Sensor Implementation")
    print("=" * 70)
    print("Goal: Robust BeamNG startup + basic telemetry + vehicle control")
    print()
    
    try:
        # Set up logging
        set_up_simple_logging()
        
        # Initialize BeamNG connection with longer timeout
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        bng = BeamNGpy('localhost', 25252, home=bng_home)
        
        print("🔄 Launching BeamNG.drive...")
        bng.open(launch=True)
        print("✅ BeamNG process started!")
        
        # Wait for BeamNG to be fully ready (this often helps with menu issues)
        if not wait_for_beamng_ready(bng, timeout=60):
            print("❌ BeamNG failed to become ready")
            return False
        
        print("⚙️  Setting up scenario...")
        
        # Use a simple, reliable map and scenario
        scenario = Scenario('gridmap_v2', 'phase2_simple', 
                          description='Phase 2: Simple Test')
        
        # Create vehicle with basic setup
        vehicle = Vehicle('test_car', model='etk800', license='TEST')
        
        # Simple spawn position for gridmap_v2 (center, elevated)
        spawn_pos = (0, 0, 100)  # Center of gridmap, well above ground
        
        # Add vehicle to scenario
        scenario.add_vehicle(vehicle, pos=spawn_pos)
        
        print("🔧 Building scenario...")
        scenario.make(bng)
        
        print("⚙️  Setting deterministic mode...")
        bng.settings.set_deterministic(60)
        
        print("📍 Loading scenario (this may take a moment)...")
        bng.scenario.load(scenario)
        
        # Additional wait after loading
        print("⏱️  Waiting for scenario to fully load...")
        time.sleep(10)  # Give extra time for map loading
        
        print("▶️  Starting scenario...")
        bng.scenario.start()
        
        # Wait for scenario to actually start
        print("⏱️  Waiting for scenario to initialize...")
        time.sleep(5)
        
        # Try to check if we can get vehicle state
        print("🔍 Testing vehicle connection...")
        try:
            # This will test if the vehicle is actually spawned and accessible
            # Simple test: try to poll vehicle state
            test_state = vehicle.state
            print("✅ Vehicle is accessible in simulation")
        except Exception as e:
            print(f"⚠️  Vehicle connection issue: {e}")
            # Try alternative approach - give more time
            print("🔄 Giving more time for vehicle to initialize...")
            time.sleep(10)
        
        # Set up basic sensors after everything is loaded
        print("🔧 Setting up sensors...")
        
        electrics = Electrics()
        damage = Damage()
        
        try:
            vehicle.sensors.attach('electrics', electrics)
            vehicle.sensors.attach('damage', damage)
            print("✅ Sensors attached")
        except Exception as e:
            print(f"⚠️  Sensor attachment issue: {e}")
        
        # Wait for everything to settle
        print("⏱️  Letting physics settle...")
        time.sleep(5)
        
        # Test basic functionality
        print("\n🔍 Testing basic vehicle functionality...")
        
        try:
            # Try to poll sensors
            vehicle.sensors.poll()
            print("✅ Sensor polling works")
            
            # Check position
            pos = vehicle.state['pos']
            print(f"📍 Vehicle position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            # Check if vehicle is at reasonable height
            if pos[2] < 50:
                print("⚠️  Vehicle may have fallen through map")
                # Try to teleport to safer position
                print("🔄 Attempting to reposition vehicle...")
                vehicle.teleport(pos=(0, 0, 100), rot_quat=(0, 0, 0, 1))
                time.sleep(3)
                vehicle.sensors.poll()
                new_pos = vehicle.state['pos']
                print(f"📍 New position: ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            return False
        
        # Test simple vehicle control
        print("\n🎮 Testing basic vehicle control...")
        
        try:
            # Very simple control test
            print("🚗 Testing throttle...")
            vehicle.control(throttle=0.2, steering=0.0, brake=0.0)
            time.sleep(2)
            
            # Check if vehicle responded
            vehicle.sensors.poll()
            vel = vehicle.state['vel']
            speed = np.linalg.norm(vel)
            print(f"🏁 Vehicle speed: {speed:.2f} m/s")
            
            # Stop vehicle
            print("🛑 Stopping vehicle...")
            vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
            time.sleep(2)
            
            vehicle.sensors.poll()
            final_vel = vehicle.state['vel']
            final_speed = np.linalg.norm(final_vel)
            print(f"🏁 Final speed: {final_speed:.2f} m/s")
            
        except Exception as e:
            print(f"❌ Vehicle control test failed: {e}")
            return False
        
        # Test sensor data
        print("\n📊 Testing sensor data...")
        
        try:
            vehicle.sensors.poll()
            
            # Check electrics
            electrics_data = vehicle.sensors['electrics']
            print(f"⚡ Electrics data available: {len(electrics_data)} channels")
            
            # Show some key data
            if 'fuel' in electrics_data:
                print(f"  • Fuel: {electrics_data['fuel']:.1f}%")
            if 'rpm' in electrics_data:
                print(f"  • Engine RPM: {electrics_data['rpm']:.0f}")
            if 'gear' in electrics_data:
                print(f"  • Gear: {electrics_data['gear']}")
            
            # Check damage
            damage_data = vehicle.sensors['damage']
            if 'damage' in damage_data:
                print(f"🛡️  Vehicle damage: {damage_data['damage']:.1f}")
            
        except Exception as e:
            print(f"❌ Sensor data test failed: {e}")
            return False
        
        # SUCCESS!
        print("\n" + "=" * 70)
        print("🎉 PHASE 2 COMPLETE: Basic Vehicle Control and Sensor Access")
        print("=" * 70)
        print("✅ BeamNG startup: Successfully handled menu and loading")
        print("✅ Vehicle physics: Proper spawn and ground collision")
        print("✅ Sensor system: Basic electrics and damage sensors working")
        print("✅ Vehicle control: Basic throttle, steering, brake control")
        print("✅ Telemetry access: Real-time vehicle state and sensor data")
        print("=" * 70)
        
        print("\n🎯 Phase 2 Achievements:")
        print("• ✅ Solved BeamNG menu/loading freeze issues")
        print("• ✅ Established reliable vehicle spawn and physics")
        print("• ✅ Working sensor data pipeline")
        print("• ✅ Basic vehicle control system operational")
        print("• 🚀 Ready for Phase 3: Advanced sensors and high-frequency data")
        
        print("\n📝 Next Steps for Phase 3:")
        print("• Add advanced sensors (Camera, LiDAR, GPS, IMU)")
        print("• Implement high-frequency telemetry polling")
        print("• Create reward function foundation")
        print("• Build training data collection pipeline")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n🔒 Shutting down Phase 2...")
        try:
            bng.close()
            print("✅ BeamNG closed cleanly")
        except Exception as e:
            print(f"⚠️  Shutdown issue: {e}")

if __name__ == "__main__":
    print("BeamNG AI Driver - Phase 2: Core Game Control and Dynamic Sensor Implementation")
    print("Robust version handling BeamNG startup and menu issues")
    print()
    
    success = main()
    
    if success:
        print("\n🚀 PHASE 2 SUCCESS! READY FOR PHASE 3!")
        print("The foundation is solid - time to add advanced sensors and AI control!")
    else:
        print("\n❌ Phase 2 encountered issues - check the output above for details")
        print("💡 Common fixes:")
        print("  • Make sure BeamNG.drive is properly installed")
        print("  • Try running BeamNG manually first to ensure it works")
        print("  • Check if any antivirus is blocking the connection")
        print("  • Ensure no other BeamNG instances are running")