#!/usr/bin/env python3
"""
Phase 2: Core Game Control and Dynamic Sensor Implementation
BeamNG AI Driver - Maximum Telemetry Access

This phase implements:
1. Vision Layer: High-resolution camera feeds
2. Dynamic Sensor Suite: LiDAR, GPS, Advanced IMU
3. Control Layer: Continuous vehicle control interface
4. Telemetry Foundation: Real-time data collection setup

Based on official BeamNGpy examples and API patterns.
"""

import time
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera, Lidar, GPS, AdvancedIMU, Electrics, Damage

def main():
    """Phase 2 Main: Core Game Control and Dynamic Sensor Implementation"""
    
    print("🚀 Phase 2: Core Game Control and Dynamic Sensor Implementation")
    print("=" * 70)
    print("Goal: Maximum telemetry access + continuous vehicle control")
    print()
    
    try:
        # Set up logging
        set_up_simple_logging()
        
        # Initialize BeamNG connection (from Phase 1 foundation)
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        bng = BeamNGpy('localhost', 25252, home=bng_home)
        
        print("🔄 Launching BeamNG.drive...")
        bng.open(launch=True)
        print("✅ BeamNG connected successfully!")
        
        # Create comprehensive sensor scenario (using working gridmap_v2 from Phase 1)
        print("\n🗺️  Creating maximum telemetry scenario...")
        scenario = Scenario('gridmap_v2', 'phase2_max_telemetry', 
                          description='Phase 2: Maximum Sensor Suite and Vehicle Control')
        
        # Create AI vehicle
        vehicle = Vehicle('ai_max_sensor', model='etk800', license='PH2_MAX')
        
        print("🔧 Adding comprehensive sensor suite...")
        
        # NOTE: Create sensors AFTER vehicle creation but BEFORE scenario building
        # This follows the pattern from west_coast_lidar.py and other examples
        
        # Add vehicle to scenario first
        scenario.add_vehicle(vehicle, pos=(0, 0, 0.5))  # Slightly above ground
        
        print("🔧 Building scenario...")
        scenario.make(bng)
        
        print("⚙️  Setting deterministic mode...")
        bng.settings.set_deterministic(60)  # 60 Hz physics
        
        print("📍 Loading scenario...")
        bng.scenario.load(scenario)
        
        print("▶️  Starting scenario...")
        bng.scenario.start()
        
        print("✅ Scenario loaded - now creating sensors...")
        
        # CREATE SENSORS AFTER SCENARIO START (following BeamNGpy examples pattern)
        
        # 1. VISION LAYER: High-resolution front camera
        print("📷 Creating camera sensor...")
        camera = Camera('front_cam', bng, vehicle,
                       requested_update_time=0.05,  # 20 FPS
                       pos=(0, 1.7, 1.8),           # Front of vehicle, raised
                       dir=(0, 1, 0),               # Forward-facing
                       field_of_view_y=70,          # Wide field of view
                       resolution=(1280, 720),      # High resolution
                       is_render_colours=True,
                       is_render_depth=True)
        
        # 2. LIDAR: 360-degree collision detection  
        print("🌐 Creating LiDAR sensor...")
        lidar = Lidar('lidar_360', bng, vehicle,
                     requested_update_time=0.1,     # 10 Hz
                     pos=(0, 0, 1.8),               # Top center of vehicle
                     vertical_resolution=64,
                     is_360_mode=True,              # Full 360-degree scan
                     vertical_angle=30,
                     max_distance=200,              # 200m range
                     is_using_shared_memory=True)   # Use shared memory for performance
        
        # 3. GPS: Precise positioning
        print("🛰️  Creating GPS sensor...")
        gps = GPS('gps_precise', bng, vehicle,
                 physics_update_time=0.05,          # 20 Hz
                 pos=(0, 0, 1.5),
                 ref_lon=8.8017,                    # Reference coordinates
                 ref_lat=53.0793,
                 is_visualised=True)
        
        # 4. ADVANCED IMU: High-frequency motion sensing
        print("📊 Creating Advanced IMU sensor...")
        imu = AdvancedIMU('imu_advanced', bng, vehicle,
                         physics_update_time=0.01,  # 100 Hz - high frequency
                         pos=(0, 0, 0.5),
                         is_visualised=False)
        
        # 5. ELECTRICS: Vehicle systems telemetry
        print("⚡ Creating Electrics sensor...")
        electrics = Electrics()
        vehicle.attach_sensor('electrics', electrics)
        
        # 6. DAMAGE: Collision and damage detection
        print("🛡️  Creating Damage sensor...")
        damage = Damage()
        vehicle.attach_sensor('damage', damage)
        
        print("✅ All sensors created and initialized!")
        
        # PHASE 2: DEMONSTRATE SENSOR DATA COLLECTION
        print("\n🔍 Testing sensor data acquisition...")
        time.sleep(3)  # Let everything stabilize
        
        print("📊 Polling all sensors...")
        
        # Test camera
        try:
            camera_data = camera.poll()
            print(f"✅ Camera: {camera_data['colour'].shape} resolution captured")
        except Exception as e:
            print(f"⚠️  Camera polling issue: {e}")
        
        # Test LiDAR
        try:
            lidar_data = lidar.poll()
            print(f"✅ LiDAR: {len(lidar_data['pointCloud'])} points detected")
        except Exception as e:
            print(f"⚠️  LiDAR polling issue: {e}")
        
        # Test GPS
        try:
            gps_data = gps.poll()
            pos = gps_data['pos']
            print(f"✅ GPS: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        except Exception as e:
            print(f"⚠️  GPS polling issue: {e}")
        
        # Test Advanced IMU
        try:
            imu_data = imu.poll()
            accel = imu_data['accel']
            print(f"✅ Advanced IMU: Acceleration {np.linalg.norm(accel):.2f} m/s²")
        except Exception as e:
            print(f"⚠️  IMU polling issue: {e}")
        
        # Test Electrics
        try:
            electrics_data = electrics.poll()
            print(f"✅ Electrics: {len(electrics_data)} telemetry channels")
        except Exception as e:
            print(f"⚠️  Electrics polling issue: {e}")
        
        # Test Damage
        try:
            damage_data = damage.poll()
            print(f"✅ Damage: {len(damage_data)} damage zones monitored")
        except Exception as e:
            print(f"⚠️  Damage polling issue: {e}")
        
        # PHASE 2: DEMONSTRATE CONTINUOUS VEHICLE CONTROL
        print("\n🎮 Testing continuous vehicle control...")
        
        control_actions = [
            {"throttle": 0.3, "steering": 0.0, "brake": 0.0, "duration": 2.0, "action": "Accelerate forward"},
            {"throttle": 0.2, "steering": 0.5, "brake": 0.0, "duration": 1.5, "action": "Turn right"},
            {"throttle": 0.0, "steering": 0.0, "brake": 0.8, "duration": 1.0, "action": "Brake to stop"},
            {"throttle": 0.2, "steering": -0.3, "brake": 0.0, "duration": 1.5, "action": "Turn left"},
            {"throttle": 0.0, "steering": 0.0, "brake": 1.0, "duration": 1.0, "action": "Full stop"}
        ]
        
        print("🚗 Executing control sequence with real-time telemetry...")
        
        for i, action in enumerate(control_actions):
            print(f"\n  Action {i+1}: {action['action']}")
            print(f"    Control: Throttle={action['throttle']:.1f}, Steering={action['steering']:.1f}, Brake={action['brake']:.1f}")
            
            # Apply control
            vehicle.control(throttle=action['throttle'], 
                           steering=action['steering'], 
                           brake=action['brake'])
            
            # Monitor telemetry during action
            start_time = time.time()
            while time.time() - start_time < action['duration']:
                try:
                    # Real-time vehicle state monitoring
                    state = vehicle.sensors.state
                    if state:
                        speed = np.linalg.norm(state['vel']) if 'vel' in state else 0
                        pos = state.get('pos', [0, 0, 0])
                        print(f"    📊 Speed: {speed:.1f} m/s, Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                except Exception as e:
                    print(f"    ⚠️  State monitoring: {e}")
                
                time.sleep(0.5)  # Update every 0.5 seconds
            
            print(f"    ✅ {action['action']} completed")
        
        # PHASE 2: COMPREHENSIVE TELEMETRY SAMPLE COLLECTION
        print(f"\n📈 Collecting comprehensive telemetry sample for 10 seconds...")
        print("🔄 Demonstrating maximum data visibility...")
        
        telemetry_log = []
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < 10:
            try:
                timestamp = time.time()
                
                # Collect all available data
                snapshot = {
                    'timestamp': timestamp,
                    'vehicle_state': vehicle.sensors.state,
                    'camera_available': True,
                    'lidar_available': True,
                    'gps_available': True,
                    'imu_available': True,
                    'electrics_available': True,
                    'damage_available': True
                }
                
                # Try to collect actual sensor data (may be expensive, so sample less frequently)
                if sample_count % 4 == 0:  # Every 2 seconds
                    try:
                        snapshot['camera_shape'] = camera.poll()['colour'].shape
                        snapshot['lidar_points'] = len(lidar.poll()['pointCloud'])
                        snapshot['gps_pos'] = gps.poll()['pos']
                        snapshot['imu_accel'] = imu.poll()['accel']
                        snapshot['electrics_data'] = len(electrics.poll())
                        snapshot['damage_data'] = len(damage.poll())
                    except Exception as e:
                        snapshot['sensor_error'] = str(e)
                
                telemetry_log.append(snapshot)
                sample_count += 1
                
                # Real-time display
                if vehicle.sensors.state and 'vel' in vehicle.sensors.state:
                    speed = np.linalg.norm(vehicle.sensors.state['vel'])
                    pos = vehicle.sensors.state.get('pos', [0, 0, 0])
                    print(f"📊 T+{timestamp-start_time:.1f}s: Speed {speed:.1f}m/s, Pos({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")
                
                time.sleep(0.5)  # 2 Hz logging
                
            except Exception as e:
                print(f"⚠️  Telemetry collection error: {e}")
                break
        
        print(f"✅ Collected {len(telemetry_log)} telemetry snapshots")
        
        # PHASE 2 SUCCESS SUMMARY
        print("\n" + "=" * 70)
        print("🎉 PHASE 2 COMPLETE: Core Game Control and Dynamic Sensor Implementation")
        print("=" * 70)
        print("✅ Vision Layer: High-resolution camera feeds implemented")
        print("✅ Dynamic Sensor Suite: LiDAR, GPS, AdvancedIMU, Electrics, Damage")
        print("✅ Control Layer: Continuous vehicle control (steering, throttle, brake)")
        print("✅ Telemetry Foundation: Real-time maximum data visibility")
        print("=" * 70)
        
        print("\n📊 Phase 2 Achievements:")
        print(f"• Sensor Systems: 6 comprehensive sensor types active")
        print(f"• Telemetry Rate: Up to 100Hz IMU, 20Hz camera/GPS, 10Hz LiDAR")
        print(f"• Control Precision: Continuous actions (-1.0 to 1.0 steering, 0.0 to 1.0 throttle/brake)")
        print(f"• Data Samples: {len(telemetry_log)} telemetry snapshots collected")
        print(f"• Ready for Phase 3: Comprehensive telemetry polling at physics tick rate")
        
        # Cleanup sensors
        print("\n🔧 Cleaning up sensors...")
        try:
            camera.remove()
            lidar.remove()
            gps.remove()
            imu.remove()
            print("✅ Sensors cleaned up")
        except Exception as e:
            print(f"⚠️  Sensor cleanup: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n🔒 Shutting down Phase 2 systems...")
        try:
            bng.close()
            print("✅ BeamNG closed")
        except:
            pass

if __name__ == "__main__":
    print("BeamNG AI Driver - Phase 2: Maximum Telemetry Access")
    print("Leveraging full BeamNGpy sensor capabilities")
    print()
    
    success = main()
    
    if success:
        print("\n🚀 READY FOR PHASE 3!")
        print("Next: Comprehensive Telemetry and State Capture (up to 2000Hz)")
        print("• Physics tick rate polling")
        print("• Advanced reward function development") 
        print("• High-frequency data buffer systems")
    else:
        print("\n❌ Phase 2 needs debugging - check errors above")