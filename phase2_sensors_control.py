#!/usr/bin/env python3
"""
Phase 2: Core Game Control and Dynamic Sensor Implementation
BeamNG AI Driver - Maximum Telemetry Access

This phase implements:
1. Vision Layer: High-resolution camera feeds
2. Dynamic Sensor Suite: LiDAR, GPS, Advanced IMU
3. Control Layer: Continuous vehicle control interface
4. Telemetry Foundation: Real-time data collection setup
"""

import time
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera, Lidar, GPS, AdvancedIMU, Electrics, Damage

class BeamNGAISensorSuite:
    """
    Maximum Telemetry Access - Comprehensive sensor and control system
    Leverages ALL available BeamNGpy sensor capabilities
    """
    
    def __init__(self):
        self.bng = None
        self.vehicle = None
        self.sensors_initialized = False
        
        # Sensor data storage
        self.camera_data = None
        self.lidar_data = None
        self.gps_data = None
        self.imu_data = None
        self.electrics_data = None
        self.damage_data = None
        self.vehicle_state = None
        
    def initialize_beamng_connection(self):
        """Initialize BeamNG connection using Phase 1 foundation"""
        print("🚀 Phase 2: Core Game Control and Dynamic Sensor Implementation")
        print("=" * 70)
        print("Goal: Maximum telemetry access + continuous vehicle control")
        print()
        
        # Set up logging
        set_up_simple_logging()
        
        # BeamNG configuration (from Phase 1)
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        self.bng = BeamNGpy('localhost', 25252, home=bng_home)
        
        print("🔄 Launching BeamNG.drive...")
        self.bng.open(launch=True)
        print("✅ BeamNG connected successfully!")
        
        return True
    
    def create_maximum_telemetry_scenario(self):
        """Create scenario with comprehensive sensor-equipped vehicle"""
        print("\n🗺️  Creating maximum telemetry scenario...")
        
        # Use working level from Phase 1
        scenario = Scenario('gridmap_v2', 'ai_max_telemetry', 
                          description='AI Driver - Phase 2: Maximum Sensor Suite')
        
        # Create AI vehicle with comprehensive sensor package
        self.vehicle = Vehicle('ai_sensor_car', model='etk800', license='MAX_TEL')
        
        print("🔧 Adding comprehensive sensor suite...")
        
        # 1. VISION LAYER: High-resolution front camera
        front_camera = Camera('front_cam', self.bng, self.vehicle,
                            requested_update_time=0.05,  # 20 FPS
                            pos=(0, 1.7, 1.8),  # Front of vehicle, raised
                            dir=(0, 1, 0),      # Forward-facing
                            field_of_view_y=70,  # Wide field of view
                            resolution=(1280, 720),  # High resolution
                            is_render_colours=True,
                            is_render_annotations=False,
                            is_render_instance=False,
                            is_render_depth=True)  # Include depth for 3D understanding
        
        # 2. LIDAR: 360-degree collision detection
        lidar_sensor = Lidar('lidar_360', self.bng, self.vehicle,
                           requested_update_time=0.1,  # 10 Hz
                           pos=(0, 0, 1.8),     # Top center of vehicle
                           vertical_resolution=64,
                           is_360_mode=True,  # Full 360-degree scan
                           vertical_angle=30,
                           max_distance=200,      # 200m range
                           is_using_shared_memory=True)  # Use shared memory for performance
        
        # 3. GPS: Precise positioning
        gps_sensor = GPS('gps_precise', self.bng, self.vehicle,
                        physics_update_time=0.05,  # 20 Hz
                        pos=(0, 0, 1.5),
                        ref_lon=8.8017,  # Reference coordinates (default)
                        ref_lat=53.0793,
                        is_visualised=True)
        
        # 4. ADVANCED IMU: Advanced motion sensing
        imu_sensor = AdvancedIMU('imu_advanced', self.bng, self.vehicle,
                        physics_update_time=0.01,  # 100 Hz - high frequency
                        pos=(0, 0, 0.5),
                        is_visualised=False)
        
        # 5. ELECTRICS: Vehicle systems telemetry
        electrics_sensor = Electrics()
        
        # 6. DAMAGE: Collision and damage detection
        damage_sensor = Damage()
        
        # Attach all sensors to vehicle using proper API
        self.vehicle.attach_sensor('front_cam', front_camera)
        self.vehicle.attach_sensor('lidar_360', lidar_sensor)
        self.vehicle.attach_sensor('gps_precise', gps_sensor)
        self.vehicle.attach_sensor('imu_advanced', imu_sensor)
        self.vehicle.attach_sensor('electrics_full', electrics_sensor)
        self.vehicle.attach_sensor('damage_monitor', damage_sensor)
        
        # Add vehicle to scenario
        scenario.add_vehicle(self.vehicle, pos=(0, 0, 0.5))
        
        print("🔧 Building scenario...")
        scenario.make(self.bng)
        
        print("📍 Loading scenario...")
        self.bng.scenario.load(scenario)
        
        print("▶️  Starting scenario...")
        self.bng.scenario.start()
        
        print("✅ Maximum telemetry scenario loaded!")
        return True
    
    def initialize_sensors(self):
        """Initialize all sensors and verify data streams"""
        print("\n🔍 Initializing sensor data streams...")
        
        time.sleep(3)  # Let everything stabilize
        
        try:
            # Poll initial sensor data to verify all systems
            print("📊 Testing sensor data acquisition...")
            
            # Test camera
            self.camera_data = self.vehicle.sensors['front_cam'].poll()
            print(f"✅ Camera: {self.camera_data['colour'].shape} resolution")
            
            # Test LiDAR
            self.lidar_data = self.vehicle.sensors['lidar_360'].poll()
            print(f"✅ LiDAR: {len(self.lidar_data['points'])} points detected")
            
            # Test GPS
            self.gps_data = self.vehicle.sensors['gps_precise'].poll()
            print(f"✅ GPS: Position ({self.gps_data['pos'][0]:.2f}, {self.gps_data['pos'][1]:.2f}, {self.gps_data['pos'][2]:.2f})")
            
            # Test IMU
            self.imu_data = self.vehicle.sensors['imu_advanced'].poll()
            print(f"✅ IMU: Acceleration {np.linalg.norm(self.imu_data['accel']):.2f} m/s²")
            
            # Test Electrics
            self.electrics_data = self.vehicle.sensors['electrics_full'].poll()
            print(f"✅ Electrics: {len(self.electrics_data)} telemetry channels")
            
            # Test Damage
            self.damage_data = self.vehicle.sensors['damage_monitor'].poll()
            print(f"✅ Damage: {len(self.damage_data)} damage zones monitored")
            
            self.sensors_initialized = True
            print("🎯 All sensors initialized successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Sensor initialization error: {e}")
            return False
    
    def demonstrate_vehicle_control(self):
        """Demonstrate continuous vehicle control interface"""
        print("\n🎮 Testing continuous vehicle control...")
        
        if not self.sensors_initialized:
            print("❌ Sensors must be initialized first!")
            return False
        
        print("🚗 Executing control sequence:")
        
        try:
            # Control sequence: Forward, turn, brake
            control_actions = [
                {"throttle": 0.3, "steering": 0.0, "brake": 0.0, "duration": 2.0, "action": "Accelerate forward"},
                {"throttle": 0.2, "steering": 0.5, "brake": 0.0, "duration": 1.5, "action": "Turn right"},
                {"throttle": 0.0, "steering": 0.0, "brake": 0.8, "duration": 1.0, "action": "Brake to stop"},
                {"throttle": 0.2, "steering": -0.3, "brake": 0.0, "duration": 1.5, "action": "Turn left"},
                {"throttle": 0.0, "steering": 0.0, "brake": 1.0, "duration": 1.0, "action": "Full stop"}
            ]
            
            for i, action in enumerate(control_actions):
                print(f"  Action {i+1}: {action['action']}")
                print(f"    Throttle: {action['throttle']:.1f}, Steering: {action['steering']:.1f}, Brake: {action['brake']:.1f}")
                
                # Apply control
                self.vehicle.control(throttle=action['throttle'], 
                                   steering=action['steering'], 
                                   brake=action['brake'])
                
                # Monitor telemetry during action
                start_time = time.time()
                while time.time() - start_time < action['duration']:
                    # Real-time telemetry monitoring
                    state = self.vehicle.sensors.state
                    if state:
                        speed = np.linalg.norm(state['vel']) if 'vel' in state else 0
                        pos = state.get('pos', [0, 0, 0])
                        print(f"    📊 Speed: {speed:.1f} m/s, Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
                    
                    time.sleep(0.5)  # Update every 0.5 seconds
                
                print(f"    ✅ {action['action']} completed")
            
            print("🎯 Vehicle control demonstration complete!")
            return True
            
        except Exception as e:
            print(f"❌ Control error: {e}")
            return False
    
    def collect_telemetry_sample(self, duration=10):
        """Collect comprehensive telemetry data sample"""
        print(f"\n📈 Collecting {duration}s telemetry sample...")
        print("🔄 Demonstrating maximum data visibility...")
        
        telemetry_log = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Collect timestamp
                timestamp = time.time()
                
                # Vehicle state (physics)
                state = self.vehicle.sensors.state
                
                # All sensor data
                camera = self.vehicle.sensors['front_cam'].poll()
                lidar = self.vehicle.sensors['lidar_360'].poll()
                gps = self.vehicle.sensors['gps_precise'].poll()
                imu = self.vehicle.sensors['imu_advanced'].poll()
                electrics = self.vehicle.sensors['electrics_full'].poll()
                damage = self.vehicle.sensors['damage_monitor'].poll()
                
                # Create telemetry snapshot
                snapshot = {
                    'timestamp': timestamp,
                    'vehicle_state': state,
                    'camera_resolution': camera['colour'].shape if camera else None,
                    'lidar_points': len(lidar['points']) if lidar else 0,
                    'gps_position': gps['pos'] if gps else None,
                    'imu_accel': imu['accel'] if imu else None,
                    'electrics_channels': len(electrics) if electrics else 0,
                    'damage_zones': len(damage) if damage else 0
                }
                
                telemetry_log.append(snapshot)
                
                # Real-time display
                if state and 'vel' in state:
                    speed = np.linalg.norm(state['vel'])
                    pos = state.get('pos', [0, 0, 0])
                    print(f"📊 T+{timestamp-start_time:.1f}s: Speed {speed:.1f}m/s, "
                          f"Pos({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}), "
                          f"LiDAR {len(lidar['points']) if lidar else 0} pts")
                
                time.sleep(0.5)  # 2 Hz logging for demo
                
            except Exception as e:
                print(f"⚠️  Telemetry collection error: {e}")
                break
        
        print(f"✅ Collected {len(telemetry_log)} telemetry snapshots")
        return telemetry_log
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n🔒 Shutting down Phase 2 systems...")
        if self.bng:
            self.bng.close()
        print("✅ Phase 2 cleanup complete")

def main():
    """Phase 2 Main: Core Game Control and Dynamic Sensor Implementation"""
    
    ai_system = BeamNGAISensorSuite()
    
    try:
        # Initialize BeamNG connection
        if not ai_system.initialize_beamng_connection():
            return False
        
        # Create comprehensive sensor scenario
        if not ai_system.create_maximum_telemetry_scenario():
            return False
        
        # Initialize all sensors
        if not ai_system.initialize_sensors():
            return False
        
        # Demonstrate vehicle control
        if not ai_system.demonstrate_vehicle_control():
            return False
        
        # Collect telemetry sample
        telemetry_data = ai_system.collect_telemetry_sample(duration=15)
        
        # Phase 2 Success Summary
        print("\n" + "=" * 70)
        print("🎉 PHASE 2 COMPLETE: Core Game Control and Dynamic Sensor Implementation")
        print("=" * 70)
        print("✅ Vision Layer: High-resolution camera feeds implemented")
        print("✅ Dynamic Sensor Suite: LiDAR, GPS, IMU, Electrics, Damage")
        print("✅ Control Layer: Continuous vehicle control (steering, throttle, brake)")
        print("✅ Telemetry Foundation: Real-time maximum data visibility")
        print("=" * 70)
        
        print("\n📊 Phase 2 Achievements:")
        print(f"• Sensor Systems: 6 comprehensive sensor types active")
        print(f"• Telemetry Rate: Up to 100Hz IMU, 20Hz camera/GPS, 10Hz LiDAR")
        print(f"• Control Precision: Continuous actions (-1.0 to 1.0 steering, 0.0 to 1.0 throttle/brake)")
        print(f"• Data Samples: {len(telemetry_data)} telemetry snapshots collected")
        print(f"• Ready for Phase 3: Comprehensive telemetry polling at physics tick rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        ai_system.cleanup()

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