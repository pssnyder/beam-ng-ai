#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5A: GPS Sensor Integration
BeamNG AI Driver - Navigation Foundation

This file integrates the GPS sensor to provide:
- Latitude/Longitude coordinates
- Altitude
- Heading
- Velocity

Dependencies:
- Phase 4C complete (SAC neural network training)
- BeamNGpy GPS sensor support

Goal: Verify GPS sensor functionality and expose navigation data
"""

import time
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Electrics, Damage, GForces, GPS

# ============================================================================
# CONFIGURATION
# ============================================================================

# BeamNG connection
BNG_HOME = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
BNG_HOST = 'localhost'
BNG_PORT = 25252

# Map and spawn
MAP_NAME = 'west_coast_usa'
SPAWN_POS = (-717.121, 101.0, 118.675)
SPAWN_ROT = (0, 0, 0.3826834, 0.9238795)

# GPS sensor configuration
GPS_UPDATE_RATE = 10  # Hz (10 updates per second)

# ============================================================================
# GPS INTEGRATION TEST
# ============================================================================

def test_gps_sensor():
    """Test GPS sensor integration and data quality"""
    print("=" * 60)
    print("Phase 5A: GPS Sensor Integration Test")
    print("=" * 60)
    
    # Setup logging
    set_up_simple_logging()
    
    # Connect to BeamNG
    print(f"\n[1/6] Connecting to BeamNG at {BNG_HOST}:{BNG_PORT}...")
    bng = BeamNGpy(BNG_HOST, BNG_PORT, home=BNG_HOME)
    
    try:
        bng.open(launch=False)  # Connect to running instance
        print("  [OK] Connected to BeamNG")
    except Exception as e:
        print(f"  [ERROR] Connection failed: {e}")
        print("  Make sure BeamNG.drive is running!")
        return False
    
    # Create scenario
    print(f"\n[2/6] Setting up scenario on {MAP_NAME}...")
    scenario = Scenario(MAP_NAME, 'gps_integration_test', 
                       description='Phase 5A GPS sensor testing')
    
    # Create vehicle with GPS sensor
    vehicle = Vehicle('test_vehicle', model='etk800', license='GPS_TEST')
    
    # Attach all sensors (GPS + existing sensors)
    print("\n[3/6] Attaching sensors...")
    vehicle.sensors.attach('electrics', Electrics())
    vehicle.sensors.attach('damage', Damage())
    vehicle.sensors.attach('gforces', GForces())
    vehicle.sensors.attach('gps', GPS())  # NEW: GPS sensor
    print("  [OK] Sensors attached: Electrics, Damage, GForces, GPS")
    
    # Add vehicle to scenario
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT)
    
    # Load scenario
    print("\n[4/6] Loading scenario...")
    scenario.make(bng)
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.scenario.start()
    
    print("  [OK] Scenario loaded, waiting for physics stabilization...")
    time.sleep(3)
    
    # Test GPS sensor
    print("\n[5/6] Testing GPS sensor data...")
    vehicle.sensors.poll()
    
    # Get GPS data
    gps_data = vehicle.sensors.get('gps')
    
    if gps_data is None:
        print("  [ERROR] GPS sensor returned None!")
        print("  This may indicate GPS is not available in BeamNG.drive")
        print("  (GPS might require BeamNG.tech research license)")
        return False
    
    print("\n  GPS Data Structure:")
    print(f"  {gps_data}")
    
    # Parse GPS data
    print("\n  GPS Fields:")
    for key, value in gps_data.items():
        print(f"    {key}: {value}")
    
    # Test GPS updates over time
    print("\n[6/6] Testing GPS update rate (5 seconds)...")
    print("  Moving vehicle with throttle to generate data...\n")
    
    vehicle.control(throttle=0.5, steering=0, brake=0, parkingbrake=0)
    
    gps_samples = []
    start_time = time.time()
    
    while time.time() - start_time < 5.0:
        vehicle.sensors.poll()
        gps = vehicle.sensors.get('gps')
        vehicle_state = vehicle.state
        
        if gps:
            sample = {
                'time': time.time() - start_time,
                'gps': gps.copy(),
                'pos': vehicle_state['pos'],
                'vel': vehicle_state['vel']
            }
            gps_samples.append(sample)
        
        time.sleep(0.1)  # 10Hz sampling
    
    vehicle.control(throttle=0, steering=0, brake=1, parkingbrake=1)
    
    # Analyze GPS data
    print(f"\n  Collected {len(gps_samples)} GPS samples")
    print(f"  Sample rate: {len(gps_samples) / 5.0:.1f} Hz")
    
    if len(gps_samples) > 0:
        print("\n  Sample GPS readings:")
        for i in [0, len(gps_samples)//2, -1]:
            sample = gps_samples[i]
            print(f"\n  Sample {i} (t={sample['time']:.1f}s):")
            print(f"    GPS: {sample['gps']}")
            print(f"    Position: {sample['pos']}")
            print(f"    Velocity: {sample['vel']}")
    
    # GPS availability check
    print("\n" + "=" * 60)
    print("GPS Sensor Test Results:")
    print("=" * 60)
    
    if len(gps_samples) > 0 and gps_samples[0]['gps']:
        print("  [SUCCESS] GPS sensor operational!")
        print(f"  Available fields: {list(gps_samples[0]['gps'].keys())}")
        print("\n  Ready for Phase 5A waypoint navigation")
        success = True
    else:
        print("  [BLOCKED] GPS sensor not available")
        print("  This may require BeamNG.tech research license")
        print("\n  FALLBACK: Use Cartesian coordinates instead of lat/lon")
        print("  - Calculate bearing using atan2(dx, dy)")
        print("  - Calculate distance using norm(pos_target - pos_current)")
        print("  - Vehicle heading from vehicle.state['dir']")
        success = False
    
    # Cleanup
    print("\n[CLEANUP] Disconnecting from BeamNG...")
    bng.disconnect()
    
    return success


def demonstrate_fallback_navigation():
    """
    Demonstrate navigation using Cartesian coordinates (fallback if GPS unavailable)
    This is the approach we'll use for Phase 5 if GPS sensor requires license
    """
    print("\n" + "=" * 60)
    print("Fallback Navigation System (Cartesian-Based)")
    print("=" * 60)
    
    # Example waypoint
    waypoint_pos = np.array([-750.0, 80.0, 118.5])
    vehicle_pos = np.array(SPAWN_POS)
    
    print(f"\nVehicle Position: {vehicle_pos}")
    print(f"Waypoint Position: {waypoint_pos}")
    
    # Calculate distance
    distance = np.linalg.norm(waypoint_pos - vehicle_pos)
    print(f"\nDistance to waypoint: {distance:.2f}m")
    
    # Calculate bearing (angle from north)
    # BeamNG: +Y is north, +X is east
    dx = waypoint_pos[0] - vehicle_pos[0]
    dy = waypoint_pos[1] - vehicle_pos[1]
    bearing_rad = np.arctan2(dx, dy)
    bearing_deg = np.degrees(bearing_rad)
    
    print(f"Bearing to waypoint: {bearing_rad:.3f} rad ({bearing_deg:.1f}°)")
    
    # Vehicle heading (from quaternion or direction vector)
    # For now, assume vehicle starts facing spawn rotation
    print(f"\nVehicle heading: [will be from vehicle.state['dir']]")
    
    # Heading error
    # heading_error = bearing - vehicle_heading
    print(f"Heading error: [bearing - vehicle_heading]")
    
    print("\n[OK] Cartesian navigation is viable fallback!")
    print("  - No external dependencies")
    print("  - Works with existing sensors")
    print("  - Sufficient for Phase 5 goals")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 5A: GPS SENSOR INTEGRATION")
    print("="*60)
    print("\nThis test will:")
    print("1. Connect to running BeamNG instance")
    print("2. Attach GPS sensor to vehicle")
    print("3. Test GPS data collection")
    print("4. Determine if GPS is available (may require BeamNG.tech)")
    print("5. Demonstrate Cartesian fallback if needed")
    print("\nMAKE SURE BeamNG.drive IS RUNNING before proceeding!")
    print("="*60)
    
    input("\nPress ENTER to begin GPS sensor test...")
    
    # Run GPS test
    gps_available = test_gps_sensor()
    
    # If GPS not available, demonstrate fallback
    if not gps_available:
        print("\n" + "="*60)
        input("\nPress ENTER to see Cartesian fallback navigation...")
        demonstrate_fallback_navigation()
    
    print("\n" + "="*60)
    print("Phase 5A GPS Integration Test Complete")
    print("="*60)
    
    if gps_available:
        print("\n[NEXT STEP] Proceed to phase5a_single_waypoint.py")
        print("  - Use GPS for bearing calculation")
        print("  - Implement heading alignment reward")
    else:
        print("\n[NEXT STEP] Proceed to phase5a_single_waypoint.py (Cartesian mode)")
        print("  - Use atan2 for bearing calculation")
        print("  - Extract heading from vehicle.state['dir']")
        print("  - Works without GPS sensor!")
    
    print("="*60 + "\n")
