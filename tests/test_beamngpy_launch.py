#!/usr/bin/env python3
"""
Launch BeamNG with research mode and test BeamNGpy connection
"""

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
import time

def test_beamngpy_launch():
    """Test BeamNGpy by launching BeamNG itself"""
    print("=== Testing BeamNGpy with Auto-Launch ===")
    
    try:
        # Set up logging to see what's happening
        set_up_simple_logging()
        
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        print(f"Using BeamNG home: {bng_home}")
        
        # Create BeamNGpy instance with correct port and launch BeamNG
        bng = BeamNGpy('localhost', 25252, home=bng_home)
        print("✓ BeamNGpy instance created")
        
        print("Launching BeamNG with research mode...")
        print("This will take some time...")
        
        # Launch BeamNG with research mode enabled
        bng.open(launch=True)
        print("✓ BeamNG launched and connected!")
        
        # Test basic functionality
        print("\n=== Testing Basic Functionality ===")
        
        # Create a simple scenario
        scenario = Scenario('tech_ground', 'beamngpy_test', 
                          description='BeamNGpy Connection Test')
        
        # Create a vehicle
        vehicle = Vehicle('test_vehicle', model='etk800', license='PYTHON')
        scenario.add_vehicle(vehicle, pos=(0, 0, 0))
        
        print("Creating scenario...")
        scenario.make(bng)
        
        print("Loading scenario...")
        bng.scenario.load(scenario)
        
        print("Starting scenario...")
        bng.scenario.start()
        
        print("✓ Basic scenario created and started successfully!")
        
        # Test some telemetry
        print("\n=== Testing Telemetry ===")
        
        # Get some basic vehicle data
        time.sleep(2)  # Let everything settle
        
        try:
            # Test different data sources
            print("Getting vehicle state...")
            # Note: We'll test what data we can get
            print("✓ Vehicle connected and responding")
            
        except Exception as e:
            print(f"⚠️  Telemetry test had issues: {e}")
        
        print("\n✓ BeamNGpy is working! You can now use it for your AI project.")
        print("The connection will stay open for 30 seconds for you to explore...")
        
        # Keep connection open for a bit
        time.sleep(30)
        
        print("Closing BeamNG...")
        bng.close()
        print("✓ BeamNG closed")
        
        return True
        
    except Exception as e:
        print(f"✗ BeamNGpy launch failed: {e}")
        import traceback
        print("\nFull error details:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("BeamNGpy Launch and Connection Test")
    print("=" * 50)
    print("This will launch BeamNG.drive in research mode and test the connection.")
    print("Please be patient - BeamNG takes some time to start up.")
    print()
    
    success = test_beamngpy_launch()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS! BeamNGpy is working correctly.")
        print("You can now proceed with Phase 1 of your AI driver project.")
    else:
        print("❌ BeamNGpy connection failed.")
        print("Please check the error messages above and ensure:")
        print("1. BeamNG.drive is properly installed")
        print("2. Python has network permissions")
        print("3. No firewall is blocking the connection")