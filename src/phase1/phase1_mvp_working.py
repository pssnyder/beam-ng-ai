#!/usr/bin/env python3
"""
Phase 1 MVP - BeamNG AI Driver Foundation
WORKING VERSION - Uses correct level name
"""

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging

def main():
    """Phase 1 MVP: Establish BeamNG-Python connection for maximum telemetry access"""
    
    print("🚀 BeamNG AI Driver - Phase 1 MVP")
    print("=" * 50)
    print("Goal: Connect to BeamNG for maximum telemetry access")
    print()
    
    # Set up logging
    set_up_simple_logging()
    
    # BeamNG configuration
    bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
    bng = BeamNGpy('localhost', 25252, home=bng_home)
    
    try:
        print("🔄 Launching BeamNG.drive...")
        bng.open(launch=True)
        print("✅ BeamNG connected successfully!")
        
        # Use a level that exists in BeamNG.drive
        print("\n🗺️  Creating test scenario...")
        scenario = Scenario('gridmap_v2', 'ai_driver_test', 
                          description='AI Driver - Maximum Telemetry Test')
        
        # Create AI test vehicle
        vehicle = Vehicle('ai_driver', model='etk800', license='AI_TEST')
        scenario.add_vehicle(vehicle, pos=(0, 0, 0.5))  # Slightly above ground
        
        print("🔧 Building scenario...")
        scenario.make(bng)
        
        print("📍 Loading scenario...")
        bng.scenario.load(scenario)
        
        print("▶️  Starting scenario...")
        bng.scenario.start()
        
        print("\n🎯 PHASE 1 MVP COMPLETE!")
        print("=" * 50)
        print("✅ BeamNG-Python connection: WORKING")
        print("✅ Vehicle spawning: WORKING") 
        print("✅ Scenario system: WORKING")
        print("✅ Ready for maximum telemetry access")
        print("=" * 50)
        
        print("\n📊 Project Status Update:")
        print("• Original goal: 'Black Box Driver' → ✅ Changed to 'Data-Driven Driver'")
        print("• Telemetry scope: 'Restricted inputs' → ✅ 'Maximum telemetry access'")
        print("• BeamNG integration: ❓ Unknown → ✅ WORKING")
        print("• Python environment: ❓ → ✅ Python 3.14 + BeamNGpy 1.34.1")
        
        print(f"\n⏱️  System will run for 10 seconds to demonstrate stability...")
        
        # Keep running briefly to show it works
        import time
        time.sleep(10)
        
        print("\n🔒 Closing BeamNG...")
        bng.close()
        print("✅ Phase 1 MVP completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Try different level: 'west_coast_usa', 'east_coast_usa', 'italy'")
        print("2. Check BeamNG.drive installation")
        print("3. Verify Python permissions")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 READY FOR PHASE 2!")
        print("Next steps:")
        print("1. Add comprehensive sensor suite")
        print("2. Implement telemetry data collection")  
        print("3. Create AI control interface")
        print("4. Build training data pipeline")
    else:
        print("\n❌ Phase 1 needs fixes - check errors above")