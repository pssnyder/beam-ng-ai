#!/usr/bin/env python3
"""
Phase 2 Diagnostic: Menu Freeze Investigation
Simple test to identify and resolve BeamNG menu freeze issues
"""

import time
from beamngpy import BeamNGpy, set_up_simple_logging

def test_beamng_startup():
    """Test BeamNG startup and responsiveness"""
    
    print("🔍 Phase 2 Diagnostic: BeamNG Menu Freeze Investigation")
    print("=" * 60)
    
    try:
        set_up_simple_logging()
        
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        bng = BeamNGpy('localhost', 25252, home=bng_home)
        
        print("🔄 Step 1: Launching BeamNG...")
        start_time = time.time()
        bng.open(launch=True)
        launch_time = time.time() - start_time
        print(f"✅ BeamNG process launched in {launch_time:.1f}s")
        
        print("⏱️  Step 2: Waiting for initial connection...")
        time.sleep(10)  # Give BeamNG time to start
        
        print("🔍 Step 3: Testing basic responsiveness...")
        for attempt in range(6):  # Try for 30 seconds
            try:
                print(f"  Attempt {attempt + 1}/6: Testing connection...")
                
                # Try a simple operation that should work if BeamNG is responsive
                bng.control.step(1)
                print(f"  ✅ BeamNG responded on attempt {attempt + 1}")
                break
                
            except Exception as e:
                print(f"  ⏳ Not responsive yet: {e}")
                if attempt < 5:
                    time.sleep(5)
                else:
                    print("  ❌ BeamNG failed to become responsive")
                    return False
        
        print("🎯 Step 4: Testing scenario creation...")
        try:
            from beamngpy import Scenario, Vehicle
            
            # Try to create a very simple scenario
            scenario = Scenario('gridmap_v2', 'diagnostic_test')
            vehicle = Vehicle('test', model='etk800')
            scenario.add_vehicle(vehicle, pos=(0, 0, 100))
            
            print("✅ Scenario creation successful")
            
            # Try to build scenario
            print("🔧 Building scenario...")
            scenario.make(bng)
            print("✅ Scenario build successful")
            
            # Try to load scenario
            print("📍 Loading scenario...")
            bng.scenario.load(scenario)
            print("✅ Scenario load successful")
            
            # Try to start scenario
            print("▶️  Starting scenario...")
            bng.scenario.start()
            print("✅ Scenario start successful")
            
            print("🎉 SUCCESS! BeamNG is working properly")
            return True
            
        except Exception as e:
            print(f"❌ Scenario test failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("🔒 Closing BeamNG...")
        try:
            bng.close()
            print("✅ BeamNG closed")
        except:
            print("⚠️  Force closing")

if __name__ == "__main__":
    print("BeamNG AI Driver - Phase 2 Diagnostic")
    print("Testing BeamNG startup and menu freeze issues")
    print()
    
    success = test_beamng_startup()
    
    if success:
        print("\n🎉 DIAGNOSTIC SUCCESS!")
        print("BeamNG startup is working correctly.")
        print("You can now proceed with full Phase 2 implementation.")
    else:
        print("\n❌ DIAGNOSTIC FAILED")
        print("BeamNG has startup issues. Common solutions:")
        print("• Close any existing BeamNG instances")
        print("• Run BeamNG.drive manually first to check it works")
        print("• Check Windows Firewall/antivirus blocking connection")
        print("• Restart computer if BeamNG was recently installed")
        print("• Try running as administrator")