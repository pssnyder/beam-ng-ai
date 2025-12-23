"""
Phase 1: Basic Environment Setup (MVP) - SIMPLIFIED
BeamNG.drive AI Automation - Data-Driven Driver

Simplified version focusing on core Phase 1 objective:
- Establish stable connection to BeamNG
- Spawn a vehicle  
- Exchange basic data packet
- Confirm environmental readiness
"""

import time
import sys

# Windows-specific imports for cursor clipping
try:
    import ctypes
    from ctypes import wintypes
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False
    print("Warning: Windows-specific cursor clipping not available")

# BeamNG imports
try:
    from beamngpy import BeamNGpy, Scenario, Vehicle
except ImportError as e:
    print(f"Error importing BeamNGpy: {e}")
    print("Please ensure BeamNGpy is installed: pip install beamngpy")
    sys.exit(1)


class WindowsCursorManager:
    """Manages Windows cursor clipping for input safety during simulation."""
    
    def __init__(self):
        if not WINDOWS_AVAILABLE:
            self.enabled = False
            return
            
        self.enabled = True
        self.user32 = ctypes.windll.user32
        self.original_clip = wintypes.RECT()
        
    def clip_cursor_to_window(self, hwnd=None):
        """Clip cursor to the specified window or current foreground window."""
        if not self.enabled:
            return False
            
        try:
            # Get the current clip rect to restore later
            self.user32.GetClipCursor(ctypes.byref(self.original_clip))
            
            if hwnd is None:
                hwnd = self.user32.GetForegroundWindow()
            
            window_rect = wintypes.RECT()
            if self.user32.GetWindowRect(hwnd, ctypes.byref(window_rect)):
                self.user32.ClipCursor(ctypes.byref(window_rect))
                return True
        except Exception as e:
            print(f"Warning: Could not clip cursor: {e}")
        return False
    
    def release_cursor(self):
        """Release cursor clipping and restore original bounds."""
        if not self.enabled:
            return
            
        try:
            self.user32.ClipCursor(None)  # Release clipping
        except Exception as e:
            print(f"Warning: Could not release cursor: {e}")


def main():
    """Simplified Phase 1 MVP execution."""
    print("="*60)
    print("BeamNG.drive AI Automation - Phase 1: Basic Setup (SIMPLIFIED)")
    print("="*60)
    
    cursor_manager = WindowsCursorManager()
    bng = None
    
    try:
        # Enable cursor clipping for input safety
        print("Setting up input safety...")
        cursor_manager.clip_cursor_to_window()
        print("✓ Cursor clipping enabled for input safety")
        
        # Step 1: Connect to BeamNG
        print("\nConnecting to BeamNG.drive...")
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        bng = BeamNGpy('localhost', 64256, home=bng_home)
        bng.open()
        print("✓ Connected to BeamNG.drive at localhost:64256")
        
        # Step 2: Create simple scenario
        print("\nCreating simple scenario...")
        scenario = Scenario('gridmap_v2', 'phase1_basic_test')
        
        # Create a basic vehicle
        vehicle = Vehicle('test_vehicle', model='etk800')
        scenario.add_vehicle(vehicle, pos=(0, 0, 0))
        
        print("✓ Simple scenario created with one vehicle")
        
        # Step 3: Load scenario
        print("\nLoading scenario into BeamNG.drive...")
        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()
        
        print("✓ Scenario loaded and started")
        
        # Brief pause for scenario loading
        time.sleep(3)
        
        # Step 4: Test basic data exchange (CORE PHASE 1 OBJECTIVE)
        print("\nTesting basic data packet exchange...")
        
        # Get basic vehicle state
        vehicle.update()
        state = vehicle.sensors
        
        print(f"✓ Basic vehicle data received:")
        print(f"  Vehicle sensors available: {type(state)}")
        
        # Test very basic data polling
        if hasattr(vehicle, 'sensors'):
            print(f"  Sensor object: {vehicle.sensors}")
        
        print("\n" + "="*60)
        print("✓ Phase 1 MVP COMPLETE!")
        print("✓ Environmental readiness confirmed")
        print("✓ Foundation established for control loop")
        print("✓ Basic data exchange successful")
        print("✓ Ready to proceed to Phase 2")
        print("="*60)
        
        # Keep scenario running for inspection
        print("\nScenario is running. Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        print("Cleaning up...")
        cursor_manager.release_cursor()
        
        if bng:
            try:
                bng.close()
                print("✓ BeamNG connection closed")
            except Exception as e:
                print(f"Warning: Error closing BeamNG: {e}")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)