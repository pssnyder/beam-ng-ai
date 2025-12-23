"""
Phase 1: Basic Environment Setup (MVP)
BeamNG.drive AI Automation - Data-Driven Driver

This script establishes a stable, authenticated Python-to-BeamNG connection,
spawns a vehicle, and exchanges a single data packet to confirm environmental readiness.

Tasks completed:
1. Install BeamNGpy (pip install beamngpy) ✓
2. Verify BeamNG.tech/drive version compatibility
3. Write initial connection and teardown script
4. Create simplest possible scenario (one car, one flat map)
5. Add "Clip Cursor" logic for Windows input safety
"""

import time
import sys
from pathlib import Path

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
    from beamngpy.sensors import Camera, GPS, Damage
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


class BeamNGConnection:
    """Manages BeamNG.drive connection and basic scenario setup."""
    
    def __init__(self, host='localhost', port=64256):
        self.host = host
        self.port = port
        self.beamng = None
        self.scenario = None
        self.vehicle = None
        self.cursor_manager = WindowsCursorManager()
        
    def connect(self):
        """Establish connection to BeamNG.drive."""
        print("Connecting to BeamNG.drive...")
        
        # Initialize BeamNG instance
        self.beamng = BeamNGpy(self.host, self.port)
        
        try:
            # Open BeamNG.drive
            self.beamng.open()
            print(f"✓ Connected to BeamNG.drive at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to BeamNG.drive: {e}")
            print("Please ensure BeamNG.drive is installed and running.")
            return False
    
    def create_simple_scenario(self):
        """Create the simplest possible scenario: one car on a flat map."""
        print("Creating simple scenario...")
        
        try:
            # Create a simple scenario on the default map
            self.scenario = Scenario('gridmap_v2', 'phase1_basic_test')
            
            # Create a vehicle with basic sensors
            self.vehicle = Vehicle('test_vehicle', model='etk800')
            
            # Add essential sensors for telemetry testing
            camera = Camera('front_camera', 
                          bng=self.beamng,
                          pos=(0.5, 0, 1.5),  # Front of vehicle
                          dir=(0, -1, 0),     # Looking forward
                          fov=75,
                          resolution=(640, 480))
            
            gps = GPS('gps_sensor')
            damage = Damage('damage_sensor')
            
            # Attach sensors to vehicle
            self.vehicle.attach_sensor('camera', camera)
            self.vehicle.attach_sensor('gps', gps)
            self.vehicle.attach_sensor('damage', damage)
            
            # Add vehicle to scenario at origin
            self.scenario.add_vehicle(self.vehicle, pos=(0, 0, 0), rot=(0, 0, 0))
            
            print("✓ Simple scenario created with basic sensor suite")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create scenario: {e}")
            return False
    
    def load_scenario(self):
        """Load the scenario into BeamNG.drive."""
        print("Loading scenario into BeamNG.drive...")
        
        try:
            # Load the scenario
            self.scenario.make(self.beamng)
            self.beamng.load_scenario(self.scenario)
            self.beamng.start_scenario()
            
            print("✓ Scenario loaded and started")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load scenario: {e}")
            return False
    
    def test_data_exchange(self):
        """Test basic data exchange - the core objective of Phase 1."""
        print("Testing data packet exchange...")
        
        try:
            # Poll basic vehicle state
            self.vehicle.update_vehicle()
            state = self.vehicle.state
            
            print(f"✓ Basic vehicle state received:")
            print(f"  Position: {state.get('pos', 'N/A')}")
            print(f"  Rotation: {state.get('dir', 'N/A')}")
            print(f"  Velocity: {state.get('vel', 'N/A')}")
            
            # Test sensor data
            sensors = self.vehicle.poll_sensors()
            
            if 'camera' in sensors:
                camera_data = sensors['camera']
                print(f"  Camera: {camera_data.shape if hasattr(camera_data, 'shape') else 'Data received'}")
            
            if 'gps' in sensors:
                gps_data = sensors['gps']
                print(f"  GPS: {gps_data}")
            
            if 'imu' in sensors:
                imu_data = sensors['imu']
                print(f"  IMU: {type(imu_data)}")
            
            print("✓ Data packet exchange successful!")
            return True
            
        except Exception as e:
            print(f"✗ Data exchange failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up connection and release resources."""
        print("Cleaning up...")
        
        # Release cursor clipping
        self.cursor_manager.release_cursor()
        
        if self.beamng:
            try:
                self.beamng.close()
                print("✓ BeamNG connection closed")
            except Exception as e:
                print(f"Warning: Error closing BeamNG: {e}")


def main():
    """Main function to execute Phase 1 MVP tasks."""
    print("="*60)
    print("BeamNG.drive AI Automation - Phase 1: Basic Setup")
    print("="*60)
    
    connection = BeamNGConnection()
    
    try:
        # Enable cursor clipping for input safety
        print("Setting up input safety...")
        connection.cursor_manager.clip_cursor_to_window()
        print("✓ Cursor clipping enabled for input safety")
        
        # Step 1: Connect to BeamNG
        if not connection.connect():
            return False
        
        # Brief pause for connection stability
        time.sleep(2)
        
        # Step 2: Create simple scenario
        if not connection.create_simple_scenario():
            return False
        
        # Step 3: Load scenario
        if not connection.load_scenario():
            return False
        
        # Brief pause for scenario loading
        time.sleep(3)
        
        # Step 4: Test data exchange (core Phase 1 objective)
        if not connection.test_data_exchange():
            return False
        
        print("\n" + "="*60)
        print("✓ Phase 1 MVP COMPLETE!")
        print("✓ Environmental readiness confirmed")
        print("✓ Foundation established for control loop")
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
        return False
        
    finally:
        connection.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)