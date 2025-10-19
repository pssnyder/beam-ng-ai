"""
Simple BeamNGpy Connection Test
Tests if our API server can handle a basic BeamNGpy connection
"""

from beamngpy import BeamNGpy
import sys

def test_connection():
    print("="*50)
    print("Testing BeamNGpy Connection to API Server")
    print("="*50)
    
    try:
        # Try to connect to our API server on port 64256
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        bng = BeamNGpy('localhost', 64256, home=bng_home)
        
        print("✓ BeamNGpy instance created")
        print("Attempting to connect...")
        
        # Try to open connection
        bng.open()
        print("✓ Connection successful!")
        
        # Try to close
        bng.close()
        print("✓ Connection closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)