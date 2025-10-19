#!/usr/bin/env python3
"""
Test BeamNGpy connection to already running BeamNG instance
"""

import socket
import traceback
from beamngpy import BeamNGpy

def test_basic_connection():
    """Test basic socket connection before BeamNGpy"""
    print("=== Basic Socket Test ===")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        result = sock.connect_ex(('localhost', 64256))
        if result == 0:
            print("✓ Socket can connect to localhost:64256")
            sock.close()
            return True
        else:
            print(f"✗ Socket connection failed with code: {result}")
            return False
    except Exception as e:
        print(f"✗ Socket test failed: {e}")
        return False

def test_beamngpy_connection():
    """Test BeamNGpy connection to existing instance"""
    print("\n=== BeamNGpy Connection Test ===")
    try:
        # Create BeamNGpy instance pointing to BeamNG home but don't launch
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        print(f"Using BeamNG home: {bng_home}")
        
        bng = BeamNGpy('localhost', 64256, home=bng_home, user=None)
        print("✓ BeamNGpy instance created")
        
        print("Attempting to connect to existing BeamNG instance...")
        # Try to connect without launching
        bng.open(launch=False)
        print("✓ Connection successful!")
        
        print("✓ Basic connection established")
        
        print("Closing connection...")
        bng.close()
        print("✓ Connection closed")
        
    except Exception as e:
        print(f"✗ BeamNGpy connection failed: {e}")
        print("\nFull error details:")
        traceback.print_exc()

if __name__ == "__main__":
    print("BeamNGpy Connection Test (Existing Instance)")
    print("=" * 60)
    
    if test_basic_connection():
        test_beamngpy_connection()
    else:
        print("\n⚠️  No server listening on port 64256")
        print("Make sure a BeamNG API server or debug server is running")