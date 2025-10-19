#!/usr/bin/env python3
"""
Test BeamNGpy connection using the correct default port 25252
"""

import socket
import traceback
from beamngpy import BeamNGpy

def test_connection_25252():
    """Test connection on correct BeamNGpy port 25252"""
    print("=== Testing BeamNGpy Connection on Port 25252 ===")
    
    # Test basic socket connection first
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 25252))
        sock.close()
        if result == 0:
            print("✓ Socket can connect to localhost:25252")
        else:
            print(f"✗ Socket connection failed to port 25252 with code: {result}")
    except Exception as e:
        print(f"✗ Socket test failed: {e}")
    
    # Test BeamNGy connection
    try:
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        print(f"Using BeamNG home: {bng_home}")
        
        # Use the correct default port 25252
        bng = BeamNGpy('localhost', 25252, home=bng_home)
        print("✓ BeamNGpy instance created")
        
        print("Attempting to connect to existing BeamNG instance...")
        bng.open(launch=False)  # Don't launch, connect to existing
        print("✓ Connected successfully!")
        
        print("Closing connection...")
        bng.close()
        print("✓ Connection closed")
        
        return True
        
    except Exception as e:
        print(f"✗ BeamNGpy connection failed: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        return False

def test_connection_64256():
    """Test connection on old BeamNGpy port 64256"""
    print("\n=== Testing BeamNGpy Connection on Port 64256 (Legacy) ===")
    
    # Test basic socket connection first
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 64256))
        sock.close()
        if result == 0:
            print("✓ Socket can connect to localhost:64256")
        else:
            print(f"✗ Socket connection failed to port 64256 with code: {result}")
    except Exception as e:
        print(f"✗ Socket test failed: {e}")
    
    # Test BeamNGy connection
    try:
        bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
        print(f"Using BeamNG home: {bng_home}")
        
        # Use the old port 64256
        bng = BeamNGpy('localhost', 64256, home=bng_home)
        print("✓ BeamNGpy instance created")
        
        print("Attempting to connect to existing BeamNG instance...")
        bng.open(launch=False)  # Don't launch, connect to existing
        print("✓ Connected successfully!")
        
        print("Closing connection...")
        bng.close()
        print("✓ Connection closed")
        
        return True
        
    except Exception as e:
        print(f"✗ BeamNGpy connection failed: {e}")
        print("\nFull error details:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("BeamNGpy Connection Test - Multiple Ports")
    print("=" * 60)
    print("Note: BeamNGpy v1.31+ uses port 25252 by default (changed from 64256)")
    print()
    
    success_25252 = test_connection_25252()
    success_64256 = test_connection_64256()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Port 25252 (current default): {'✓ Success' if success_25252 else '✗ Failed'}")
    print(f"Port 64256 (legacy default):  {'✓ Success' if success_64256 else '✗ Failed'}")
    
    if not success_25252 and not success_64256:
        print("\n⚠️  No BeamNG API server is running on either port.")
        print("Recommendations:")
        print("1. Start BeamNG manually")
        print("2. Try launching BeamNG with: bng.open(launch=True)")
        print("3. Or start BeamNG with research mode arguments:")
        print("   BeamNG.drive.exe -tcom -tport 25252")