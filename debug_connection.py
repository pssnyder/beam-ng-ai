#!/usr/bin/env python3
"""
Simple test to see if BeamNGpy can connect at all, with more detailed error info
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
        else:
            print(f"✗ Socket connection failed with code: {result}")
    except Exception as e:
        print(f"✗ Socket test failed: {e}")

def test_beamngpy_connection():
    """Test BeamNGpy connection with detailed error reporting"""
    print("\n=== BeamNGpy Connection Test ===")
    try:
        # Create BeamNGpy instance with detailed logging
        print("Creating BeamNGpy instance...")
        bng = BeamNGpy('localhost', 64256)
        print("✓ BeamNGpy instance created")
        
        print("Attempting to connect...")
        bng.open()
        print("✓ Connection successful!")
        
        print("Closing connection...")
        bng.close()
        print("✓ Connection closed")
        
    except Exception as e:
        print(f"✗ BeamNGpy connection failed: {e}")
        print("Full error details:")
        traceback.print_exc()

if __name__ == "__main__":
    print("BeamNGpy Connection Debug Test")
    print("=" * 50)
    
    test_basic_connection()
    test_beamngpy_connection()