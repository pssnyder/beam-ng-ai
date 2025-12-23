"""
BeamNG Connection Troubleshooting Tool
Tests various connection methods and ports
"""

import socket
import time
from beamngpy import BeamNGpy

def test_port_connection(host, port):
    """Test if a port is open and accepting connections."""
    print(f"Testing connection to {host}:{port}...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print(f"✓ Port {port} is open and accepting connections")
            return True
        else:
            print(f"✗ Port {port} is not accessible (error code: {result})")
            return False
            
    except Exception as e:
        print(f"✗ Error testing port {port}: {e}")
        return False

def test_beamng_connection(host, port, bng_home):
    """Test BeamNGpy connection."""
    print(f"\nTesting BeamNGpy connection to {host}:{port}...")
    
    try:
        bng = BeamNGpy(host, port, home=bng_home)
        print("✓ BeamNGpy instance created")
        
        # Try to connect
        bng.open()
        print("✓ BeamNG connection successful!")
        
        bng.close()
        return True
        
    except Exception as e:
        print(f"✗ BeamNG connection failed: {e}")
        return False

def main():
    print("="*60)
    print("BeamNG Connection Troubleshoot Tool")
    print("="*60)
    
    host = 'localhost'
    bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
    
    # Test common ports
    common_ports = [64256, 64255, 64257, 4444, 8080]
    
    print("Testing common BeamNG ports...")
    open_ports = []
    
    for port in common_ports:
        if test_port_connection(host, port):
            open_ports.append(port)
    
    print(f"\nOpen ports found: {open_ports}")
    
    if not open_ports:
        print("\n⚠ No open ports found. BeamNG.drive may not be running with API enabled.")
        print("\nTroubleshooting steps:")
        print("1. Make sure BeamNG.drive is running")
        print("2. Open console in BeamNG (~ key)")
        print("3. Try commands like 'connect' or 'api start'")
        print("4. Check BeamNG settings for Research/API mode")
        return False
    
    # Test BeamNGpy connection on open ports
    print(f"\nTesting BeamNGpy connections...")
    for port in open_ports:
        if test_beamng_connection(host, port, bng_home):
            print(f"\n✓ SUCCESS! Use port {port} for BeamNG connections.")
            return True
    
    print("\n⚠ Port tests passed but BeamNGpy connection failed.")
    print("BeamNG may be running but API not enabled.")
    
    return False

if __name__ == "__main__":
    main()