#!/usr/bin/env python3
"""
Script to launch BeamNG with research mode enabled and test connection
"""

import subprocess
import time
import socket
from beamngpy import BeamNGpy

def check_port_open(host, port, timeout=5):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def launch_beamng_research():
    """Try to launch BeamNG with research mode"""
    beamng_path = "S:/SteamLibrary/steamapps/common/BeamNG.drive/Bin64/BeamNG.research.x64.exe"
    
    print("Attempting to launch BeamNG in research mode...")
    try:
        # Try different possible executables and arguments
        possible_commands = [
            [beamng_path],
            ["S:/SteamLibrary/steamapps/common/BeamNG.drive/BeamNG.drive.exe", "-research"],
            ["S:/SteamLibrary/steamapps/common/BeamNG.drive/BeamNG.drive.exe", "-luadebug", "-research"],
            ["S:/SteamLibrary/steamapps/common/BeamNG.drive/Bin64/BeamNG.drive.x64.exe", "-research"],
        ]
        
        for cmd in possible_commands:
            print(f"Trying: {' '.join(cmd)}")
            try:
                process = subprocess.Popen(cmd, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
                
                # Wait a bit to see if it starts
                time.sleep(3)
                
                if process.poll() is None:  # Still running
                    print(f"✓ BeamNG started with command: {' '.join(cmd)}")
                    
                    # Wait for the API port to become available
                    print("Waiting for API port to become available...")
                    for i in range(30):  # Wait up to 30 seconds
                        if check_port_open('localhost', 64256):
                            print("✓ API port 64256 is now available!")
                            return process
                        time.sleep(1)
                        print(f"  Waiting... ({i+1}/30)")
                    
                    print("⚠️  API port did not become available")
                    return process
                    
            except FileNotFoundError:
                print(f"✗ File not found: {cmd[0]}")
                continue
            except Exception as e:
                print(f"✗ Error with command {' '.join(cmd)}: {e}")
                continue
        
        print("✗ Could not start BeamNG in research mode")
        return None
        
    except Exception as e:
        print(f"✗ Error launching BeamNG: {e}")
        return None

def test_beamngpy_connection():
    """Test BeamNGpy connection"""
    print("\n=== Testing BeamNGpy Connection ===")
    
    try:
        bng = BeamNGpy('localhost', 64256, 
                      home="S:/SteamLibrary/steamapps/common/BeamNG.drive")
        print("✓ BeamNGpy instance created")
        
        print("Attempting to connect...")
        bng.open(launch=False)  # Don't launch, connect to existing
        print("✓ Connected to BeamNG!")
        
        # Test basic functionality
        print("Testing basic commands...")
        # Add any basic tests here
        
        bng.close()
        print("✓ Connection closed")
        
        return True
        
    except Exception as e:
        print(f"✗ BeamNGpy connection failed: {e}")
        return False

if __name__ == "__main__":
    print("BeamNG Research Mode Launcher and Tester")
    print("=" * 50)
    
    # Check if port is already open
    if check_port_open('localhost', 64256):
        print("✓ Port 64256 is already open, testing connection...")
        test_beamngpy_connection()
    else:
        print("Port 64256 is not open, attempting to launch BeamNG...")
        process = launch_beamng_research()
        
        if process:
            try:
                if check_port_open('localhost', 64256):
                    test_beamngpy_connection()
                else:
                    print("⚠️  BeamNG started but API port is not available")
                    print("You may need to enable research mode manually in BeamNG")
                    
                input("Press Enter to close BeamNG...")
                process.terminate()
                
            except KeyboardInterrupt:
                print("\nShutting down...")
                if process:
                    process.terminate()
        else:
            print("\nCould not start BeamNG. Try starting it manually with:")
            print("1. Launch BeamNG normally")  
            print("2. Open console with ~ key")
            print("3. Run the Lua script: beamng_api_server.lua")