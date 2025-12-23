#!/usr/bin/env python3
"""
Debug script to understand BeamNGpy protocol by capturing the exact messages
"""

import socket
import struct
import traceback
from beamngpy import BeamNGpy

def create_debug_server():
    """Create a server that logs all incoming data to understand the protocol"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('localhost', 64256))
        server_socket.listen(1)
        print("✓ Debug server listening on localhost:64256")
        
        while True:
            print("\nWaiting for connection...")
            client_socket, address = server_socket.accept()
            print(f"✓ Connection from {address}")
            
            try:
                # Read all data sent by BeamNGpy
                data = client_socket.recv(4096)
                print(f"Raw data received ({len(data)} bytes):")
                print(f"Hex: {data.hex()}")
                print(f"ASCII: {data}")
                
                # Try to understand the structure
                if len(data) >= 4:
                    # Check if first 4 bytes are a length prefix
                    length_prefix = struct.unpack('<I', data[:4])[0]
                    print(f"Potential length prefix: {length_prefix}")
                    
                    if len(data) > 4:
                        message_data = data[4:]
                        print(f"Message data: {message_data}")
                        try:
                            decoded = message_data.decode('utf-8')
                            print(f"Decoded message: {decoded}")
                        except:
                            print("Could not decode as UTF-8")
                
                # Send a simple response
                response = b"OK"
                length = struct.pack('<I', len(response))
                client_socket.send(length + response)
                print(f"Sent response: {length.hex()} + {response}")
                
            except Exception as e:
                print(f"Error handling client: {e}")
                traceback.print_exc()
            finally:
                client_socket.close()
                print("Client disconnected")
                
    except KeyboardInterrupt:
        print("\nShutting down debug server...")
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
    finally:
        server_socket.close()

if __name__ == "__main__":
    create_debug_server()