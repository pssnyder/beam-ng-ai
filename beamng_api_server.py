"""
BeamNG API Server Emulator
Creates a server that can handle BeamNGpy connections properly
"""

import socket
import struct
import json
import threading
import time

class BeamNGAPIServer:
    def __init__(self, host='localhost', port=64256):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
    def send_prefixed_message(self, client_socket, message):
        """Send a length-prefixed message like BeamNGpy expects."""
        if isinstance(message, dict):
            message = json.dumps(message)
        if isinstance(message, str):
            message = message.encode('utf-8')
        
        # Send length prefix (4 bytes, big endian)
        length = len(message)
        length_prefix = struct.pack('>I', length)
        client_socket.send(length_prefix + message)
        print(f"Sent: {message.decode('utf-8')[:100]}...")
    
    def receive_prefixed_message(self, client_socket):
        """Receive a length-prefixed message."""
        # Receive length prefix (4 bytes)
        length_data = client_socket.recv(4)
        if len(length_data) < 4:
            return None
        
        length = struct.unpack('>I', length_data)[0]
        
        # Receive the actual message
        message = b''
        while len(message) < length:
            chunk = client_socket.recv(length - len(message))
            if not chunk:
                return None
            message += chunk
        
        try:
            decoded = message.decode('utf-8')
            print(f"Received: {decoded[:100]}...")
            return decoded
        except:
            return message.hex()
    
    def handle_client(self, client_socket, client_address):
        """Handle a BeamNGpy client connection."""
        print(f"Client connected from {client_address}")
        
        try:
            while self.running:
                # Receive message from client
                message = self.receive_prefixed_message(client_socket)
                if not message:
                    break
                
                print(f"Processing message: {message}")
                
                # Parse the message if it's JSON
                try:
                    data = json.loads(message)
                    response = self.process_beamng_message(data)
                except json.JSONDecodeError:
                    response = {"type": "error", "message": "Invalid JSON"}
                
                # Send response
                self.send_prefixed_message(client_socket, response)
                
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
            print(f"Client {client_address} disconnected")
    
    def process_beamng_message(self, data):
        """Process BeamNGpy protocol messages."""
        print(f"Processing BeamNG message: {data}")
        
        # Handle common BeamNGpy messages
        if data.get("type") == "Hello":
            return {
                "type": "Hello",
                "message": "BeamNG API Server Ready",
                "version": "1.0.0"
            }
        elif data.get("type") == "GetVersion":
            return {
                "type": "Version", 
                "version": "0.37.6.0"
            }
        elif data.get("type") == "GetStatus":
            return {
                "type": "Status",
                "status": "Ready"
            }
        else:
            return {
                "type": "Acknowledge",
                "message": f"Received: {data.get('type', 'Unknown')}"
            }
    
    def start(self):
        """Start the API server."""
        print(f"Starting BeamNG API Server on {self.host}:{self.port}")
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"✓ Server listening on {self.host}:{self.port}")
            print("✓ Ready for BeamNGpy connections!")
            
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client, 
                        args=(client_socket, client_address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")
                        
        except Exception as e:
            print(f"Failed to start server: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the API server."""
        print("Stopping BeamNG API Server...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()

def main():
    print("="*60)
    print("BeamNG API Server Emulator")
    print("="*60)
    print("This creates a server that BeamNGpy can connect to for testing.")
    print("Press Ctrl+C to stop the server.")
    print()
    
    server = BeamNGAPIServer()
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()

if __name__ == "__main__":
    main()