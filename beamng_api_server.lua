-- BeamNG Lua script to start API server for BeamNGpy
-- This should be run in the BeamNG console

local socket = require("socket")
local json = require("json")

-- API Server configuration
local API_HOST = "localhost"
local API_PORT = 64256

-- Create the server socket
local server = socket.tcp()
server:bind(API_HOST, API_PORT)
server:listen(1)

print("BeamNG API Server started on " .. API_HOST .. ":" .. API_PORT)

-- Simple message handling
local function handle_client(client)
    print("Client connected")
    
    while true do
        -- Receive length prefix (4 bytes, little endian)
        local length_data = client:receive(4)
        if not length_data then
            print("Client disconnected")
            break
        end
        
        -- Unpack length (assuming little endian)
        local length = string.unpack("<I4", length_data)
        print("Message length: " .. length)
        
        -- Receive the actual message
        local message = client:receive(length)
        if not message then
            print("Failed to receive message")
            break
        end
        
        print("Received: " .. message)
        
        -- Parse JSON message
        local parsed = json.decode(message)
        print("Parsed message type: " .. (parsed.type or "unknown"))
        
        -- Simple response based on message type
        local response = {}
        
        if parsed.type == "Hello" then
            response = {
                type = "Hello",
                data = {
                    version = "1.0",
                    status = "OK"
                }
            }
        else
            response = {
                type = "Response",
                data = {
                    status = "OK",
                    message = "Command received"
                }
            }
        end
        
        -- Send response
        local response_json = json.encode(response)
        local response_length = string.pack("<I4", #response_json)
        
        client:send(response_length)
        client:send(response_json)
        
        print("Sent response: " .. response_json)
    end
    
    client:close()
end

-- Main server loop
while true do
    local client = server:accept()
    if client then
        handle_client(client)
    end
end