import time
import socket

def send_message_to_server(message):
    # Define the server address and port
    server_ip = '127.0.0.1'
    server_port = 8080

    # Create a socket object
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            # Connect to the server
            client_socket.connect((server_ip, server_port))
            print(f"Connected to server {server_ip} on port {server_port}")

            # Send the message to the server
            client_socket.sendall(message.encode('utf-8'))
            print(f"Sent message: {message}")

        except ConnectionRefusedError:
            print("Connection failed. Is the server running?")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    while True:
        msg = "Vehicle_ReleaseForwardBackward,0,0"
        send_message_to_server(msg)
        time.sleep(1)        
        
        msg = "Vehicle_Forward,0,0"
        send_message_to_server(msg)        
        time.sleep(1) # sleep for half a seconds
        
        msg = "Vehicle_Left,0,0"
        send_message_to_server(msg)        
        time.sleep(1)
        
        msg = "Vehicle_ReleaseLeftRight,0,0"
        send_message_to_server(msg)
        time.sleep(1)
        
        msg = "Vehicle_Right,0,0"
        send_message_to_server(msg)
        
        msg = "Vehicle_ReleaseLeftRight,0,0"
        send_message_to_server(msg)
        time.sleep(1)
        
        msg = "Vehicle_ReleaseForwardBackward,0,0"
        send_message_to_server(msg)
        time.sleep(1)
        
        msg = "Vehicle_Backward,0,0"
        send_message_to_server(msg)
        time.sleep(1)
