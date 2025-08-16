import socket
import time

def start_client(host='localhost', port=10000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        print(f"Connected to server at {host}:{port}")

        messages = ["Hello, server!", "How are you?", "Goodbye!"]

        for msg in messages:
            print(f"Sending: {msg}")
            sock.sendall(msg.encode())

            data = sock.recv(1024)
            print(f"Received echo: {data.decode()}")

            time.sleep(1)  # Optional delay between messages

        print("Closing client connection")

if __name__ == "__main__":
    start_client()
