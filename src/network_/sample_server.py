import socket

def start_server(host='localhost', port=10000):
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Connection from {addr} established")

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                print("Client closed connection")
                break
            print(f"Received: {data.decode()}")
            conn.sendall(data)  # Echo back the received data
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        conn.close()
        server_socket.close()
        print("Server closed")

if __name__ == "__main__":
    start_server()
