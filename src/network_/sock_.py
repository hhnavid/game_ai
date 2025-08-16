import socket
import time
import numpy as np
import random


class TcpClient:
    
    def __init__(self, ip='127.0.0.1', port=8080):
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Define the server address and port to connect to
        self.server_address = (ip, port)        
        print('Connecting to {} port {}'.format(self.server_address[0], self.server_address[1]))

        # Connect to the server
        try:
            self.sock.connect(self.server_address)
            print("Socket connection OK.")
        except socket.error as e:
            self.sock = None
            print("Socket connection failed: {}".format(e))    
    
    def send_data(self, msg):                        
        """sends msg to tcp server
        Args:
            msg (str): message to be sent to the server
        """                 
        if self.sock is None:
            print("Socket is not connected!")                  
                
        # Send data        
        print('sending "{}"'.format(msg))
        self.sock.sendall(msg.encode('utf-8'))
        response = 0
        
        # Receive response        
        response = self.sock.recv(4096)
        print('Received response from server: {}\n'.format(response.decode('utf-8')))        
                
    def close(self):        
        print('closing socket...')
        self.sock.close()
        
    def is_socket_open(self):
        try:
            # This checks for socket errors without blocking
            err = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            print("status in is_socket_open: {}".format(err))
            return err == 0
        except socket.error:
            print("status in is_socket_open: {}".format(err))
            return False


if __name__ == "__main__":
    tcp_client = TcpClient()        
    
    commands = np.array([        
        "Vehicle_Forward,0,0", # ok
        "Vehicle_Backward,0,0", # ok
        "Vehicle_Left,0,0", # ok
        "Vehicle_Right,0,0", # ok
        "Vehicle_ReleaseForwardBackward,0,0", # failed
        "Vehicle_ReleaseLeftRight,0,0", # failed
        "Vehicle_Handbrake,0,0", # failed
        "Vehicle_ReleaseHandbrake,0,0" # failed
        ])
    
    while True:
        # execute a random command
        i = random.randint(0, commands.size-1)    
        print("Selected command index: {}".format(i))
        tcp_client.send_data(commands[i])
        time.sleep(0.25)                 
                                             
    tcp_client.close()