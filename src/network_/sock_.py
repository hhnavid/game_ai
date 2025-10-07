import socket
import time
import numpy as np
import random


class TcpClient:

    def __init__(self, ip="127.0.0.1", port=8080):
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Define the server address and port to connect to
        self.server_address = (ip, port)
        print(
            "Connecting to {} port {}".format(
                self.server_address[0], self.server_address[1]
            )
        )

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
        self.sock.sendall(msg.encode("utf-8"))
        response = 0

        # Receive response
        response = self.sock.recv(4096)
        # print("Received response from server: {}\n".format(response.decode("utf-8")))
        return response

    def close(self):
        print("closing socket...")
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

    commands = {        
            "Level_Reload": "70001,0", 
            "Vehicle_Forward": "61804,0,0",
            "Vehicle_Backward": "61805,0,0",
            "Vehicle_Left": "61806,0,0",
            "Vehicle_Right":"61807,0,0", 
            "Vehicle_ReleaseForwadBackward": "61808,0,0",  
            "Vehicle_ReleaseLeftRight": "61809,0,0",  
            # "Vehicle_Handbrake": "61810,0,0",  
            "Vehicle_ReleaseHandbrake": "61811,0,0",  
            # "Timer2D_Reset": "41001,2,0",  
            # "Spline_GetAllPoints": "62601,0,0",  
            # "Spline_GetNearestPoints": "62602,0,0,4,0.0,92.0,0.0",  # max points 4, (0.0,92.0,0.0) vehicle position
            # "Spline_GetWidth": "62603,0,0",
            # "Spline_GetWayPercent": "62604,0,0,60.0,92.0,0.0",  #  : (60.0,92.0,0.0) vehicle position
            # "Timer2D_GetTimerTime": "41002,2,0",  # game_screen index: 2, timer2d index: 0
            # "Vehicle_GetPosition": "61801,0,0",
            # "Vehicle_GetForwardVector": "61802,0,0",
            # "Vehicle_HasCollided": "61803,0,0,1,2",  # 1 tag id for rivals and 2 for walls, return 0 or 1
        }

    while True:
        # execute a random command
        # i = random.randint(0, len(commands) - 1)
        randomKey = random.choice(list(commands.keys()))
        print("Selected command index: {}".format(randomKey))
        tcp_client.send_data(commands[randomKey])
        time.sleep(0.25)

    tcp_client.close()
