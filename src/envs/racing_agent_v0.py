from network_.sock_ import TcpClient


class RacingAgent_v0:    
    
    def __init__(self):
        # create tcp client
        self.tcp_client = TcpClient(ip="127.0.0.1", port=8080)
        self.commandSet = {
            "Level_Reload": "70001,0",
            "Vehicle_Forward": "61804,0,0",
            "Vehicle_Backward": "61805,0,0",
            "Vehicle_Left": "61806,0,0",
            "Vehicle_Right": "61807,0,0",
            "Vehicle_ReleaseForwadBackward": "61808,0,0",
            "Vehicle_ReleaseLeftRight": "61809,0,0",
            "Vehicle_Handbrake": "61810,0,0",
            "Vehicle_ReleaseHandbrake": "61811,0,0",
            "Timer2D_Reset": "41001,2,0",
            "Spline_GetAllPoints": "62601,0,0",
            
            # response format: n, <x0,y0,z0>, ..., <xn,yn,zn> where
            # n is the number of returned points and maximum allowable
            # n is 4., (0.0,92.0,0.0) vehicle position
            "Spline_GetNearestPoints": "62602,0,0,4,0.0,92.0,0.0",  
            
            "Spline_GetWidth": "62603,0,0",
            "Spline_GetWayPercent": "62604,0,0,60.0,92.0,0.0",  #  : (60.0,92.0,0.0) vehicle position
            "Timer2D_GetTimerTime": "41002,2,0",  # game_screen index: 2, timer2d index: 0
            "Vehicle_GetPosition": "61801,0,0",
            "Vehicle_GetForwardVector": "61802,0,0",
            "Vehicle_HasCollided": "61803,0,0,1,2",  # 1 tag id for rivals and 2 for walls, return 0 or 1
        }
        # state:
        #   - n waypoints in front of the vehicle that must be followed,  
        #   - lidar points
        #   - ?
        self.state_dim = 0 # n_waypoints * waypoint_dim + n_lidar_beams
        self.action_dim = 8

    def close(self):
        pass
        
    def reset(self):
        """resets the environment state
        Returns:
            obs: initial state after env. reset
            {}: a dict
        """
        # send reset command to the racing app
        self.tcp_client.send_data(self.commandSet["Level_Reload"])
        
        # prepare the initial obs after reset
        obs = None
        return obs, {}
    
    def send_test_command(self, command_str):
        response = self.tcp_client.send_data(self.commandSet[command_str])        
        return response
