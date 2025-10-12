import sys

sys.path.append("I:/projs/game-ai/src")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from network_.sock_ import TcpClient


class RacingAgent_v0:

    def __init__(self, num_rivals, n_nearest_spline_pts):
        """
        Args:
            num_rivals (int): number of rivals that are present in the race
            n_nearest_spline_pts (int): number of nearest spline points to be considered wrt to the
            current position of the agent
        """
        self.fig, self.ax = plt.subplots()

        self.num_rivals = num_rivals
        self.n_nearest_spline_pts = str(n_nearest_spline_pts)
        # create tcp client
        self.tcp_client = TcpClient(ip="127.0.0.1", port=8080)
        self.commandSet = {
            "Level_Reload": "70001",
            # for the bunch of commands below for moving forward, backward, ...
            # args: lvl idx, vehicle idx
            "Vehicle_Forward": "61804",
            "Vehicle_Backward": "61805",
            "Vehicle_Left": "61806",
            "Vehicle_Right": "61807",
            "Vehicle_ReleaseForwadBackward": "61808",
            "Vehicle_ReleaseLeftRight": "61809",
            "Vehicle_Handbrake": "61810",
            "Vehicle_ReleaseHandbrake": "61811",
            # args: game screen index (set it to 2),
            # timer idx (just set it to 0 since there's only one timer)
            "Timer2D_Reset": "41001",
            # args: lvl idx, spline idx
            "Spline_GetAllPoints": "62601",
            # args: lvl idx, spline idx, # of nearest points to be returned, <vehicle xyz position>
            # response format: n, <x0,y0,z0>, ..., <xn,yn,zn> where
            "Spline_GetNearestPoints": "62602",  # ,0,0,4,0.0,92.0,0.0",
            # args: lvl idx, spline idx
            "Spline_GetWidth": "62603",
            # args: lvl idx, spline idx, <xyz vehicle position>
            "Spline_GetWayPercent": "62604",
            # args: index for the game screen which is 2,
            # timer index (just set it to 0 since there's only one timer)
            "Timer2D_GetTimerTime": "41002",
            # args: lvl idx, vehicle idx
            "Vehicle_GetPosition": "61801",
            # args: lvl idx, vehicle idx
            "Vehicle_GetForwardVector": "61802",
            # args: lvl idx, veh idx,
            # <tag id to detect collision with rivals> == 1,
            # <tag id to detect collision with road boundary> == 2
            # returns 0 (no collision) or 1 (collided)
            "Vehicle_HasCollided": "61803",
        }
        # state:
        #   - n waypoints in front of the vehicle that must be followed,
        #   - lidar points
        #   - ?
        self.state_dim = 0  # n_waypoints * waypoint_dim + n_lidar_beams
        self.action_dim = 8

    def close(self):
        self.tcp_client.close()

    def reset(self):
        """resets the environment state
        Returns:
            obs: initial state after env. reset
            {}: a dict
        """
        # send reset command to the racing app
        response = self.tcp_client.send_data(self.commandSet["Level_Reload"])
        print("Resetting level, eng. response: {}".format(response))

        # prepare the initial obs after reset
        obs = self.get_obs()
        return obs, {}

    def vehicle_get_position_command(self, lvl_id, veh_id):
        """
        prepares the command for getting the vehicle position
        Args:
            lvl_id (str): level index
            veh_id (str): index of the vehicle that its 3d position will be returned
        """
        return self.commandSet["Vehicle_GetPosition"] + "," + lvl_id + "," + veh_id

    def vehicle_get_heading_command(self, lvl_id, veh_id):
        """
        prepares the command for getting the vehicle heading vector
        """
        return self.commandSet["Vehicle_GetForwardVector"] + "," + lvl_id + "," + veh_id
    
    def get_nearest_spline_points_command(self, lvl_id, spline_id, _3dpos):
        """_summary_

        Args:
            lvl_id (str): level index
            spline_id (str): spline index
            _3dpos (str): comma-seperated xyz position wrt which nearest spline
            points are determined

        Returns:
            _type_: _description_
        """
        return self.commandSet["Spline_GetNearestPoints"] + "," + lvl_id + "," + spline_id + "," + self.n_nearest_spline_pts + "," + 

    def get_obs(self):
        """
        get the current state of the env.
        """
        # get current pose of the agent & rivals
        agent_pos_str = self.tcp_client.send_data(self.vehicle_get_position_command("0", "0"))
        items = agent_pos_str.split(",")
        agent_position = float(items[0]), float(items[1]), float(items[2])

        rival_positions = np.zeros((self.num_rivals, 3))  # [#rivals x 3]
        for i in range(self.num_rivals):
            resp = self.tcp_client.send_data(
                self.vehicle_get_position_command("0", str(i + 1))
            )
            items = resp.split(",")
            rival_positions[i, :] = float(items[0]), float(items[1]), float(items[2])

        #  get current heading vector for the agent & rivals
        resp = self.tcp_client.send_data(self.vehicle_get_heading_command("0", "0"))
        items = resp.split(",")
        agent_heading = np.array([float(items[0]), float(items[1]), float(items[2])])

        rival_headings = np.zeros((self.num_rivals, 3))  # [#rivals x 3]
        for i in range(self.num_rivals):
            resp = self.tcp_client.send_data(
                self.vehicle_get_heading_command("0", str(i + 1))
            )
            items = resp.split(",")
            rival_headings[i, :] = float(items[0]), float(items[1]), float(items[2])

        # print("Agent position: {}, heading: {}".format(agent_position, agent_heading))
        # for i in range(self.num_rivals):
        #     print(
        #         "Rival {} position: {}, heading: {}".format(
        #             i, rival_positions[i], rival_headings[i]
        #         )
        #     )
        
        # Get n nearest spline points wrt to the agent position
        # resp = self.tcp_client.send_data(self.)
        
        # just for debug
        env.plot_env_obs(agent_position, agent_heading, rival_positions, rival_headings)

    def step(self, action):
        """
        execute the given action in the env and return the
        new env state as the result
        """
        # todo: exec action

        # update the env state
        new_obs = self.get_obs()
        return new_obs

    def plot_env_obs(
        self, agent_position, agent_heading, rival_positions, rival_headings
    ):        
        img_size = np.array([510, 500]) # (w, h)
        image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        radius = 3
        thickness = 2
        # draw agent at the center of the image
        agent_pos = int(img_size[0] / 2), int(img_size[1] / 2)
        cv2.circle(image,
                   agent_pos,  # center
                   radius,
                   (0, 255, 0),  # color
                   thickness)
        # draw agent heading
        agent_head = (agent_heading * 20)[:2] + img_size / 2
        cv2.line(image, agent_pos, 
                 (int(agent_head[0]), int(agent_head[1])),
                 (0, 255, 0), thickness)
        
        # draw rivals wrt to the agent
        for rival_position, rival_heading in zip(rival_positions, rival_headings):
            rival_head = (rival_position + rival_heading * 20 - agent_position)[:2] + img_size / 2
            rival_head = (int(rival_head[0]), int(rival_head[1]))
            rival_a = (rival_position - agent_position)[:2] + img_size / 2            
            rival_a = (int(rival_a[0]), int(rival_a[1]))
            cv2.circle(image, rival_a, radius, (255, 0, 0), thickness)
            cv2.line(image, rival_a, rival_head, (255, 0, 0), thickness)

        # Display the image
        cv2.imshow("Env obs.", image)
        cv2.waitKey(1)

    def send_test_command(self, command_str):
        response = self.tcp_client.send_data(self.commandSet[command_str])
        return response


if __name__ == "__main__":
    env = RacingAgent_v0(num_rivals=3)
    obs, _ = env.reset()
    for i in range(10000):
        env.get_obs()
    env.close()
