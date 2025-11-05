import sys

sys.path.append("I:/projs/game-ai/src")

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from network_.sock_ import TcpClient
from common.tf import rotate_point_around_axis

deg2rad = 0.017453292519943295
rad2deg = 57.2957795130823208


class RacingAgent_v0:

    def __init__(
        self,
        num_rivals,
        n_nearest_spline_pts,
        lidar_max_range,
        lidar_start_angle=-120,
        lidar_stop_angle=120,
        lidar_res=10,
    ):
        """
        Args:
            num_rivals (int): number of rivals that are present in the race
            n_nearest_spline_pts (int): number of nearest spline points to be considered wrt to the
            current position of the agent
            lidar_max_range (float): specifies the max range of rays emitted for ray tracing
                                     (i.e. obstacle detection). Beyond this value no obstacle
                                     is detected.
            lidar_start_angle (float, deg): the angle of the 1st ray
            lidar_stop_angle (float, deg): the angle of the last ray
            beam_resolution (float, deg): the angle between two successive rays which specifies
            the total number of rays that must emitted
        """
        self.start_time = time.time()
        self.end_time = self.start_time

        self.fig, self.ax = plt.subplots()

        self.num_rivals = num_rivals
        self.n_nearest_spline_pts = str(n_nearest_spline_pts)
        # create tcp client
        self.tcp_client = TcpClient(ip="127.0.0.1", port=8080)
        self.command_set = {
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
            # response format: n, <x0,y0,z0>, ..., <xn,yn,zn> where <xi,yi,zi> is the
            # ith nearest spline point
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
            # args: <xyz position from which the ray is emitted>,
            # <xyz normal direction of the ray>,
            # <float: max range of the ray beyond which obstacle detection is not performed>
            "EmitRay": "11101",
            # args lvl idx, veh idx
            # resp: max speed, current speed
            "Vehicle_GetSpeed": "61812",
        }

        self.lidar_max_range = lidar_max_range
        self.lidar_start_angle = lidar_start_angle * deg2rad
        self.lidar_stop_angle = lidar_stop_angle * deg2rad
        self.lidar_res = lidar_res
        # state:
        #   - agent's 3d position
        #   - agent's 3d heading
        #   - next waypoint (3d)
        #   - range data
        n_beams = (
            int((self.lidar_stop_angle - self.lidar_start_angle) / self.lidar_res) + 1
        )
        self.state_dim = 3 + 3 + 3 + n_beams
        self.action_dim = 8
        self.action_set = np.array(
            [
                self.veh_forward_cmd,
                self.veh_backward_cmd,
                self.veh_left_cmd,
                self.veh_right_cmd,
                self.veh_release_forwardbackward_cmd,
                self.veh_release_leftright_cmd,
                self.veh_handbrake_cmd,
                self.veh_release_handbrake_cmd,
            ]
        )
        self.step_count = 0
        self.max_steps = 40000

    def close(self):
        self.tcp_client.close()

    def reset(self):
        """resets the environment state
        Returns:
            obs: initial state after env. reset
            {}: a dict
        """
        # send reset command to the racing app
        response = self.tcp_client.send_data(self.command_set["Level_Reload"])
        print("Resetting level, eng. response: {}".format(response))
        self.step_count = 0

        # prepare the initial obs after reset
        obs = self.get_obs()
        return obs

    def veh_forward_cmd(self, lvl_id, veh_id):
        return self.command_set["Vehicle_Forward"] + "," + lvl_id + "," + veh_id

    def veh_backward_cmd(self, lvl_id, veh_id):
        return self.command_set["Vehicle_Backward"] + "," + lvl_id + "," + veh_id

    def veh_left_cmd(self, lvl_id, veh_id):
        return self.command_set["Vehicle_Left"] + "," + lvl_id + "," + veh_id

    def veh_right_cmd(self, lvl_id, veh_id):
        return self.command_set["Vehicle_Right"] + "," + lvl_id + "," + veh_id

    def veh_release_forwardbackward_cmd(self, lvl_id, veh_id):
        return (
            self.command_set["Vehicle_ReleaseForwadBackward"]
            + ","
            + lvl_id
            + ","
            + veh_id
        )

    def veh_release_leftright_cmd(self, lvl_id, veh_id):
        return (
            self.command_set["Vehicle_ReleaseLeftRight"] + "," + lvl_id + "," + veh_id
        )

    def veh_handbrake_cmd(self, lvl_id, veh_id):
        return self.command_set["Vehicle_Handbrake"] + "," + lvl_id + "," + veh_id

    def veh_release_handbrake_cmd(self, lvl_id, veh_id):
        return (
            self.command_set["Vehicle_ReleaseHandbrake"] + "," + lvl_id + "," + veh_id
        )

    def veh_get_position_cmd(self, lvl_id, veh_id):
        """
        prepares the command for getting the vehicle position
        Args:
            lvl_id (str): level index
            veh_id (str): index of the vehicle that its 3d position will be returned
        """
        return self.command_set["Vehicle_GetPosition"] + "," + lvl_id + "," + veh_id

    def veh_get_heading_cmd(self, lvl_id, veh_id):
        """
        prepares the command for getting the vehicle heading vector
        """
        return (
            self.command_set["Vehicle_GetForwardVector"] + "," + lvl_id + "," + veh_id
        )

    def get_nearest_spline_points_cmd(self, lvl_id, spline_id, _3dpos):
        """
        prepares the command for getting the n nearest spline points to
        the given 3d position
        Args:
            lvl_id (str): level index
            spline_id (str): spline index
            _3dpos (str): comma-seperated xyz position wrt which nearest spline
            points are determined
        """
        return (
            self.command_set["Spline_GetNearestPoints"]
            + ","
            + lvl_id
            + ","
            + spline_id
            + ","
            + self.n_nearest_spline_pts
            + ","
            + _3dpos
        )

    def get_lap_progress_cmd(self, lvl_id, spline_id, veh_pos):
        """
        Args:
            lvl_id (str): level index
            spline_id (str): bSpline index
            veh_pos (str): x,y,z position of the vehicle for which
                           lap progress is computed
        Returns:
            progress (float, in range [0,1]): percentage of the lap that has been
                              completed by the vehicle
        """
        return (
            self.command_set["Spline_GetWayPercent"]
            + ","
            + lvl_id
            + ","
            + spline_id
            + ","
            + veh_pos
        )

    def get_ray_trace_cmd(self, origin, direction):
        """
        Detect nearest obstacle (if any) along the given direction
        Args:
            origin (str): x,y,z position from which the ray is emitted
            direction (str): x,y,z direction of the ray
            range_ (str): max range (float number) of the ray beyond which obstacle are ignored
        """
        command = (
            self.command_set["EmitRay"]
            + ","
            + origin
            + ","
            + direction
            + ","
            + str(self.lidar_max_range)
        )
        return command

    def get_range_data(self, ray_origin, zero_heading):
        """
        Get range to obstacle (if any) with the specified angle resolution and range
        Args:
            ray_origin (np array): the 3d position at which ray is emitted
            zero_heading (np array): the beam direction at angle zero. This is the same as the
            vehicle heading vector expressed in vehicle's own coord frame
        Returns
            ray_angles_array (np array): the array of angles (radian) at which a ray is emitted
            range_array (np array): the array of range values corresponding to the emitted rays
            collision_array (np array): the array of 3d collision points (if any) corresponding to the emitted rays
        """        
        ray_angles_array = []
        range_array = []
        collision_array = []
        rot_axis = np.array([0.0, 0.0, 1.0])  # z axis

        for ang in np.arange(
            self.lidar_start_angle,
            self.lidar_stop_angle + self.lidar_res,
            self.lidar_res,
        ):
            ray_angles_array.append(ang)
            ray_dir = rotate_point_around_axis(zero_heading, rot_axis, ang)

            # prepare ray tracing command
            ray_trace_cmd = self.get_ray_trace_cmd(
                "{:.3f},{:.3f},{:.3f}".format(
                    ray_origin[0], ray_origin[1], ray_origin[2]
                ),
                "{:.3f},{:.3f},{:.3f}".format(ray_dir[0], ray_dir[1], ray_dir[2]),
            )
            resp_str = self.tcp_client.send_data(ray_trace_cmd)            
            items = resp_str.split(",")
            had_collided, group_id, ray_endpoint = (
                int(items[0]),
                int(items[1]),
                items[2:],
            )

            # just for debug
            agent_pos_str = self.tcp_client.send_data(
                self.veh_get_position_cmd("0", "0")
            )

            if had_collided:
                ray_endpoint = np.array(
                    [
                        float(ray_endpoint[0]),
                        float(ray_endpoint[1]),
                        float(ray_endpoint[2]),
                    ],
                )
                range_ = np.linalg.norm(ray_endpoint - ray_origin)
            else:
                # added 1 unit so that free space within the lidar
                # range will be distinguishable from occupied space
                range_ = self.lidar_max_range + 1
                ray_endpoint = ray_origin + ray_dir * range_  # no collision has occured

            range_array.append(range_)
            collision_array.append(ray_endpoint)

        ray_angles_array = np.array(ray_angles_array)
        range_array = np.array(range_array)
        collision_array = np.array(collision_array)

        return ray_angles_array, range_array, collision_array

    def get_obs(self):
        """
        get the current state of the env.
        Returns:
            state (nparray): [stateDim,]
            state = [agent3dPosition, agent3dHeading, next3DbSpline, rangeArray]
        """
        # get agent 3d position
        agent_pos_str = self.tcp_client.send_data(self.veh_get_position_cmd("0", "0"))        
        items = agent_pos_str.split(",")
        agent_position = np.array([float(items[0]), float(items[1]), float(items[2])])

        #  get agent heading
        resp = self.tcp_client.send_data(self.veh_get_heading_cmd("0", "0"))
        items = resp.split(",")
        agent_heading = np.array([float(items[0]), float(items[1]), float(items[2])])        

        # Get nearest spline point wrt to the agent position
        splines_str = self.tcp_client.send_data(
            self.get_nearest_spline_points_cmd("0", "0", agent_pos_str)
        )        
        items = splines_str.split(",")
        # n_splines = items[0]
        spline_pnts = np.array(items[1:], dtype=float)

        # Perform obstacle detection by ray tracing
        ray_angles_array, range_array, collision_array = self.get_range_data(
            agent_position, agent_heading
        )
        obs = np.hstack((agent_position, agent_heading, spline_pnts, range_array))
        return obs

    def step(self, action_idx):
        """
        execute the given action in the env and return the
        new env state as the result
        Args:
            action_idx (int): action index

        Returns:
            :
        """
        # execute chosen action
        action_cmd = self.action_set[action_idx]
        resp = self.tcp_client.send_data(action_cmd)
        print("action executed with response: {}".format(resp))

        # get new env state & reward
        new_obs = self.get_obs()
        reward = self.reward()  # todo
        self.step_count += 1

        # determine rollout termination status
        terminated = self.is_rollout_terminated()  # todo
        truncated = self.is_rollout_truncated  # todo
        info = {}
        return new_obs, reward, terminated, truncated, info

    def reward(self):
        r = 0
        return r

    def is_rollout_truncated(self):
        """
        the current rollout is truncated when
        the maximum number of time steps is reached
        """
        return self.step_count >= self.max_steps

    def is_rollout_terminated(self):
        terminated = False
        return terminated

    def demo_act(self):
        dt = self.end_time - self.start_time
        if dt < 2:  # wait for 2 secs between changing action
            cmd = self.veh_forward_cmd("0", "0")
        elif 2 <= dt <= 4:
            cmd = self.veh_left_cmd("0", "0")
        elif 4 < dt < 6:
            cmd = self.veh_backward_cmd("0", "0")
        else:
            cmd = self.veh_backward_cmd("0", "0")
        resp = env.tcp_client.send_data(cmd)

        self.end_time = time.time()

        # update the env state
        new_obs = self.get_obs()
        return new_obs

    def plot_env_obs(
        self,
        agent_position,
        agent_heading,
        rival_positions,
        rival_headings,
        range_array=None,
        collision_array=None,
    ):
        """
        x,y axes of positions and heading vectors are swapped in this method to
        convert engine world coords to openCV coords
        Args:
            agent_position (_type_): _description_
            agent_heading (_type_): _description_
            rival_positions (_type_): _description_
            rival_headings (_type_): _description_
            range_array (_type_, optional): _description_. Defaults to None.
            collision_array (_type_, optional): _description_. Defaults to None.
        """

        img_size = np.array([510, 500])  # (w, h)
        image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        radius = 3
        thickness = 2

        # draw agent at the center of the image
        agent_pos = int(img_size[0] / 2), int(img_size[1] / 2)
        cv2.circle(image, agent_pos, radius, (0, 255, 0), thickness)  # center  # color

        # draw agent heading
        agent_head = (agent_heading * 20)[:2] + img_size / 2
        cv2.line(
            image,
            agent_pos,
            (int(agent_head[1]), int(agent_head[0])),
            (0, 255, 0),
            thickness,
        )

        # draw rivals wrt to the agent
        for rival_position, rival_heading in zip(rival_positions, rival_headings):
            rival_head = (rival_position + rival_heading * 20 - agent_position)[
                :2
            ] + img_size / 2
            rival_head = (int(rival_head[1]), int(rival_head[0]))
            rival_a = (rival_position - agent_position)[:2] + img_size / 2
            rival_a = (int(rival_a[1]), int(rival_a[0]))
            cv2.circle(image, rival_a, radius, (255, 0, 0), thickness)
            cv2.line(image, rival_a, rival_head, (255, 0, 0), thickness)

        if range_array is not None and collision_array is not None:
            # draw detected obstacles surrounding the agent
            for pnt, rng in zip(collision_array, range_array):
                # transform collision point to the agent frame &
                # then to the cv img frame
                pnt_a = (pnt - agent_position)[:2] + img_size / 2
                pnt_a = (int(pnt_a[1]), int(pnt_a[0]))
                cv2.line(image, agent_pos, pnt_a, (0, 0, 255), thickness=1)
                # if rng < self.lidar_max_range:
                #     # the ray has indeed collided with an obstacle
                #     cv2.circle(image, pnt_a, radius=2, color=(0, 0, 255), thickness=1)
                # else:
                #     cv2.circle(image, pnt_a, radius=2, color=(0, 255, 0), thickness=1)

        # Display the image
        cv2.imshow("Env obs.", image)
        cv2.waitKey(1)

    def send_test_cmd(self, command_str):
        response = self.tcp_client.send_data(self.command_set[command_str])
        return response


def test_pos_n_ray_trace(env):
    while 1:
        # get current pose of the agent & rivals
        agent_pos_str = env.tcp_client.send_data(
            env.vehicle_get_position_command("0", "0")
        )
        agent_heading_resp = env.tcp_client.send_data(
            env.vehicle_get_heading_command("0", "0")
        )        

        # ray tracing
        ray_trace_cmd = env.get_ray_trace_command("0.0,103.0,0.7", "1.0,0.0,0.0")
        resp_str = env.tcp_client.send_data(ray_trace_cmd)        

        # get current pose of the agent & rivals
        agent_pos_str = env.tcp_client.send_data(
            env.vehicle_get_position_command("0", "0")
        )
        agent_heading_resp = env.tcp_client.send_data(
            env.vehicle_get_heading_command("0", "0")
        )                


if __name__ == "__main__":

    env = RacingAgent_v0(num_rivals=3, n_nearest_spline_pts=3, lidar_max_range=100.0)
    obs = env.reset()
    try:
        for i in range(70000):
            # print("env step: {}-------------------------------------".format(i))            
            obs = env.demo_act()
            veh_id = "0"
            veh_pos = env.tcp_client.send_data(env.veh_get_position_cmd("0", veh_id))
            # print("veh_pos: {}".format(veh_pos))

            lap_prog_cmd = env.get_lap_progress_cmd("0", "0", veh_pos)
            # print("lap progress cmd: {}".format(lap_prog_cmd))
            lap_progress = env.tcp_client.send_data(lap_prog_cmd)
            print("lap progress for veh {}: {}".format(veh_id, lap_progress))

    except KeyboardInterrupt:
        print("Script aborted by Ctrl+C")
        env.close()
