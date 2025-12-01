import sys

sys.path.append("I:/projs/game-ai/src")

import os
import csv
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from network_.sock_ import TcpClient
from common.tf import rotate_point_around_axis

deg2rad = 0.017453292519943295
rad2deg = 57.2957795130823208


class RacingAgent_v1:

    def __init__(
        self,
        num_rivals,
        n_nearest_spline_pts,
        lidar_max_range,
        lidar_start_angle=-180,
        lidar_stop_angle=170,
        lidar_res=45,
        debug_plot=False,
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
        # log_path_prefix is set by the RL algorithm
        # when resume path is set in its constructor
        self.log_path_prefix = None

        self.fig, self.ax = plt.subplots()

        self.num_rivals = num_rivals
        self.n_nearest_spline_pts = str(n_nearest_spline_pts)

        # create tcp client
        self.tcp_client = TcpClient(ip="127.0.0.1", port=8080)

        self.group_ids4ray_trace = {"rival": "1", "wall": "2"}
        self.command_set = {
            "Level_Reload": "70001",
            "Level_Pause": "10101,0",
            "Level_Play": "10101,1",
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
        self.agent_last_pos_time = 0
        self.agent_last_position = None
        self.agent_pos_str = ""
        # initial position of the agent at the start of the lap
        self.agent_position0 = None
        self.agent_last_prog = 0.0
        self.rivals_lap_progress = np.zeros(self.num_rivals)

        self.n_beams = int((lidar_stop_angle - lidar_start_angle) / lidar_res)
        self.lidar_max_range = lidar_max_range
        self.lidar_start_angle = lidar_start_angle * deg2rad
        self.lidar_stop_angle = lidar_stop_angle * deg2rad
        self.lidar_res = lidar_res * deg2rad

        # state
        #   car velocity vectr, 3D
        #   angle between car's heading and the track axis, 1D
        #   distance between the car and the track axis (line at the center of the track) 1D
        #   num lidar beams
        self.state_dim = 5 + self.n_beams
        self.state_dtype = np.float32
        self.action_dim = 8
        self.action_dtype = np.int64
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
        # step_count is used to detect rollout timeout when
        # it exceeds the max_steps per rollout
        self.step_count = 0
        self.max_steps = 200000

        self.debug_plot = debug_plot
        if self.debug_plot:
            self.img_size = np.array([700, 700])  # (w, h)
            self.static_img = np.zeros(
                (self.img_size[0], self.img_size[1], 3), dtype=np.uint8
            )
        else:
            self.img_size = None
            self.static_img = None
        self.reset()

    def close(self):
        self.tcp_client.close()

    def seed(self):
        print("RacingAgent evn. doesn't support random seeding...")

    def reset(self):
        """resets the environment state
        Returns:
            obs: initial state after env. reset
            {}: a dict to be compatible with gym env syntax
        """
        while True:
            # send reset command to the racing app
            reset_status = self.tcp_client.send_data(self.command_set["Level_Reload"])
            print("Resetting level, eng. response: {}".format(reset_status))
            time.sleep(0.3)  # sleep for 300ms to avoid crashing
            if reset_status == "Successful":
                break

        self.step_count = 0
        self.agent_last_prog = 0.0
        self.rivals_lap_progress[:] = 0.0

        # prepare the initial obs after reset
        obs, _, _, _ = self.get_obs()

        items = self.agent_pos_str.split(",")
        self.agent_position0 = np.array(items, dtype=float)
        
        self.agent_last_pos_time = time.perf_counter()
        self.agent_last_position = self.agent_position0.copy()

        if self.debug_plot:
            # reset the trajectories of the last rollout
            self.static_img = np.zeros(
                (self.img_size[0], self.img_size[1], 3), dtype=np.uint8
            )

            # initialize the nearest spline points log file
            if self.log_path_prefix is not None:
                with open(
                    os.path.join(self.log_path_prefix, "env_log.csv"), "w", newline=""
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(
                        [
                            "rival 3d pos.x",
                            "rival 3d pos.y",
                            "rival 3d pos.z",
                            "spline pnt0.x",
                            "spline pnt0.y",
                            "spline pnt0.z",
                            "spline pnt1.x",
                            "spline pnt1.y",
                            "spline pnt1.z",
                            "spline pnt2.x",
                            "spline pnt2.y",
                            "spline pnt2.z",
                        ]
                    )

        return obs, {}

    def get_nearest_spline_pnts(self, lvl_id, spline_id, veh_3dpos_str):
        """
        Args:
            lvl_id (str): level ID
            spline_id (str): the spline ID from which the spline points nearest to the vehicle is extracted
            veh_pos (str): 3d position of the vehicle wrt which the nearest spline points are determined
        """
        splines_str = self.tcp_client.send_data(
            self.get_nearest_spline_points_cmd(lvl_id, spline_id, veh_3dpos_str)
        )
        items = splines_str.split(",")
        n_splines_pnts = int(items[0])
        spline_pnts = np.array(items[1:], dtype=float)
        return n_splines_pnts, spline_pnts

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
            self.lidar_stop_angle,
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
            has_collided, group_id, ray_endpoint = (
                int(items[0]),
                items[1],
                items[2:],
            )

            # only consider walls (road boundaries and rivals as obstacles)
            if has_collided and (
                group_id == self.group_ids4ray_trace["rival"]
                or group_id == self.group_ids4ray_trace["wall"]
            ):
                ray_endpoint = np.array(
                    [
                        float(ray_endpoint[0]),
                        float(ray_endpoint[1]),
                        float(ray_endpoint[2]),
                    ],
                )
                range_ = np.linalg.norm(ray_endpoint - ray_origin)
                # if group_id == 1:
                #     print("collided with a rival")
            else:
                # just for debug
                # ray_endpoint = np.array(
                #     [
                #         float(ray_endpoint[0]),
                #         float(ray_endpoint[1]),
                #         float(ray_endpoint[2]),
                #     ],
                # )
                # print(
                #     "group id: {}, range: {}".format(
                #         group_id, np.linalg.norm(ray_endpoint - ray_origin)
                #     )
                # )
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
        """                
        # angle between car's heading and the track axis, 1D
        # --------------------------------------------------
        # get agent 3d position
        self.agent_pos_str = self.tcp_client.send_data(
            self.veh_get_position_cmd("0", "0")
        )
        cur_time = time.perf_counter()
        items = self.agent_pos_str.split(",")
        agent_position = np.array(items, dtype=float)                

        # agent's heading angle to track direction
        # --------------------------------------------------
        # get agent's heading
        resp = self.tcp_client.send_data(self.veh_get_heading_cmd("0", "0"))
        items = resp.split(",")
        agent_heading = np.array(items, dtype=float)
        agent_heading /= np.linalg.norm(agent_heading)

        # get spline points that are nearest to the agent position
        # note: the agent's position is always between the 1st and 2nd
        #       nearest spline points
        n_splines, spline_pnts = self.get_nearest_spline_pnts(
            "0", "0", self.agent_pos_str
        )
        spline_pnts = spline_pnts.reshape((n_splines, 3))

        # prepare track axis direction vector
        track_direction = spline_pnts[1] - spline_pnts[0]
        track_dir_norm = np.linalg.norm(track_direction)
        track_direction /= track_dir_norm
        rot_axis = np.array(
            [0.0, 0.0, 1.0]
        )  # normal vector used to define the positive/negative direction of rotation
        x_prod = np.cross(agent_heading, track_direction)
        dot_prod = np.dot(agent_heading, track_direction)
        angle2track = np.atan2(np.dot(rot_axis, x_prod), dot_prod)  # radian
        
        # agent velocity along the track direction
        # --------------------------------------------------
        if self.step_count > 0:
            dt = cur_time - self.agent_last_pos_time
            agent_velocity = (agent_position - self.agent_last_position) / dt
            velocity_mag = np.linalg.norm(agent_velocity)
            normalized_velocity = agent_velocity / (velocity_mag + 0.0001)
            self.agent_last_position = agent_position.copy()
            self.agent_last_pos_time = cur_time
            x_prod = np.cross(normalized_velocity, track_direction)
            dot_prod = np.dot(normalized_velocity, track_direction)
            vel_angle2track = np.atan2(np.dot(rot_axis, x_prod), dot_prod)
        else:
            vel_angle2track = angle2track
            agent_velocity = np.zeros(3, dtype=float)

        # distance between the car and the track axis (line at the center of the track) 1D
        # lidar beams
        # --------------------------------------------------------------------------------
        vec2agent = agent_position - spline_pnts[0]
        dist2track = (
            np.linalg.norm(np.cross(track_direction, vec2agent)) / track_dir_norm
        )

        # Perform obstacle detection by ray tracing
        # ---------------------------------------------------------------------------------
        ray_trace_origin = agent_position.copy()
        ray_angles_array, range_array, collision_array = self.get_range_data(
            ray_trace_origin, agent_heading
        )

        # collided_with_rivals, collided_with_walls
        # ---------------------------------------------------------------------------------
        coll_resp = self.tcp_client.send_data(
            self.get_veh_has_collided_cmd("0", "0", self.group_ids4ray_trace["rival"])
        )
        collided_with_rivals = int(coll_resp.split(",")[0])

        coll_resp = self.tcp_client.send_data(
            self.get_veh_has_collided_cmd("0", "0", self.group_ids4ray_trace["wall"])
        )
        collided_with_walls = int(coll_resp.split(",")[0])

        obs = np.hstack((agent_velocity, angle2track, dist2track, range_array))
        # print(
        #     "obs: spd: {:.3f}, angle2track: {:.3f}, dist2track: {:.3f} ".format(
        #         cur_speed, angle2track * rad2deg, dist2track
        #     )
        # )

        if self.debug_plot:
            rival_id4spline = 0
            rival_positions = np.zeros((self.num_rivals, 3))  # [#rivals x 3]
            rival_headings = np.zeros((self.num_rivals, 3))  # [#rivals x 3]
            for i in range(self.num_rivals):
                # get rival position
                rival_pos_str = self.tcp_client.send_data(
                    self.veh_get_position_cmd("0", str(i + 1))
                )
                items = rival_pos_str.split(",")
                rival_positions[i, :] = (
                    float(items[0]),
                    float(items[1]),
                    float(items[2]),
                )

                if i == rival_id4spline:
                    # Get spline points that are nearest to one of the rivals
                    rival_n_splines, rival_spline_pnts = self.get_nearest_spline_pnts(
                        "0", "0", rival_pos_str
                    )

                # get rival heading
                resp = self.tcp_client.send_data(
                    self.veh_get_heading_cmd("0", str(i + 1))
                )
                items = resp.split(",")
                rival_headings[i, :] = float(items[0]), float(items[1]), float(items[2])

            if rival_n_splines > 0:
                rival_spline_pnts = rival_spline_pnts.reshape((rival_n_splines, 3))
            else:
                rival_spline_pnts = None

            # log nearest spline points for the rival
            if self.log_path_prefix is not None:
                with open(
                    os.path.join(self.log_path_prefix, "env_log.csv"), "a", newline=""
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    row = rival_positions[rival_id4spline].tolist()
                    for j in range(rival_n_splines):
                        row += rival_spline_pnts[j, :].tolist()
                    writer.writerow(row)
            if (
                self.step_count > 0
            ):  # avoid plotting at the 1st step since agent_position0 hasn't been set yet
                self.plot_env_obs(
                    agent_position,
                    agent_heading,
                    rival_positions,
                    rival_headings,
                    range_array,
                    collision_array,
                    spline_pnts,
                    spline_pnts[0],
                    rival_spline_pnts,
                )
        return obs, collided_with_rivals, collided_with_walls, vel_angle2track

    def plot_env_obs(
        self,
        agent_position,
        agent_heading,
        rival_positions,
        rival_headings,
        range_array=None,
        collision_array=None,
        nearest_spline_pnts=None,
        next_spline_pnt=None,
        rival_spline_pnts=None,
    ):
        """
        x,y axes of positions and heading vectors are swapped in this method to
        convert engine world coords to openCV coords
        """
        scale_ = 1.5
        # contains alpha transparent channel
        dynamic_img = np.zeros((self.img_size[0], self.img_size[1], 4), dtype=np.uint8)
        radius = 3
        thickness = 2

        # draw agent position wrt its initial position
        agent_pos = (agent_position - self.agent_position0)[
            :2
        ] * scale_ + self.img_size / 2
        agent_pos = int(agent_pos[1]), int(agent_pos[0])

        # draw agent heading
        agent_head = (agent_position + agent_heading * 20 - self.agent_position0)[
            :2
        ] * scale_ + self.img_size / 2
        cv2.line(
            dynamic_img,
            agent_pos,
            (int(agent_head[1]), int(agent_head[0])),
            (0, 0, 255, 255),
            thickness,
        )

        # draw rivals wrt to the agent start position
        colors = [
            (5, 143, 255, 255),  # orange
            (0, 255, 0, 255),  # green
            (255, 0, 0, 255),  # blue
        ]
        for rival_id, (rival_position, rival_heading) in enumerate(
            zip(rival_positions, rival_headings)
        ):
            rival_head = (rival_position + rival_heading * 20 - self.agent_position0)[
                :2
            ] * scale_ + self.img_size / 2
            rival_head = (int(rival_head[1]), int(rival_head[0]))
            rival_a = (rival_position - self.agent_position0)[
                :2
            ] * scale_ + self.img_size / 2
            rival_a = (int(rival_a[1]), int(rival_a[0]))
            cv2.circle(dynamic_img, rival_a, radius, colors[rival_id], thickness)
            cv2.line(dynamic_img, rival_a, rival_head, colors[rival_id], thickness)
            cv2.circle(self.static_img, rival_a, 1, colors[rival_id], thickness=1)
        if range_array is not None and collision_array is not None:
            # draw detected obstacles surrounding the agent
            for pnt, rng in zip(collision_array, range_array):
                # transform collision point to the agent frame &
                # then to the cv img frame
                pnt_a = (pnt - self.agent_position0)[:2] * scale_ + self.img_size / 2
                pnt_a = (int(pnt_a[1]), int(pnt_a[0]))
                if rng >= self.lidar_max_range:
                    cv2.line(
                        dynamic_img, agent_pos, pnt_a, (0, 255, 0, 255), thickness=1
                    )
                    # cv2.circle(dynamic_img, pnt_a, radius, (255, 255, 255, 255), thickness=1)
                else:
                    cv2.line(
                        dynamic_img, agent_pos, pnt_a, (0, 0, 255, 255), thickness=1
                    )

        cv2.circle(dynamic_img, agent_pos, radius, (0, 0, 255, 255), thickness)
        cv2.circle(self.static_img, agent_pos, 1, (0, 0, 255, 255), thickness=1)

        # draw spline points nearest to the agent/rival
        if nearest_spline_pnts is not None:
            for i in range(nearest_spline_pnts.shape[0]):
                spline_pnt = (nearest_spline_pnts[i, :] - self.agent_position0)[
                    :2
                ] * scale_ + self.img_size / 2
                spline_pnt = (int(spline_pnt[1]), int(spline_pnt[0]))
                cv2.circle(
                    dynamic_img, spline_pnt, radius + 1, (255, 255, 0, 255), thickness=1
                )
                cv2.putText(
                    dynamic_img,
                    "sp" + str(i),
                    spline_pnt,
                    cv2.FONT_HERSHEY_SIMPLEX,  # font face
                    0.5,  # font scale
                    (255, 255, 255, 255),
                    1,  # thickness
                    cv2.LINE_AA,
                )

                if rival_spline_pnts is not None and i < rival_spline_pnts.shape[0]:
                    spline_pnt = (rival_spline_pnts[i, :] - self.agent_position0)[
                        :2
                    ] * scale_ + self.img_size / 2
                    spline_pnt = (int(spline_pnt[1]), int(spline_pnt[0]))
                    cv2.circle(dynamic_img, spline_pnt, radius, colors[0], thickness=1)
                    cv2.putText(
                        dynamic_img,
                        "sp" + str(i),
                        spline_pnt,
                        cv2.FONT_HERSHEY_SIMPLEX,  # font face
                        0.5,  # font scale
                        (255, 255, 255, 255),
                        1,  # thickness
                        cv2.LINE_AA,
                    )

            nxt_pnt = (next_spline_pnt - self.agent_position0)[
                :2
            ] * scale_ + self.img_size / 2
            nxt_pnt = (int(nxt_pnt[1]), int(nxt_pnt[0]))
            cv2.line(dynamic_img, agent_pos, nxt_pnt, (0, 255, 0, 255), thickness=1)

        # Split foreground into color and alpha channels
        b, g, r, a = cv2.split(dynamic_img)

        # Normalize alpha channel to 0-1 range
        alpha = a.astype(float) / 255

        # Prepare 3-channel alpha for blending
        alpha_3 = cv2.merge([alpha, alpha, alpha])

        # Convert fg BGR channels to float
        fg_color = cv2.merge([b, g, r]).astype(float)
        bg_color = self.static_img.astype(float)

        # Alpha blend foreground and background
        out_img = fg_color * alpha_3 + bg_color * (1 - alpha_3)
        out_img = out_img.astype(np.uint8)

        # Display the image
        cv2.imshow("Env obs.", out_img)
        cv2.waitKey(1)

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
        action_cmd = self.action_set[action_idx]("0", "0")  # lvl_idx, veh_idx
        resp = self.tcp_client.send_data(action_cmd)
        # print(
        #     "step: {}, action {} exec resp: {}".format(
        #         self.step_count,
        #         self.action_set[action_idx].__name__, resp
        #     )
        # )

        # wait a bit for the action to take effect
        time.sleep(0.1)  # 100 ms

        # get new env state & reward
        new_obs, collided_with_rivals, collided_with_walls, vel_angle2track = self.get_obs()
        reward = self.reward(
            new_obs, collided_with_rivals, collided_with_walls, vel_angle2track
        )
        self.step_count += 1

        # determine rollout termination status
        terminated = self.is_rollout_terminated()
        truncated = self.is_rollout_timed_out()
        if terminated:
            print("rollout is terminated-------------------------------")
        info = {}

        return new_obs, reward, terminated, truncated, info

    def is_rollout_timed_out(self):
        """
        the current rollout is timed out (truncated) when
        the maximum number of time steps is reached
        """
        return self.step_count >= self.max_steps

    def is_rollout_terminated(self):
        """
        the current rollout is terminated when
        the agent finishes its lap or all of the
        rivals finish the lap
        """
        rivals_finished = False
        for i in range(self.num_rivals):
            rivals_finished = rivals_finished or (self.rivals_lap_progress[i] > 0.99)
        return (1.0 - self.agent_last_prog < 0.01) or bool(rivals_finished)

    def get_rank(self):
        rank = self.num_rivals + 1
        agent_position = np.array(self.agent_pos_str.split(","), dtype=float)
        for i in range(1, self.num_rivals + 1):
            rival_pos_str = self.tcp_client.send_data(
                self.veh_get_position_cmd("0", str(i))
            )
            lap_prog_cmd = self.get_lap_progress_cmd(
                "0", "0", rival_pos_str  # lvl_idx, spline_idx
            )

            prgs = float(self.tcp_client.send_data(lap_prog_cmd))
            # only consider the rivals progress before completing
            # the 1st lap. After the lap completion, the rivals progress
            # will be reset which must not be used:
            if prgs > self.rivals_lap_progress[i - 1]:
                self.rivals_lap_progress[i - 1] = prgs
            if self.agent_last_prog > self.rivals_lap_progress[i - 1]:
                # agent's lap rank is higher than ith rival due
                # to having a higher lap progress
                rank -= 1
        return rank

    def reward(self, obs, collided_with_rivals, collided_with_walls, vel_angle2track):
        """
        obs = np.hstack(cur_speed, angle2track, dist2track, range_array)
        Args:
            obs (np array): env state {cur_speed, angle2track, dist2track, range_array}            
            collided_with_rivals (int): 1 if the agent has collided with rival(s), 0 o.w.
            collided_with_walls (int): 1 if the agent has collided with the track boundaries, 0 o.w.
            vel_angle2track (float, radian): the angle between the car velocity and track direction
        """
        # reward for vehicle speed along track direction
        vel_mag = np.linalg.norm(obs[:3])
        # print("velocity mag: {:.3f}, velocity_ang2track: {:.3f}".format(vel_mag, vel_angle2track * rad2deg))
        r_speed =  0.1 * vel_mag * np.cos(vel_angle2track)
        
        # penalty for agent heading difference with track direction
        cos_ang2track = np.cos(obs[3])
        r_angle2track = 0.3 * cos_ang2track

        # penalty for distance from the track central axis
        r_dist2center = -0.1 * obs[4]        

        # penalty for getting too close to obstacles
        min_lidar_range = np.min(obs[5:])
        r_obs_avoid = -0.2 / (min_lidar_range + 0.0001)

        # penalty for colliding with track boundaries or other rivals
        r_collision = -10 * collided_with_walls - 3 * collided_with_rivals

        # progress reward (percentage of lap completion)
        lap_prog_cmd = self.get_lap_progress_cmd(
            "0", "0", self.agent_pos_str  # lvl_idx, spline_idx
        )
        cur_prog = float(self.tcp_client.send_data(lap_prog_cmd))
        if cur_prog - self.agent_last_prog < 0.5:
            r_prog = cur_prog - self.agent_last_prog
        else:
            # the agent has moved backward toward the starting line -> penalize it
            print(
                "Ignoring lap progress since the agent has moved backward toward the starting line!"
            )
            r_prog = (cur_prog - 1) - self.agent_last_prog
        r_prog *= 10
        self.agent_last_prog = cur_prog

        # rank reward
        agent_rank = (
            self.get_rank()
        )  # DON'T COMMENT THIS LINE! IT'S NEEDED FOR CHECKING ROLLOUT TERMINATION CONDITION
        # r_rank = 1.0 / agent_rank

        r_total = (
            r_speed + r_dist2center + r_angle2track + r_obs_avoid + r_collision + r_prog
        )
        print(
            "r_spd: {:.3f}, r_dist2trk: {:.3f}, r_ang2trk: {:.3f}, r_obs: {:.3f}, r_coll: {:.3f}, r_prog: {:.3f}".format(
                r_speed, r_dist2center, r_angle2track, r_obs_avoid, r_collision, r_prog
            )
        )
        return r_total

    def pause_(self):
        """Pause env."""
        resp = self.tcp_client.send_data(self.level_pause_cmd())
        print("level pausing response: {}".format(resp))

    def play_(self):
        """Resume playing the env. after a pause command"""
        resp = self.tcp_client.send_data(self.level_play_cmd())
        print("level playing response: {}".format(resp))

    def level_play_cmd(self):
        """
        returns command to resume playing the level, physics, ...
        after a pause command
        """
        return self.command_set["Level_Play"]

    def level_pause_cmd(self):
        """
        returns command to pause the level rendering, physics, ...
        """
        return self.command_set["Level_Pause"]

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

    def get_all_spline_points_cmd(self, lvl_id, spline_id):
        """
        prepares the command for getting all of the spline points
        Args:
            lvl_id (str): level index
            spline_id (str): spline index
        """
        return self.command_set["Spline_GetAllPoints"] + "," + lvl_id + "," + spline_id

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

    def get_veh_speed_cmd(self, lvl_idx, veh_idx):
        return self.command_set["Vehicle_GetSpeed"] + "," + lvl_idx + "," + veh_idx

    def get_veh_has_collided_cmd(self, lvl_idx, veh_idx, tag_ids):
        """
        Checks whether the vehicle has collided with walls or other rivals
        Args:
            lvl_idx (str): level index
            veh_idx (str): vehicle index for which collision test is performed
            tag_ids (str): comma-seperated tag IDs of the entities with which
                           the agent collision is checked
        """
        command = (
            self.command_set["Vehicle_HasCollided"]
            + ","
            + lvl_idx
            + ","
            + veh_idx
            + ","
            + tag_ids
        )
        return command

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
