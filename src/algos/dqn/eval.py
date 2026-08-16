import sys

sys.path.append("I:/projs/game-ai/src")

import torch
import importlib
import gymnasium as gym
from envs.racing_agent_v0 import RacingAgent_v0
from envs.racing_agent_v1 import RacingAgent_v1
from envs.racing_agent_v2 import RacingAgent_v2
from envs.racing_agent_v3 import RacingAgent_v3
from common.parse_args import parse_arguments


def main():
    # load experiment config from file
    print("sys.args: {}".format(sys.argv))
    args_dict = parse_arguments(sys.argv)
    if args_dict["env_type"] == "gym":
        eval_gym(args_dict)
    elif args_dict["env_type"] == "codeArt":
        eval_codeart(args_dict)       
        
        
def eval_codeart(args_dict):
    # create env
    env = None
    if args_dict["env_id"] == "RacingAgent_v0":
        env = RacingAgent_v0(
            num_rivals=3,
            n_nearest_spline_pts=3,
            lidar_start_angle=-180,
            lidar_stop_angle=170,
            lidar_res=10,
            lidar_max_range=100.0,
            debug_plot=bool(args_dict["debug_plot"]),            
        )
    elif args_dict["env_id"] == "RacingAgent_v1":        
        env = RacingAgent_v1(
            num_rivals=3,
            n_nearest_spline_pts=3,
            lidar_start_angle=-180,
            lidar_stop_angle=180,
            lidar_res=10,
            lidar_max_range=100.0,
            debug_plot=bool(args_dict["debug_plot"]),            
        )
    elif args_dict["env_id"] == "RacingAgent_v2":
        env = RacingAgent_v2(
            num_rivals=3,
            n_nearest_spline_pts=3,
            lidar_start_angle=-180,
            lidar_stop_angle=180,
            lidar_res=18,
            lidar_max_range=200.0,
            debug_plot=bool(args_dict["debug_plot"]),            
        )
    elif args_dict["env_id"] == "RacingAgent_v3":
        env = RacingAgent_v3(
            num_rivals=3,
            n_nearest_spline_pts=3,
            lidar_start_angle=-180,
            lidar_stop_angle=180,
            lidar_res=18,
            lidar_max_range=200.0,
            debug_plot=bool(args_dict["debug_plot"]),
            # log_name="env_log.csv"
        )
    else:
        raise NotImplementedError

    # create Q network & load its params
    hidden_layers = []
    for l in args_dict["critic_hidden_layers"]:
        hidden_layers.append(int(l))
    
    if args_dict["critic_activation"] == "ReLU":
        activation_ = torch.relu
    else:
        raise NotImplementedError
    
    fan_ins = [self.obs_dim] + hidden_layers + [self.action_dim]
    self.q_network = QNetwork(fan_ins, activation_, lr).to(self.device)
    
    return


def eval_gym():
    pass


if __name__ == "__main__":
    main()