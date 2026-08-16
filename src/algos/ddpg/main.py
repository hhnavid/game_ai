import sys
sys.path.append("I:/projs/game-ai/src")

import ntpath
import argparse
import time
import torch
import numpy as np
import gymnasium as gym

from envs.racing_agent_cont_v2 import RacingAgentContinuous_v2
from envs.racing_agent_cont_v3 import RacingAgentContinuous_v3
from envs.racing_agent_cont_v4 import RacingAgentContinuous_v4

from common.parse_args import parse_arguments
from algos.ddpg.ddpg import DDPG
from algos.ddpg.noise import OrnsteinUhlenbeckActionNoise


def main():
    # load experiment config from file
    print("sys.args: {}".format(sys.argv))
    args_dict = parse_arguments(sys.argv)
    if args_dict["env_type"] == "gym":
        learn_gym(args_dict)
    elif args_dict["env_type"] == "codeArt":
        learn_codeart(args_dict)


def learn_codeart(args_dict):
    # create env
    env = None
    if args_dict["env_id"] == "RacingAgentContinuous_v2":
        env = RacingAgentContinuous_v2(
            num_rivals=3,
            n_nearest_spline_pts=3,
            lidar_max_range=200.0,
            lidar_start_angle=-180,
            lidar_stop_angle=180,
            lidar_res=18,
            debug_plot=bool(args_dict["debug_plot"]),
            write_log=False # don't log socket cmd send/recv
        )
    if args_dict["env_id"] == "RacingAgentContinuous_v3":
        env = RacingAgentContinuous_v3(
            num_rivals=3,
            n_nearest_spline_pts=3,
            lidar_max_range=200.0,
            lidar_start_angle=-180,
            lidar_stop_angle=180,
            lidar_res=18,
            randomize_init_pos=bool(args_dict["randomize_init_pos"]),
            debug_plot=bool(args_dict["debug_plot"]),
            write_log=False # don't log socket cmd send/recv
        )
    if args_dict["env_id"] == "RacingAgentContinuous_v4":
        env = RacingAgentContinuous_v4(
            num_rivals=3,
            n_nearest_spline_pts=3,
            lidar_max_range=200.0,
            lidar_start_angle=-180,
            lidar_stop_angle=180,
            lidar_res=18,
            randomize_init_pos=bool(args_dict["randomize_init_pos"]),
            debug_plot=bool(args_dict["debug_plot"]),
            write_log=False # don't log socket cmd send/recv
        )
    else:
        raise NotImplementedError    
    
    setup_ddpg(
        args_dict,
        env.state_dim,
        env.action_dim,
        env.max_action,
        env,
        "codeArt",
        eval_env=None,
    )


def learn_gym(args_dict):
    # env rendering mode
    if args_dict["explore_render"]:
        expl_render = "human"
    else:
        expl_render = ""
    
    if args_dict["eval_render"]:
        eval_render = "human"
    else:
        eval_render = ""
        
    # setup envs    
    env = gym.make(args_dict["env_id"], render_mode=expl_render)
    eval_env = gym.make(args_dict["env_id"], render_mode=eval_render)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = np.abs(env.action_space.low)

    setup_ddpg(args_dict, state_dim, action_dim, max_action, env, "gym", eval_env)
    return


def setup_ddpg(
    args_dict,
    state_dim,
    action_dim,
    max_action,
    env,
    env_type,
    eval_env=None,
):
    # critic & actor networks' config
    critic_hidden_layers = []
    for layer_num_units in args_dict["critic_hidden_layers"]:
        critic_hidden_layers.append(int(layer_num_units))
    if args_dict["critic_activation"] == "ReLU":
        critic_activation = torch.relu
    else:
        raise NotImplementedError

    actor_hidden_layers = []
    for layer_num_units in args_dict["actor_hidden_layers"]:
        actor_hidden_layers.append(int(layer_num_units))
    if args_dict["actor_activation"] == "ReLU":
        actor_activation = torch.relu
    else:
        raise NotImplementedError

    if args_dict["resume"]:
        resume_path_prefix = args_dict["resume_path"]
    else:
        resume_path_prefix = None

    action_noise = parse_noise_type(env, args_dict["noise_type"], action_dim)
    ddpg = DDPG(
        int(args_dict["n_train_steps"]),
        int(args_dict["n_rollout_steps"]),
        int(args_dict["n_eval_steps"]),
        int(float(args_dict["n_total_timesteps"])),
        env,
        args_dict["env_id"],
        args_dict["env_type"],
        eval_env,
        state_dim,
        action_dim,
        max_action,
        action_noise,
        actor_hidden_layers,
        actor_activation,
        critic_hidden_layers,
        critic_activation,
        False,  # layer normalization
        bool(args_dict["normalize_obs"]),
        False,  # reward normalization
        float(args_dict["gamma"]),
        float(args_dict["tau"]),
        float(args_dict["actor_lr"]),
        float(args_dict["critic_lr"]),
        int(args_dict["batch_size"]),
        replay_buff_size=int(float(args_dict["replay_buff_size"])),
        explore_render=bool(args_dict["explore_render"]),
        eval_render=bool(args_dict["eval_render"]),
        log_interval=int(args_dict["log_interval"]),
        return_plot=bool(args_dict["return_plot"]),
        obs_rms_plot=bool(args_dict["obs_rms_plot"]),
        resume=bool(args_dict["resume"]),
        checkpoint_every_n_epoch=int(args_dict["checkpoint_every_n_epoch"]),
        resume_path_prefix=resume_path_prefix,
    )
    ddpg.learn()
    env.close()
    if eval_env is not None:
        eval_env.close()
    return


def parse_noise_type(env, noise_type, action_dim):
    action_noise = None
    for current_noise_type in noise_type.split(","):

        current_noise_type = current_noise_type.strip()
        if "ou" in current_noise_type:
            _, stddev = current_noise_type.split("_")
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim), sigma=float(stddev) * np.ones(action_dim)
            )
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))
    return action_noise


if __name__ == "__main__":
    main()
