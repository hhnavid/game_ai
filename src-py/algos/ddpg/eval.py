import sys

sys.path.append("I:/projs/game-ai/src")

import ntpath
import argparse
import time
import torch
import numpy as np
import gymnasium as gym

# from envs.racing_agent_cont_v2 import RacingAgentContinuous_v2

from common.parse_args import parse_arguments
from algos.ddpg.ddpg import DDPG
from algos.ddpg.noise import OrnsteinUhlenbeckActionNoise


def main():
    # load experiment config from file
    print("sys.args: {}".format(sys.argv))
    args_dict = parse_arguments(sys.argv)
    if args_dict["env_type"] == "gym":
        eval_gym(args_dict)
    elif args_dict["env_type"] == "codeArt":
        eval_codeart(args_dict)


def eval_codeart(args_dict):
    pass


def eval_gym(args_dict):
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

    action_noise = None
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
        0,
        0,
        0,
        0,
        0,
        replay_buff_size=0,
        explore_render=bool(args_dict["explore_render"]),
        eval_render=bool(args_dict["eval_render"]),
        log_interval=0,
        return_plot=bool(args_dict["return_plot"]),
        obs_rms_plot=bool(args_dict["obs_rms_plot"]),
        resume=False,
        checkpoint_every_n_epoch=0,
        resume_path_prefix=False,
    )
