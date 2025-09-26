import sys
sys.path.append('I:/projs/game-ai/src')

from common.parse_args import parse_arguments
import numpy as np
import gymnasium as gym
import torch.nn as nn
from algos.dqn.dqn import DQN


def main():          
    # load experiment config from file    
    print("sys.args: {}".format(sys.argv)) 
    args_dict = parse_arguments(sys.argv)
    
    # setup env        
    env = gym.make(args_dict['env_id'], render_mode="human")
    eval_env = gym.make(args_dict['env_id'], render_mode="human")
    
    state_dim = env.observation_space.shape[0]            
    if env.action_space.shape == ():
        # for discrete action spaces, action_dim == number of available discrete actions 
        action_dim = int(env.action_space.n)
    else:
        # for continuous action spaces, action_dim == action vector dimension        
        action_dim = env.action_space.shape
        
    hidden_layers = []
    for l in args_dict['critic_hidden_layers']:
        hidden_layers.append(int(l))
    
    # setup DQN agent        
    activation_ = getattr(nn, args_dict['critic_activation'])
    dqn = DQN(int(args_dict['n_train_steps']), 
              int(args_dict['n_rollout_steps']),
              int(args_dict['n_eval_steps']),
              int(float(args_dict['n_total_timesteps'])),
              env, args_dict['env_id'],
              eval_env,
              state_dim, env.observation_space.dtype,
              action_dim, env.action_space.dtype,
              hidden_layers=hidden_layers,
              activation_=activation_,
              normalize_obs=bool(args_dict['normalize_obs']),
              gamma=float(args_dict['gamma']),
              tau=float(args_dict['tau']),
              lr=float(args_dict['lr']),
              bs=int(args_dict['batch_size']),
              exploration_fraction=0.1,
              exploration_initial_eps=1.0, exploration_final_eps=0.05,
              replay_buff_size=int(float(args_dict['replay_buff_size'])),
              explore_render=bool(args_dict['explore_render']),
              eval_render=bool(args_dict['eval_render']),
              log_interval=int(args_dict['log_interval']))
    dqn.learn()
    env.close()
    
if __name__ == '__main__':
    main()