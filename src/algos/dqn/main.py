import sys
sys.path.append('I:/projs/game-ai/src')

import torch
import importlib      
import gymnasium as gym
from envs.racing_agent_v0 import RacingAgent_v0
from common.parse_args import parse_arguments
from algos.dqn.dqn import DQN


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
    if args_dict["env_id"] == "RacingAgent_v0":        
        env = RacingAgent_v0(num_rivals=3, n_nearest_spline_pts=1, lidar_max_range=60.0)
    else:
        raise NotImplementedError   
    
    setup_dqn(args_dict, env.state_dim, env.state_dtype,
              env.action_dim, env.action_dtype,
              env, "codeArt", eval_env=None)
    return
        
        
def learn_gym(args_dict):
    # setup env        
    env = gym.make(args_dict['env_id'])#, render_mode="human")
    eval_env = gym.make(args_dict['env_id'], render_mode="human")
    
    state_dim = env.observation_space.shape[0]            
    if env.action_space.shape == ():
        # for discrete action spaces, action_dim == number of available discrete actions 
        action_dim = int(env.action_space.n)
    else:
        # for continuous action spaces, action_dim == action vector dimension        
        action_dim = env.action_space.shape
        
    setup_dqn(args_dict,
              state_dim, env.observation_space.dtype,
              action_dim, env.action_space.dtype,
              env, "gym", eval_env)
    return
        
def setup_dqn(args_dict, state_dim, state_dtype,
              action_dim, action_dtype,
              env, env_type, eval_env=None):
        
    hidden_layers = []
    for l in args_dict['critic_hidden_layers']:
        hidden_layers.append(int(l))
    
    # setup DQN agent          
    if args_dict['critic_activation'] == 'ReLU':
        activation_ = torch.relu    
    else:
        raise NotImplementedError
    if args_dict['resume']:
        resume_path_prefix = args_dict['resume_path']
    else:
        resume_path_prefix = None
    dqn = DQN(int(args_dict['n_train_steps']), 
              int(args_dict['n_rollout_steps']),
              int(args_dict['n_eval_steps']),
              int(float(args_dict['n_total_timesteps'])),
              env, args_dict['env_id'], env_type,
              eval_env,
              state_dim, state_dtype,
              action_dim, action_dtype,
              is_action_discrete=True,
              hidden_layers=hidden_layers,
              activation_=activation_,
              normalize_obs=bool(args_dict['normalize_obs']),
              gamma=float(args_dict['gamma']),
              tau=float(args_dict['tau']),
              lr=float(args_dict['lr']),
              bs=int(args_dict['batch_size']),
              exploration_fraction=0.12,
              exploration_initial_eps=1., exploration_final_eps=0.05,
              replay_buff_size=int(float(args_dict['replay_buff_size'])),
              explore_render=bool(args_dict['explore_render']),
              eval_render=bool(args_dict['eval_render']),
              log_interval=int(args_dict['log_interval']),
              return_plot=bool(args_dict['return_plot']),
              obs_rms_plot=bool(args_dict['obs_rms_plot']),
              resume=bool(args_dict['resume']),
              checkpoint_every_n_epoch=int(args_dict['checkpoint_every_n_epoch']),
              resume_path_prefix=resume_path_prefix)
    
    dqn.learn()
    env.close()
    if eval_env is not None:
        eval_env.close()
    return

if __name__ == '__main__':
    main()