import sys
sys.path.append('I:\projs\game-ai\src')

import gymnasium as gym
import torch.nn as nn
from algos.dqn.dqn import DQN


if __name__ == '__main__':
    
    # setup env    
    env = gym.make("Acrobot-v1", render_mode="human")
    
    state_dim = env.observation_space.shape[0]        
    if len(env.action_space.shape) == 0:
        # discrete action space
        action_dim = 1
    else:
        # continuous action space
        action_dim = env.action_space.shape
    
    # setup DQN agent
    dqn = DQN(env, state_dim, action_dim,
              hidden_layers=[64, 63], activation_=nn.ReLU,
              normalize_obs=True, gamma=0.95, tau=1.0, lr=1e-4,
              bs=32, exploration_fraction=0.1,
              exploration_initial_eps=1.0, exploration_final_eps=0.05,
              replay_buff_size=10000)