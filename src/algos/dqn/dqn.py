import sys
sys.path.append('I:\projs\game-ai\src')

import torch
import torch.nn as nn
import numpy as np
import random
from common.stbl3_buffers import ReplayBuffer
from common.utils import LinearSchedule
from common.net_param_manip import init_xavier_weights
from common.running_mean_std import *


class QNetwork(nn.Module):
    
    def __init__(self, fan_ins, activation, lr):
        """Q network class for DQN
        Args:
            fan_ins: list containing input dimension for each net. layer. 1st item in the list must be state_dim and last item must be action_dim
            activation: Network activation function {torch.relu, torch.tanh, ...}
            learning_rate:
        """
        super(QNetwork, self).__init__()
        
        self.activation = activation        
        
        # create network layers
        self.layers = nn.Sequential()
        layer_items = []
        for i in range(len(fan_ins) - 1):
            layer_items.append(('Linear{}'.format(i), torch.nn.Linear(fan_ins[i], fan_ins[i + 1])))
            
        [self.layers.add_module(name, layer) for name, layer in layer_items]

        # Init network weights
        self.apply(init_xavier_weights)                            
    
    def forward(self, states):
        # Hidden layers
        out = states
        i = 0
        while i < len(self.layers) - 1:
            out = self.layers[i](out)           # Linear layer: Wx+b
            i += 1
            if self.layer_norm:
                out = self.layers[i](out)       # Layer normalization
                i += 1
            out = self.activation(out)          # activation function

        # Last layer doesn't need layer normalization
        out  = self.layers[-1](out)            
        return out
    

class DQN:
    
    def __init__(self, env, 
                 state_dim, action_dim, 
                 hidden_layers=[64, 63], activation_=nn.ReLU,
                 normalize_obs=True, gamma=0.95, tau=1.0, lr=1e-4, bs=32,
                 exploration_fraction=0.1, exploration_initial_eps=1.0,
                 exploration_final_eps=0.05, replay_buff_size=10000):
        
        # the environment that the agent must learn
        self.env = env        
        
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction        
        
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        fan_ins = [self.state_dim] + hidden_layers + [self.action_dim]
        self.q_network = QNetwork(fan_ins, activation_, lr).to(self.device)
        
        self.target_network = QNetwork(fan_ins, activation_, lr).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buff_size,
                                          obs_dim=state_dim, obs_dtype=np.float32,
                                          action_dim=action_dim, action_dtype=np.float32,
                                          device=device_name,
                                          handle_timeout_termination=True)
        self.gamma = gamma
        self.batch_size = bs                                
        
        # num of steps that the agent has interacted with the env
        self.steps_done = 0  
        
        # Observation normalization
        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.obs_rms = RunningMeanStdMPI(shape=(self.state_dim,))
        else:
            self.obs_rms = None
            
    def collect_rollouts(self, ):
        """
        Interact with the env and store observed transitions in
        the replay buffer
        """
        pass
    
    def learn(self, total_timesteps):
        while self.steps_done < total_timesteps:
            pass
    
    def train(self, ):
        pass
    
    def get_action(self, ):
        pass
    
    def store_transition(self, ):
        pass
    


if __name__ == '__main__':
    pass