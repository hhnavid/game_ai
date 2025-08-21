import torch
import torch.nn as nn
import numpy as np
import random
from common.replay_buffer import ReplayBuffer
from common.net_param_manip import init_xavier_weights


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
        self.net = nn.Sequential()
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
    
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 63], activation_,
                 gamma=0.95, lr=1e-4, bs=32, replay_buff_size=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):                
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        fan_ins = [self.state_dim] + hidden_layers + [self.action_dim]
        self.q_network = QNetwork(fan_ins, activation_, lr).to(self.device)
        
        self.target_network = QNetwork(fan_ins, activation_, lr).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buff_size)
        self.gamma = gamma
        self.batch_size = bs
        
        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_decay = epsilon_decay
        
        self.eps = epsilon_start
        self.action_dim = action_dim
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        self.eps = self.eps_final + (self.eps_start - self.eps_final) * np.exp(-1. * self.steps_done / self.eps_decay)

        if random.random() < self.eps:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.max(1)[1].item()
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.q_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_state).max(1)[0]
            target = reward + self.gamma * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
