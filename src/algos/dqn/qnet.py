import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from common.replay_buffer import ReplayBuffer
from common.net_param_manip import init_xavier_weights


class QNetwork(nn.Module):
    
    def __init__(self, fan_ins, activation, learning_rate):
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

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            
    
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
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, gamma=0.99, lr=1e-3, batch_size=64,
                 replay_buffer_size=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        fan_ins = [self.state_dim] + hidden_layers + [self.action_dim]
        self.q_network = QNetwork(fan_ins, activation, learning_rate).to(self.device)
        
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        
        self.epsilon = epsilon_start
        self.action_dim = action_dim
        self.steps_done = 0

    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = self.epsilon_final + \
            (self.epsilon_start - self.epsilon_final) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)

        if random.random() < self.epsilon:
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
