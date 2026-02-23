import torch
from common.net_param_manip import init_xavier_weights


class Actor(torch.nn.Module):

    def __init__(self, fan_ins, activation, learning_rate, layer_norm=False):
        """
        Actor class for DDPG
        :param fan_ins: list containing input dimension for each net. layer. 1st item in the list
        must be state_dim and last item must be action_dim
        :param activation: Network activation function {torch.relu, torch.tanh, ...}
        :param learning_rate:
        """
        super(Actor, self).__init__()

        self.activation = activation
        self.layer_norm = layer_norm

        # Create network layers
        self.layers = torch.nn.Sequential()
        layer_items = []
        for i in range(len(fan_ins) - 1):
            layer_items.append(('Linear{}'.format(i), torch.nn.Linear(fan_ins[i], fan_ins[i + 1])))
            if self.layer_norm and i < len(fan_ins) - 2:
                layer_items.append(('LayerNorm{}'.format(i), torch.nn.LayerNorm(fan_ins[i + 1])))
        [self.layers.add_module(name, layer) for name, layer in layer_items]

        # Init network weights
        self.apply(init_xavier_weights)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, states):
        """
        According to Gym DDPG, we always assume actor output is in interval [-1, 1].
        I think that's why we don't use runningAvgStd normalization on action when
        it is input to critic network
        :param states:
        :return:
        """
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
        out = self.layers[-1](out)
        return torch.tanh(out), out     # action scaling using max_action is done in ddpg::train()


class Critic(torch.nn.Module):

    def __init__(self, fan_ins, action_dim, activation, learning_rate, layer_norm=False, feed_action_to_1st_layer=False):
        """
        Critic class for DDPG
        :param fan_ins: list containing input dimension for each net. layer. 1st item in the list
        must be state_dim and last item must be 1
        :param action_dim: action space dimension
        :param activation: Network activation function {torch.relu, torch.tanh, ...}
        :param learning_rate:
        """
        super(Critic, self).__init__()

        # Since action is not included until 2nd layer, we need at least 2 layers
        # in critic network
        assert len(fan_ins) >= 3

        self.activation = activation
        self.layer_norm = layer_norm
        self.feed_action_to_1st_layer = feed_action_to_1st_layer

        # Create network layers
        self.layers = torch.nn.Sequential()
        layer_items = []
        for i in range(len(fan_ins) - 1):
            if (self.feed_action_to_1st_layer and i == 0) or (not self.feed_action_to_1st_layer and i == 1):
                f_in = fan_ins[i] + action_dim
            else:
                f_in = fan_ins[i]
            layer_items.append(('Linear{}'.format(i),torch.nn.Linear(f_in, fan_ins[i + 1])))
            if self.layer_norm and i < len(fan_ins) - 2:
                layer_items.append(('LayerNorm{}'.format(i), torch.nn.LayerNorm(fan_ins[i + 1])))
        [self.layers.add_module(name, layer) for name, layer in layer_items]

        # Init network weights
        self.apply(init_xavier_weights)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, states, actions):
        """
        Computes Q(states, actions).
        :param states: Batch of states [batch_size x state_dim]
        :param actions: Batch of actions [batch_size x action_dim]
        :return:
        """
        # 1st layer
        if self.feed_action_to_1st_layer:
            out = torch.cat([states, actions], 1)
            i = 0
        else:
            out = self.layers[0](states)        # Linear layer: Wx+b
            i = 1
            if self.layer_norm:
                out = self.layers[i](out)       # Layer normalization
                i += 1
            out = self.activation(out)          # Activation function

            # Hidden layers
            out = torch.cat([out, actions], 1)  # According to DDPG, actions is included in 2nd layer
        while i < len(self.layers) - 1:
            out = self.layers[i](out)       # Linear layer: Wx+b
            i += 1
            if self.layer_norm:
                out = self.layers[i](out)   # Layer normalization
                i += 1
            out = self.activation(out)      # activation function

        # Last layer doesn't need layer normalization or activation
        out = self.layers[-1](out)          # Wx+b
        return out

