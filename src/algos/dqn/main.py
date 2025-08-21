from qnet import DQN


if __name__ == '__main__':
    
    # setup env
    
    
    # setup DQN agent
    dqn = DQN(state_dim, action_dim,
              hidden_layers=[64, 63], activation_,
              gamma=0.95, lr=1e-4, bs=32, replay_buff_size=10000, 
              epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500)