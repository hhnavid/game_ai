import sys
sys.path.append('I:/projs/game-ai/src')

import time
import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import random
from datetime import datetime
from common.my_replay_buffer import ReplayBuffer
from common.utils import LinearSchedule
from common.net_param_manip import init_xavier_weights
from common.running_mean_std import *
from common.save_dict2csv import CSVLogger


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
    
    def __init__(self,
                 n_train_steps, n_rollout_steps, n_eval_steps,
                 total_timesteps,                 
                 env, env_name,
                 eval_env,
                 obs_dim, obs_dtype, 
                 action_dim, action_dtype, is_action_discrete,
                 hidden_layers=[64, 63], activation_=nn.ReLU,
                 normalize_obs=True, gamma=0.95, tau=1.0, lr=1e-4, bs=32,
                 exploration_fraction=0.1, exploration_initial_eps=1.0,
                 exploration_final_eps=0.05, replay_buff_size=10000,
                 explore_render=False, eval_render=False,
                 log_interval=20,
                 path_prefix=".\\results"):        
        """ 
        trains the DQN agent
        Args:
            n_train_steps (int): number of training steps before interacting with the env again
            n_rollout_steps (int): number of explorative steps taken in the env before 
                                   2 successive training sessions
            n_eval_steps (_type_): _description_
            total_timesteps (int): Total number of interaction steps performed in
                                   the env. After `total_timesteps`, learning ends.
            env (_type_): _description_
            env_name (_type_): _description_
            eval_env (_type_): _description_
            obs_dim (_type_): _description_
            obs_dtype (_type_): _description_
            action_dim (_type_): _description_
            action_dtype (_type_): _description_
            is_action_discrete (bool): true means the action space of the env is discrete
            hidden_layers (list, optional): _description_. Defaults to [64, 63].
            activation_ (_type_, optional): _description_. Defaults to nn.ReLU.
            normalize_obs (bool, optional): _description_. Defaults to True.
            gamma (float, optional): _description_. Defaults to 0.95.
            tau (float, optional): _description_. Defaults to 1.0.
            lr (_type_, optional): _description_. Defaults to 1e-4.
            bs (int, optional): _description_. Defaults to 32.
            exploration_fraction (float, optional): _description_. Defaults to 0.1.
            exploration_initial_eps (float, optional): _description_. Defaults to 1.0.
            exploration_final_eps (float, optional): _description_. Defaults to 0.05.
            replay_buff_size (int, optional): _description_. Defaults to 10000.
            explore_render (bool, optional): _description_. Defaults to False.
            eval_render (bool, optional): _description_. Defaults to False.
            log_interval (int): After every `log_interval` epochs of {perform rollouts, train, evaluate}, statistics are logged.
            path_prefix (str, optional): _description_. Defaults to ".\\results"
        """                                
        # the environment that the agent must learn
        self.env = env
        self.env_name = env_name
        self.explore_render = explore_render
        self.eval_env = eval_env
        self.eval_render = eval_render
        
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  
        run_name = "dqn_" + env_name + "_" + date_str
        self.save_path_prefix = os.path.join(path_prefix, run_name)        
        if not os.path.exists(self.save_path_prefix):
            os.makedirs(self.save_path_prefix)
            print('Create log dir: {}'.format(self.save_path_prefix))
        
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction  
        self.current_progress_remaining = 1.0      
        self.exploration_rate = 1.0
        
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction)
        
        self.last_obs = None
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_action_discrete = is_action_discrete
        fan_ins = [self.obs_dim] + hidden_layers + [self.action_dim]
        self.q_network = QNetwork(fan_ins, activation_, lr).to(self.device)
        
        self.target_network = QNetwork(fan_ins, activation_, lr).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        if is_action_discrete:
            # for discrete actions, only the chosen action is stored in the buffer
            action_shape_ = (1,) 
        else:
            action_shape_ = (action_dim,)
        self.replay_buffer = ReplayBuffer(limit=replay_buff_size,
                                          obs_shape=(obs_dim,),
                                          obs_dtype=obs_dtype, 
                                          action_shape=action_shape_, 
                                          action_dtype=action_dtype)            
        self.gamma = gamma
        self.batch_size = bs                                                
        
        # Observation normalization
        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.obs_rms = RunningMeanStdMPI(shape=(self.obs_dim,))
        else:
            self.obs_rms = None
        self.normalize_reward = False
        self.reward_rms = None
            
        self.n_train_steps = n_train_steps
        self.n_rollout_steps = n_rollout_steps
        self.n_eval_steps = n_eval_steps                
        self.total_timesteps = total_timesteps        
        self.log_interval = log_interval
                                
    
    def reset(self):
        pass
    
        
    def init_train_variables(self):                
        self.epochs_so_far = 0 # Total number of epochs performed so far
        self.episodes_so_far = 0  # Total number of explorative episodes performed so far
        self.steps_so_far = 0  # Total number of explorative steps performed so far
        
        # Sum of immediate rewards in 1 episode. When episode is done, it's saved in
        # `epoch_episode_rewards` list & `episode_rewards_history` queue. Next it is reset to 0
        self.episode_reward = 0.
        
        # Each item of this list contains sum of immediate rewards for 1 explorative episode        
        self.epoch_episode_rewards = []  
        
        # Number of explorative steps performed in 1 episode. When episode is done, it's saved in
        #  `epoch_self.episode_steps` list. Next it is reset to 0.
        self.episode_step = 0        
        
        combined_stats = {'rollout/return': 0.,
                          'eval/return': 0.,
                          'total/epochs': 0,
                          'total/episodes': 0,
                          'total/steps': 0,
                          'total/hours': 0.}
        # Logger
        self.logger = CSVLogger(os.path.join(self.save_path_prefix, 'results.csv'),
                                keys=combined_stats.keys(), mode='write')
        return
            
    
    def collect_rollout_steps(self):
        is_train_over = False
        obs = self.last_obs.copy()        
        
        for _ in range(self.n_rollout_steps):
            if self.steps_so_far >= self.total_timesteps: # Training is over so return
                is_train_over = True
                break
                                                    
            # Select action
            action = self.get_action(obs, deterministic=False)
            
            # Render env
            if self.explore_render and self.epochs_so_far % self.log_interval == 0:
                self.env.render()
                
            # Execute action
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            # terminated == true: episode ended naturally (goal state is reached)
            # truncated == true: episode ended due to exceeding time limit or other limits
            done = terminated or truncated                        

            # Update statistics
            self.steps_so_far += 1
            self.episode_reward += reward
            self.episode_step += 1
            
            # update exploration rate after each env step
            self.exploration_rate = self.exploration_schedule(self.current_progress_remaining)
            # update the current progress needed for the exploration schedule
            self.current_progress_remaining = 1.0 - float(self.steps_so_far) / float(self.total_timesteps)

            # Store observed transition
            self.store_transition(obs, action, reward, new_obs, done)                        
            obs = new_obs                        
            
            if done:
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.episode_step = 0
                self.episodes_so_far += 1

                # Episode done => Reset agent noises, reset environment
                self.reset()
                obs = self.env.reset()[0]
        self.last_obs = obs.copy()
        return is_train_over
        
        
    def train(self):
        epoch_actor_losses = []
        epoch_critic_losses = []
        epoch_adaptive_distances = []
        for t_train in range(self.n_train_steps):
            pass
        
        
    def train_step(self):
        self.replay_buffer.sample(batch_size=self.batch_size)
    
    
    def learn(self):        
        # Reset agent state (i.e. reset action/param noises)
        self.reset()
        
        # Reset env and set initial state (i.e obs)
        self.last_obs = self.env.reset()[0]
        
        # if eval_env is available, reset it too and set initial evaluation state (i.e.eval_obs)
        eval_obs = None
        if self.eval_env is not None:
            eval_obs = self.eval_env.reset()[0]
            if self.eval_render:
                self.eval_env.render()     # (for pyBullet env.) call before env.reset to show a window of the env.
            
            
        # Define statistics variables
        # ----------------------------                
        total_hours = 0.0
        eval_episode_reward = 0.0
        eval_episode_rewards = []  # Each item of this list contains sum of immediate rewards for 1 evaluation episode                        
        self.init_train_variables()        
        
        # The main learning loop. It ends when `self.total_steps >= total_timesteps`
        while True:
            # This is epoch loop, that every `log_interval` epochs is ended to update `combined_stats`
            for _ in range(self.log_interval):
                epoch_start_time = time.time()
                self.epochs_so_far += 1
                
                self.collect_rollout_steps()                
                self.train()                
                                
         
    def get_action(self, obs, deterministic):
        """
        Select an action based on the given state 'obs'
        Args:
            obs (torch tensor): input state(s) [bs x obs_dim]            
        returns:
            action (np array): one of the |action_dim| discrete actions [bs x 1]
        """
        # normalize obs
        norm_obs = Tensor(normalize(obs, self.obs_rms)).to(self.device)
        
        if deterministic:
            # Greedy action selection
            q_values = self.q_network(norm_obs)
            action = q_values.argmax(dim=1).reshape(-1)
        else:
            if np.random.rand() < self.exploration_rate:
                # Select one of the discrete actions randomly
                action = np.random.choice(self.action_dim)            
            else:
                # Greedy action selection
                q_values = self.q_network(norm_obs)
                action = q_values.argmax(dim=1).reshape(-1)
                action = action.cpu().data.numpy()
        return action
    
    def store_transition(self, curr_obs, action, reward, next_obs, done):
        """
        Stores a transitions in replay buffer
        :param curr_state:[state_dim,]
        :param action: [action_dim,]
        :param reward: [1,]
        :param next_state: [state_dim,]
        :param done: [1,]
        :return:
       """        
        self.replay_buffer.append(curr_obs, action, reward, next_obs, done)

        # Running avg/std update
        if self.normalize_obs:
            self.obs_rms.update( np.array([curr_obs]) )
        if self.normalize_reward:
            self.reward_rms.update(np.array([reward]))
    


