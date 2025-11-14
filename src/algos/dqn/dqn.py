import sys
sys.path.append('I:/projs/game-ai/src')

import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import random
from datetime import datetime
from common.my_replay_buffer import ReplayBuffer
from common.utils import LinearSchedule
from common.net_param_manip import init_xavier_weights, soft_update
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
        
        # layer norm is disabled for now
        self.layer_norm = False

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
                 env, env_name, env_type,
                 eval_env,
                 obs_dim, obs_dtype, 
                 action_dim, action_dtype, is_action_discrete,
                 hidden_layers=[64, 63], activation_=nn.ReLU,
                 normalize_obs=True, gamma=0.94, tau=0.93, lr=1e-4, bs=32,
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
        self.env_type = env_type # needed to address the differences between gym envs and codeArt ones
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
        self.exploration_rate = self.exploration_initial_eps
        
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
        
        self.target_q_network = QNetwork(fan_ins, activation_, lr).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

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
        self.tau = tau
        self.batch_size = bs                                                
        
        # Observation normalization
        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.obs_rms = RunningMeanStdMPI(shape=(self.obs_dim,))
            self.obs_rms_history = {'mean': [], 'std': []}
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
        
        # Each item of this list contains sum of immediate rewards for 1 evaluation episode                        
        self.eval_episode_rewards = []  
        
        # loss values during the training process
        self.q_losses = []
        
        # Number of explorative steps performed in 1 episode. When episode is done, it's saved in
        #  `epoch_self.episode_steps` list. Next it is reset to 0.
        self.episode_step = 0        
        
        self.combined_stats = {'rollout/return': [],
                               'eval/return': [],
                               'total/epochs': [],
                               'total/episodes': [],
                               'total/steps': [],
                               'total/hours': []}
        # Logger
        self.logger = CSVLogger(os.path.join(self.save_path_prefix, 'results.csv'),
                                keys=self.combined_stats.keys(), mode='write')
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
            # if self.explore_render and self.epochs_so_far % self.log_interval == 0:
                # self.env.render()
                
            # Execute action
            new_obs, reward, terminated, timed_out, info = self.env.step(action)
            # terminated == true: episode ended naturally (goal state is reached)
            # truncated == true: episode ended due to exceeding time limit or other limits
            print("terminate: {}, timed_out: {}".format(terminated, timed_out))
            done = terminated or timed_out

            # Update statistics
            self.steps_so_far += 1
            self.episode_reward += reward
            self.episode_step += 1
            
            # update exploration rate after each env step
            self.exploration_rate = self.exploration_schedule(self.current_progress_remaining)
            # print('exploration rate: {}'.format(self.exploration_rate))
            # update the current progress needed for the exploration schedule
            self.current_progress_remaining = 1.0 - float(self.steps_so_far) / float(self.total_timesteps)

            # Store observed transition
            self.store_transition(obs, action, reward, new_obs, done)                        
            obs = new_obs                        
            
            if done:
                print('Episode done, steps so far: {}, episode reward: {}, exploration rate: {}'.format(
                    self.steps_so_far, self.episode_reward, self.exploration_rate))
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.episode_step = 0
                self.episodes_so_far += 1

                # Episode done => Reset agent noises, reset environment
                self.reset()
                obs, _ = self.env.reset()
        self.last_obs = obs.copy()
        return is_train_over
        
        
    def train(self):                               
        for _ in range(self.n_train_steps):            
            qloss = self.train_step()
            self.q_losses.append(qloss)        
            
            # update the target network
            soft_update(target=self.target_q_network,
                        source=self.q_network,
                        tau=self.tau)
        print("trained for {} steps".format(self.n_train_steps))
        
    def train_step(self):
        batch = self.replay_buffer.sample(self.batch_size)
        
        # update critic
        self.q_network.zero_grad()        
        qloss = self.q_loss(batch)
        qloss.backward()
        self.optimizer.step()
        return qloss.detach().cpu().data.numpy()
        
        
    def q_loss(self, batch):
        prev_states = batch['prev_states']
        next_states = batch['next_states']
        rewards = batch['rewards']
        actions = batch['actions']
        done = batch['done']
        
        norm_obs0 = Tensor(normalize(prev_states, self.obs_rms)).to(self.device)
        norm_obs1 = Tensor(normalize(next_states, self.obs_rms)).to(self.device)
        rewards = Tensor(rewards).to(self.device)
        actions = Tensor(actions).to(self.device)
        done = Tensor(done).to(self.device)
        
        predicted_q = self.q_network(norm_obs0) # [bs x actionDim]
        # predicted_q = predicted_q[actions.to(torch.int32)] # [bs x 1]
        predicted_q = predicted_q.gather(1, actions.to(torch.int64))
        target_q = self.target_q_network(norm_obs1)        
        max_target_q, _ = torch.max(target_q, dim=1) # [bs x 1]
        max_target_q = max_target_q.reshape(-1, 1)
        desired_q = rewards + (1. - done) * self.gamma * max_target_q # Only use target_q if obs1 isn't a terminal state
        desired_q = desired_q.detach()
        return F.mse_loss(predicted_q, desired_q)
    
    
    def learn(self):        
        # Reset agent state (i.e. reset action/param noises)
        self.reset()
        
        # Reset env and set initial state (i.e obs)
        self.last_obs, _ = self.env.reset()
        
        # if eval_env is available, reset it too and set initial evaluation state (i.e.eval_obs)
        eval_obs = None
        if self.eval_env is not None:
            eval_obs, _ = self.eval_env.reset()
            # if self.eval_render:
            #     self.eval_env.render()     # (for pyBullet env.) call before env.reset to show a window of the env.
            
            
        # Define statistics variables
        # ----------------------------                
        total_hours = 0.0
        eval_episode_reward = 0.0
        
        self.init_train_variables()        
        
        # The main learning loop. It ends when `self.total_steps >= total_timesteps`
        while True:
            # This is epoch loop, that every `log_interval` epochs is ended to update `combined_stats`
            for _ in range(self.log_interval):
                epoch_start_time = time.time()
                self.epochs_so_far += 1
                
                self.collect_rollout_steps()                
                self.train()                                
            
            # Log statistics
            if len(self.epoch_episode_rewards) > 0:
                self.combined_stats['total/epochs'].append(self.epochs_so_far)
                self.combined_stats['total/episodes'].append(self.episodes_so_far)
                self.combined_stats['total/steps'].append(self.steps_so_far)
                self.combined_stats['rollout/return'].append(np.mean(self.epoch_episode_rewards))            

            if self.eval_env is not None and len(self.eval_episode_rewards) > 0:
                self.combined_stats['eval/return'].append(np.mean(self.eval_episode_rewards))

            if len(self.combined_stats['rollout/return']) > 0:
                self.plot_return_vs_step(self.combined_stats)
                self.print_stats()
            if self.obs_rms is not None:
                self.plot_running_avg('obs')
                                
         
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
                action = q_values.argmax(dim=0).reshape(-1)
                action = action.cpu().data.numpy().astype(np.int64)[0]
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
            self.obs_rms_history['mean'].append(np.mean(self.obs_rms.mean))
            self.obs_rms_history['std'].append(np.mean(self.obs_rms.std))
        if self.normalize_reward:
            self.reward_rms.update(np.array([reward]))
            
            
    def plot_running_avg(self, running_avg_name='obs'):
    
        if running_avg_name == 'obs':
            m = np.array(self.obs_rms_history['mean'])
            sig = np.array(self.obs_rms_history['std'])
        else:
            m = np.array(self.reward_rms_history['mean'])
            sig = np.array(self.reward_rms_history['std'])
        x = np.arange(0, m.size) / 1000.0
    
        plt.figure("Running mean/std")
        plt.title('Running mean/std', fontsize=10)
        plt.plot(x, m, 'green')
        plt.fill_between(x, m - sig, m + sig, alpha=0.3, facecolor='green', linewidth=0)
        plt.xlabel('Update step', fontsize=8)
        plt.ylabel('mean/std', fontsize=8)
        plt.savefig(
            os.path.join(
                self.save_path_prefix,
                running_avg_name + '_rms_' + self.env_name + '.png'))
        plt.close()
        
        
    def plot_return_vs_step(self, combined_stats):

        steps = np.array(combined_stats['total/steps']) / 1000.0
        returns = combined_stats['rollout/return']
        eval_returns = combined_stats['eval/return']
        fig_title = 'episode_reward_vs_Step'
        plt.figure(fig_title)
        plt.title('Episode reward vs step')
        plt.plot(steps, returns, label='rollout ret.')
        if self.eval_env is not None and len(eval_returns) > 0:
            plt.plot(steps, eval_returns, label='eval. ret.')
        plt.xlabel('Steps (thousands)')
        plt.ylabel('Episode reward average')
        plt.grid()
        plt.legend()
        plt.savefig(
            os.path.join(self.save_path_prefix,
                         fig_title + self.env_name + '.png'))
        plt.close()        
    
    def print_stats(self):
        print('----------------------------------------\n')
        for key, value in self.combined_stats.items():
            if type(value) == list and len(value) > 0:
                print('{0}: {1}'.format(key, value[-1]))
            else:
                print('{0}: {1}'.format(key, value))
    


