import sys

sys.path.append("I:/projs/game-ai/src")

import copy
import os
import torch
import random
import time
import pickle
import shutil
import ntpath
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from datetime import datetime

from common.running_mean_std import *
from common.utils import (
    set_global_seeds,
    get_random_generators_state,
    set_random_generators_state,
)
from common.save_dict2csv import CSVLogger
from common.memory_utils import mem_usage_in_mb
from common.net_param_manip import init_xavier_weights, soft_update, hard_update

from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG(object):
    def __init__(
        self,
        n_train_steps,
        n_rollout_steps,
        n_eval_steps,
        total_timesteps,
        env,
        env_name,
        env_type,
        eval_env,
        obs_dim,
        action_dim,
        max_action,
        action_noise,
        actor_hidden_layers=[64, 63],
        actor_activation=nn.ReLU,
        critic_hidden_layers=[64, 63],
        critic_activation=nn.ReLU,
        layer_normalization=False,
        normalize_obs=True,
        normalize_reward=False,
        gamma=0.94,
        tau=0.93,
        actor_lr=1e-4,
        critic_lr=1e-4,
        bs=32,
        replay_buff_size=10000,
        explore_render=False,
        eval_render=False,
        log_interval=20,
        return_plot=False,
        obs_rms_plot=False,
        path_prefix=".\\results",
        resume=False,
        checkpoint_every_n_epoch=100,
        resume_path_prefix=None,
    ):
        self.seed = random.randint(0, 2**32 - 1)
        
        # the environment that the agent must learn
        self.env_type = env_type  # needed to address the differences between gym envs and codeArt ones
        self.env_name = env_name
        self.env = env
        self.eval_env = eval_env

        self.return_plot = return_plot
        self.obs_rms_plot = obs_rms_plot

        self.explore_render = explore_render
        self.eval_render = eval_render

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = "ddpg_" + env_name + "_" + date_str
        if resume_path_prefix is None:
            self.save_path_prefix = os.path.join(path_prefix, run_name)
            if not os.path.exists(self.save_path_prefix):
                os.makedirs(self.save_path_prefix)
                print("Create log dir: {}".format(self.save_path_prefix))
        else:
            self.save_path_prefix = resume_path_prefix
            print("using existing log dir {}".format(resume_path_prefix))
        self.env.log_path_prefix = self.save_path_prefix

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)

        self.last_obs = None
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action

        actor_fan_ins = [self.obs_dim] + actor_hidden_layers + [self.action_dim]
        critic_fan_ins = [self.obs_dim] + critic_hidden_layers + [1]

        # Actor
        self.actor = Actor(
            actor_fan_ins, actor_activation, actor_lr, layer_normalization
        ).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)        

        # Critic
        self.critic = Critic(
            critic_fan_ins,
            self.action_dim,
            critic_activation,
            critic_lr,
            layer_normalization,
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)        

        # Make sure actor/critic and its target have the same initial weights
        hard_update(target=self.target_critic, source=self.critic)
        hard_update(target=self.target_actor, source=self.actor)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            limit=replay_buff_size,
            action_shape=(self.action_dim,),
            observation_shape=(self.obs_dim,),
        )

        # Hyper-parameters
        self.tau = tau
        self.gamma = gamma
        self.batch_size = bs

        # Observation normalization
        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.obs_rms = RunningMeanStdMPI(shape=(self.obs_dim,))
            self.obs_rms_history = {"mean": [], "std": []}
        else:
            self.obs_rms = None

        # Reward normalization
        self.normalize_reward = normalize_reward
        if self.normalize_reward:
            self.reward_rms = RunningMeanStdMPI(shape=(1,))            
        else:
            self.reward_rms = None

        # Action space exploration noise
        self.action_noise = action_noise

        self.n_train_steps = n_train_steps
        self.n_rollout_steps = n_rollout_steps
        self.n_eval_steps = n_eval_steps
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval

        # Resume stuff
        self.model_config = {}
        self.resume = resume
        self.checkpoint_path = os.path.join(self.save_path_prefix, "checkpoint")
        self.checkpoint_every_n_epoch = checkpoint_every_n_epoch

    def init_train_variables(self):
        self.combined_stats = {
            "rollout/return": 0.0,
            "eval/return": 0.0,
            "total/epochs": 0,
            "total/episodes": 0,
            "total/steps": 0,
            "total/hours": 0.0,
            "total/mem_usage": 0,
            "train/critic_loss": 0.0,
            "train/actor_loss": 0.0,
        }
        # Logger
        self.logger = CSVLogger(
            os.path.join(self.save_path_prefix, "results.csv"),
            keys=self.combined_stats.keys(),
            mode="write",
        )

        if self.resume:
            print("Resuming...")
            self.resume_checkpoint()
            print("Resuming OK.")
        else:
            self.epochs_so_far = 0  # Total number of epochs performed so far
            self.episodes_so_far = (
                0  # Total number of explorative episodes performed so far
            )
            self.steps_so_far = 0  # Total number of explorative steps performed so far
            self.total_hours = 0.0

            # Sum of immediate rewards in 1 episode. When episode is done, it's saved in
            # `epoch_episode_rewards` list & `episode_rewards_history` queue. Next it is reset to 0
            self.episode_reward = 0.0
            # Number of explorative steps performed in 1 episode. When episode is done, it's saved in
            #  `epoch_episode_steps` list. Next it is reset to 0.
            self.episode_step = 0
            self.epoch_episode_rewards = (
                []
            )  # Each item of this list contains sum of immediate rewards for 1 explorative episode
            self.epoch_episode_steps = (
                []
            )  # Each item of this list contains number of steps performed in 1 explorative episode

            self.eval_episode_reward = 0.0
            # Each item of this list contains sum of immediate rewards for 1 evaluation episode
            self.eval_episode_rewards = []

            # Number of explorative steps performed in 1 episode. When episode is done, it's saved in
            #  `self.epoch_episode_steps` list. Next it is reset to 0.
            self.episode_step = 0
        # loss values during the training process
        self.critic_losses = []
        self.actor_losses = []
        return

    def collect_rollout_steps(self):
        is_train_over = False
        obs = self.last_obs.copy()

        for rollout_step in range(self.n_rollout_steps):
            if self.steps_so_far >= self.total_timesteps:  # Training is over so return
                is_train_over = True
                break
            
            if self.steps_so_far < 1000:
                # select random actions                                
                action = np.random.uniform(-1, 1, self.action_dim)
                # print('random action: {}'.format(action))
            else:
                # Select action
                action = self.get_action(obs, apply_noise=True)

            # Render env
            # if self.explore_render and self.epochs_so_far % self.log_interval == 0:
            # self.env.render()

            # Execute action
            new_obs, reward, terminated, timed_out, info = self.env.step(
                action * self.max_action
            )  # DDPG assumes [-1,1] action range => rescale if needed
            # terminated == true: episode ended naturally (goal state is reached)
            # truncated == true: episode ended due to exceeding time limit or other limits
            done = terminated or timed_out
            
            # the following if is just for debug purposes
            # if rollout_step % 10 == 0:
            #     self.env.pause_()
            #     input("Env. paused, press Enter to continue...")
            #     self.env.play_()

            # Update statistics
            self.steps_so_far += 1
            self.episode_reward += reward
            self.episode_step += 1

            # Store observed transition
            self.store_transition(obs, action, reward, new_obs, done)
            obs = new_obs

            if done:
                print(
                    "Episode done, steps so far: {}, episode reward: {:.3f}".format(
                        self.steps_so_far, self.episode_reward
                    )
                )
                self.epoch_episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.episode_step = 0
                self.episodes_so_far += 1

                # Episode done => reset environment
                obs, _ = self.env.reset()
                self.reset()
        self.last_obs = obs.copy()
        return is_train_over

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()

    def get_action(self, state, apply_noise):
        """
        :param state: [1 x state_dim]
        :param apply_noise: If True, explorative noise is added
        :return: actor output
        """
        # Normalize state
        norm_state = Tensor(normalize(state, self.obs_rms)).to(device)

        # Select action
        action, _ = self.actor(norm_state)
        action = action.cpu().data.numpy()  # << detach removed here

        # Apply action noise
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action.shape
            action += noise
            action = np.clip(action, -1.0, 1.0)
        return action

    def store_transition(self, curr_state, action, reward, next_state, done):
        """
        Stores a transitions in replay buffer
        :param curr_state:[state_dim,]
        :param action: [action_dim,]
        :param reward: [1,]
        :param next_state: [state_dim,]
        :param done: [1,]
        :return:
        """
        # TODO: reward scale should be added here.

        self.replay_buffer.append(curr_state, action, reward, next_state, done)

        # Running avg/std update
        if self.normalize_obs:
            self.obs_rms.update(np.array([curr_state]))
            self.obs_rms_history["mean"].append(np.mean(self.obs_rms.mean))
            self.obs_rms_history["std"].append(np.mean(self.obs_rms.std))
        if self.normalize_reward:
            self.reward_rms.update(np.array([reward]))

    def critic_loss(self, batch):
        """
        :param curr_states: (Tensor) [batch_size x state_dim]
        :param actions: (Tensor) [batch_size x action_dim]
        :param rewards: [batch_size x 1]
        :param next_states: (Tensor) [batch_size x state_dim]
        :param gamma: Discount factor
        :return:
        """
        prev_states = batch["prev_states"]
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        actions = batch["actions"]
        done = batch["done"]

        norm_obs0 = Tensor(normalize(prev_states, self.obs_rms)).to(device)
        norm_obs1 = Tensor(normalize(next_states, self.obs_rms)).to(device)
        rewards = Tensor(rewards).to(device)
        actions = Tensor(actions).to(device)
        done = Tensor(done).to(device)

        values = self.critic(norm_obs0, actions)

        # TODO: L2 regularization

        target_action, _ = self.target_actor(norm_obs1)
        target_q = self.target_critic(norm_obs1, target_action)
        desired_values = (
            rewards + (1.0 - done) * self.gamma * target_q
        )  # Only use target_q if obs1 isn't a terminal state
        desired_values = desired_values.detach()

        return F.mse_loss(values, desired_values)

    def actor_loss(self, curr_states):

        # self.critic_with_actor_tf = denormalized( clipByValue(  Q(normalize(s), mu(normalize(s)))  )
        # self.actor_loss: -(1/N) * sum_i=1:N{ Q[normalize(s), mu(normalize(s))] }
        norm_obs0 = Tensor(normalize(curr_states, self.obs_rms)).to(device)
        action, _ = self.actor(norm_obs0)
        return -torch.mean(self.critic(norm_obs0, action))

    def train_step(self):
        batch = self.replay_buffer.sample(batch_size=self.batch_size)

        # Update critic   
        self.critic.zero_grad()     
        self.critic.optimizer.zero_grad()
        critic_loss = self.critic_loss(batch)
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update actor
        self.actor.zero_grad()
        self.actor.optimizer.zero_grad()
        actor_loss = self.actor_loss(batch["prev_states"])
        actor_loss.backward()
        self.actor.optimizer.step()

        return (
            critic_loss.detach().cpu().data.numpy(),
            actor_loss.detach().cpu().data.numpy(),
        )    

    def create_resume_checkpoint(self):
        """
        Create checkpoint so that we can resume an interrupted learning process from it later
        """
        print(
            "<<<<<<<<Creating checkpoint at step: {}>>>>>>>>".format(self.steps_so_far)
        )
        # Log models
        model_path = os.path.join(
            self.save_path_prefix, self.env_name + "_actor_critic.pth"
        )
        models_dict = {
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "critic_optim_state_dict": self.critic.optimizer.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "actor_optim_state_dict": self.actor.optimizer.state_dict(),
        }
        torch.save(models_dict, model_path)

        self.model_config["random_seed"] = self.seed
        if self.obs_rms is not None:
            # Log running avg
            self.model_config["obs_rms._sum"] = self.obs_rms._sum
            self.model_config["obs_rms._sumsq"] = self.obs_rms._sumsq
            self.model_config["obs_rms._count"] = self.obs_rms._count
        if self.reward_rms is not None:
            self.model_config["reward_rms._sum"] = self.reward_rms._sum
            self.model_config["reward_rms._sumsq"] = self.reward_rms._sumsq
            self.model_config["reward_rms._count"] = self.reward_rms._count

        # Log models config
        pickle_path = os.path.join(
            self.save_path_prefix, self.env_name + "_models_config.pickle"
        )
        with open(pickle_path, "wb") as handle:
            pickle.dump(self.model_config, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Create checkpoint dir.
        checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Combined stats is already stored in `[env]_results_.csv` => just make a copy in checkpoint dir.
        combined_stats_file_name = ntpath.basename(
            self.logger.save_path
        )  # Get the combined_stats csv file name
        shutil.copyfile(
            src=self.logger.save_path,
            dst=os.path.join(checkpoint_path, combined_stats_file_name),
        )

        # Replay buffer checkpoint
        self.replay_buffer.save(checkpoint_path)

        # pseudo random generator
        torch_rnd_state, np_rnd_state, py_rnd_state, torch_cuda_rnd_state = (
            get_random_generators_state(torch.cuda.is_available())
        )

        # learn method variables checkpoint
        learn_method_vars = {
            "total_epochs": self.epochs_so_far,
            "total_episodes": self.episodes_so_far,
            "total_steps": self.steps_so_far,
            "total_hours": self.total_hours,
            "eval_episode_reward": self.eval_episode_reward,
            "eval_episode_rewards": self.eval_episode_rewards,
            "episode_reward": self.episode_reward,
            "episode_step": self.episode_step,
            "epoch_episode_rewards": self.epoch_episode_rewards,
            "epoch_episode_steps": self.epoch_episode_steps,
            "torch_rnd_state": torch_rnd_state,
            "np_rnd_state": np_rnd_state,
            "py_rnd_state": py_rnd_state,
            "torch_cuda_rnd_state": torch_cuda_rnd_state
        }
        pickle_path = os.path.join(checkpoint_path, "learn_method_vars.pickle")
        with open(pickle_path, "wb") as handle:
            pickle.dump(learn_method_vars, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def resume_checkpoint(self):
        """
        Resumes learner state from a previously saved checkpoint
        """
        checkpoint_path = self.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise Exception("Invalid checkpoint path!")

        # learn method variables checkpoint
        pickle_path = os.path.join(checkpoint_path, "learn_method_vars.pickle")
        with open(pickle_path, "rb") as handle:
            learn_method_vars = pickle.load(handle)

        # Resume state of pseudo random generators
        set_random_generators_state(
            learn_method_vars["torch_rnd_state"],
            learn_method_vars["np_rnd_state"],
            learn_method_vars["py_rnd_state"],
            learn_method_vars["torch_cuda_rnd_state"]
        )

        # Load Actor/Critic, dyn. model params
        models_path = os.path.join(
            self.save_path_prefix, self.env_name + "_actor_critic.pth"
        )
        models = torch.load(models_path, map_location=self.device)
        self.critic.load_state_dict(models["critic_state_dict"])
        self.actor.load_state_dict(models["actor_state_dict"])
        self.target_critic.load_state_dict(models["target_critic_state_dict"])
        self.target_actor.load_state_dict(models["target_actor_state_dict"])
        self.critic.optimizer.load_state_dict(models["critic_optim_state_dict"])
        self.actor.optimizer.load_state_dict(models["actor_optim_state_dict"])

        # Load replay buffer content
        self.replay_buffer.load(checkpoint_path)

        # Load models configs
        pickle_path = os.path.join(
            self.save_path_prefix, self.env_name + "_models_config.pickle"
        )
        with open(pickle_path, "rb") as handle:
            model_configs = pickle.load(handle)
            
            self.seed = model_configs["random_seed"]
            print("resuming seed to {}".format(self.seed))
                         
            # Running avg/std
            transition = self.replay_buffer.sample(1)
            if "obs_rms.mean" in model_configs and self.obs_rms is not None:
                self.obs_rms._sum = model_configs["obs_rms._sum"]
                self.obs_rms._sumsq = model_configs["obs_rms._sumsq"]
                self.obs_rms._count = model_configs["obs_rms._count"]
                self.obs_rms.update(
                    transition["prev_states"]
                )  # To set obs_rms.mean & std
            if "reward_rms.mean" in model_configs and self.reward_rms is not None:
                self.reward_rms._sum = model_configs["reward_rms._sum"]
                self.reward_rms._sumsq = model_configs["reward_rms._sumsq"]
                self.reward_rms._count = model_configs["reward_rms._count"]
                self.reward_rms.update(
                    transition["rewards"]
                )  # To set reward_rms.mean & std

        # Replace combined stats csv file from existing checkpoint
        combined_stats_file_name = ntpath.basename(
            self.logger.save_path
        )  # Get the combined_stats csv file name
        shutil.copyfile(
            src=os.path.join(checkpoint_path, combined_stats_file_name),
            dst=self.logger.save_path,
        )
        self.epochs_so_far = learn_method_vars["total_epochs"]
        self.episodes_so_far = learn_method_vars["total_episodes"]
        self.steps_so_far = learn_method_vars["total_steps"]
        self.total_hours = learn_method_vars["total_hours"]
        self.eval_episode_reward = learn_method_vars["eval_episode_reward"]
        self.eval_episode_rewards = learn_method_vars["eval_episode_rewards"]
        self.episode_reward = learn_method_vars["episode_reward"]
        self.episode_step = learn_method_vars["episode_step"]
        self.epoch_episode_rewards = learn_method_vars["epoch_episode_rewards"]
        self.epoch_episode_steps = learn_method_vars["epoch_episode_steps"]
        return

    def train(self):
        self.actor_losses = []
        self.critic_losses = []
        for _ in range(self.n_train_steps):
            critic_loss, actor_loss = self.train_step()
            self.critic_losses.append(critic_loss)
            self.actor_losses.append(actor_loss)

            # Update target networks
            soft_update(target=self.target_actor, source=self.actor, tau=self.tau)
            soft_update(target=self.target_critic, source=self.critic, tau=self.tau)

        print("trained for {} steps".format(self.n_train_steps))

    def evaluate(self):
        print("Performing evaluation steps>>>>>>>>>>>>>>>>>>>>")
        if self.env_type == "gym":            
            obs, _ = self.eval_env.reset()            
            done = False
            while not done:            
                action = self.get_action(obs, apply_noise=False)
                new_obs, reward, terminated, timed_out, info = self.eval_env.step(
                    action * self.max_action
                )
                done = terminated or timed_out
                # update statistics
                self.eval_episode_reward += reward                            
                obs = new_obs
            # eval episode is over:
            print("eval episode reward: {}".format(self.eval_episode_reward))
            self.eval_episode_rewards.append(self.eval_episode_reward)             
            self.eval_episode_reward = 0.                            
        elif self.env_type == "codeArt":
            eval_steps = 0
            obs, _ = self.env.reset()            
            done = False
            while not done and eval_steps < self.n_eval_steps:
                action = self.get_action(obs, apply_noise=False)
                new_obs, reward, terminated, timed_out, info = self.env.step(
                    action * self.max_action
                )
                done = terminated or timed_out
                # update statistics
                self.eval_episode_reward += reward                            
                obs = new_obs
                eval_steps += 1
            # eval episode is over:
            print("eval episode reward: {}".format(self.eval_episode_reward))
            self.eval_episode_rewards.append(self.eval_episode_reward)             
            self.eval_episode_reward = 0.                                        
        print("Evaluation steps finished>>>>>>>>>>>>>>>>>>>>")            

    def learn(self):                          
        # We assume symmetric actions.                                        
        assert np.all(
                np.abs(self.env.action_space.low) == self.env.action_space.high
            )

        self.last_obs, _ = self.env.reset()  # Reset env & set initial state (i.e obs)
        self.reset()  # Reset agent's action/param noises                    

        self.init_train_variables()  # define statistics variables
        
        # set the random seed        
        print("using random seed: {}".format(self.seed))
        set_global_seeds(self.seed, torch.cuda.is_available())

        # if self.env_type == "gym":
        #     self.env.seed(self.seed)
        #     if self.eval_env is not None:
        #         self.eval_env.seed(self.seed)

        # The main learning loop
        training_done = False
        while not training_done:
            # This is epoch loop, that every `log_interval` epochs is ended to update `combined_stats`
            for _ in range(self.log_interval):
                print(
                    "epochs so far: {}, episodes so far: {}, steps so far: {}".format(
                        self.epochs_so_far, self.episodes_so_far, self.steps_so_far
                    )
                )
                epoch_start_time = time.time()
                self.epochs_so_far += 1

                training_done = self.collect_rollout_steps()

                # pause env state during the training steps
                if self.env_type == "codeArt":
                    self.env.pause_()

                self.train()

                # resume env state after training
                if self.env_type == "codeArt":
                    self.env.play_()

                epoch_end_time = time.time()
                self.total_hours += (epoch_end_time - epoch_start_time) / 3600

                # Perform evaluation steps
                print("Skipping evaluation.............................................")
                # self.evaluate()

            # Log statistics
            if len(self.epoch_episode_rewards) > 0:
                self.combined_stats["rollout/return"] = np.mean(
                    self.epoch_episode_rewards
                )
                self.combined_stats["total/epochs"] = self.epochs_so_far
                self.combined_stats["total/episodes"] = self.episodes_so_far
                self.combined_stats["total/steps"] = self.steps_so_far
                self.combined_stats["total/hours"] = self.total_hours
                self.combined_stats["train/critic_loss"] = np.mean(self.critic_losses)
                self.combined_stats["train/actor_loss"] = np.mean(self.actor_losses)                

            if len(self.eval_episode_rewards) > 0:
                self.combined_stats["eval/return"] = np.mean(self.eval_episode_rewards)
                self.combined_stats["eval/episodes"] = len(self.eval_episode_rewards)

            self.combined_stats["total/mem_usage"] = mem_usage_in_mb()
            
            # Log stats
            self.logger.append_dict_as_row(self.combined_stats, self.combined_stats.keys())

            if self.steps_so_far % 1000 == 0:
                self.plot_stats()
            self.print_stats()
            
            # Create resume checkpoint
            if self.epochs_so_far % self.checkpoint_every_n_epoch == 0:
                self.create_resume_checkpoint()                

    def plot_running_avg(self, running_avg_name="obs"):

        if running_avg_name == "obs":
            m = np.array(self.obs_rms_history["mean"])
            sig = np.array(self.obs_rms_history["std"])
        else:
            m = np.array(self.reward_rms_history["mean"])
            sig = np.array(self.reward_rms_history["std"])
        x = np.arange(0, m.size) / 1000.0

        plt.figure("Running mean/std")
        plt.title("Running mean/std", fontsize=10)
        plt.plot(x, m, "green")
        plt.fill_between(x, m - sig, m + sig, alpha=0.3, facecolor="green", linewidth=0)
        plt.xlabel("Update step", fontsize=8)
        plt.ylabel("mean/std", fontsize=8)
        plt.savefig(
            os.path.join(
                self.save_path_prefix,
                running_avg_name + "_rms_" + self.env_name + ".png",
            )
        )
        plt.close()
        
    def plot_stats(self):
        if not self.return_plot and not self.plot_param_noise_dist:
            return
        # Load stats
        stats = self.logger.csv2dict()

        # Plot returns
        self.plot_return_vs_step(stats)        
        
        if self.obs_rms is not None and self.obs_rms_plot:
            self.plot_running_avg("obs")
        del stats

    def plot_return_vs_step(self, combined_stats):

        steps = np.array(combined_stats["total/steps"]) / 1000.0
        returns = combined_stats["rollout/return"]
        eval_returns = combined_stats["eval/return"]
        fig_title = "episode_reward_vs_Step"
        plt.figure(fig_title)
        plt.title("Episode reward vs step")
        plt.plot(steps, returns, label="rollout ret.")
        if len(eval_returns) > 0:
            plt.plot(steps, eval_returns, label="eval. ret.")
        plt.xlabel("Steps (thousands)")
        plt.ylabel("Episode reward average")
        plt.grid()
        plt.legend()
        plt.savefig(
            os.path.join(self.save_path_prefix, fig_title + self.env_name + ".png")
        )
        plt.close()

    def print_stats(self):
        print("----------------------------------------\n")
        for key, value in self.combined_stats.items():
            if type(value) == list and len(value) > 0:
                print("{0}: {1}".format(key, value[-1]))
            else:
                print("{0}: {1}".format(key, value))