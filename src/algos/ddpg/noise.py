# Code adapted from:
# * Gym baseline DDPG
# * [1] https://awesomeopensource.com/project/navneet-nmk/pytorch-rl & [2] https://github.com/navneet-nmk/pytorch-rl

import numpy as np
import torch
from random import random
from common.net_param_manip import count_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdaptiveParamNoiseSpec(object):
    """
    Implements adaptive parameter noise

    :param initial_stddev: (float) the initial value for the standard deviation of the noise
    :param desired_action_stddev: (float) the desired value for the standard deviation of the noise
    :param adoption_coefficient: (float) the update coefficient for the standard deviation of the noise
    """
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        """
        update the standard deviation for the parameter noise

        :param distance: (float) the noise distance applied to the parameters
        """
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        """
        return the standard deviation for the parameter noise

        :return: (dict) the stats of the noise
        """
        return {'param_noise_stddev': self.current_stddev}

    @staticmethod
    def get_parameters(model, only_perturbable):
        """
        :param model: the neural net. from which parameters are fetched
        :param only_perturbable: If true, only set of perturbable parameters are returned; otherwise all parameters are
        returned
        """
        parameters = []
        for name, params in model.named_parameters():
            if only_perturbable and 'LayerNorm' in name:
                # Layer normalization parameters can't be perturbed so ignore them
                continue
            parameters.append(params)

        return parameters

    def perturb_actor(self, actor, actor_to_perturb):
        """
        Receives 2 actors and sets the perturbable params of `actor_to_perturb` to the perturbed params of `actor`.
        The rest of the params which are non-perturbable (like LayerNorm params) are copies exactly from actor to
        actor_to_perturb
        """
        assert count_params(actor, only_trainable=False) == count_params(actor_to_perturb, only_trainable=False)
        # actor_params = self.get_parameters(actor, only_perturbable=False)
        # perturbed_actor_params = self.get_parameters(actor_to_perturb, only_perturbable=False)
        # assert len(actor_params) == len(perturbed_actor_params)

        key = 0
        param_names = [key for key in actor.state_dict().keys()]

        for params, perturbed_params in zip(actor.parameters(), actor_to_perturb.parameters()):
            if 'LayerNorm' not in param_names[key]:
                # Copy perturbed version of the layer params
                perturbed_params.data.copy_(params + torch.normal(mean=torch.zeros(params.shape).to(device),
                                                                  std=self.current_stddev).to(device))
            else:
                # Copy layerNorm params as they are (no change)
                perturbed_params.data.copy_(params)
            key += 1

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(object):
    """
    The action noise base class
    """
    def reset(self):
        """
        Call end of episode reset for the noise
        """
        pass


class NormalActionNoise(ActionNoise):
    """
    A gaussian action noise
    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    """
    def __init__(self, mean, sigma):
        self.mu = mean
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class EpsilonNormalActionNoise(ActionNoise):
    """
    Adopted from Agarwal code `skill-based exploration`
    Based on Hindsight Experience Replay paper: https://pdfs.semanticscholar.org/9734/9dee55ba13067f467695eecb3a3bb68e43bd.pdf
    """
    def __init__(self, mu=0.0, sigma=0.1, epsilon=0.2):
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        if random() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma,
                                             action.shape)  # action.shape should be [batchSize x actionDim]
        else:
            return np.random.uniform(-1., 1., size=action.shape)

    def __repr__(self):
        return 'EpsilonNormalActionNoise(mu={}, sigma={}, epsilon={})'.format(self.mu, self.sigma, self.epsilon)


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    :param initial_noise: ([float]) the initial value for the noise output, (if None: 0)
    """
    def __init__(self, mean, sigma, theta=.15, dt=1e-2, initial_noise=None):
        self.theta = theta
        self.mu = mean
        self.sigma = sigma
        self.dt = dt
        self.initial_noise = initial_noise
        self.prev_noise = None     # Added to fix warning prev_noise defined outside __init__
        self.reset()

    def __call__(self):
        noise = self.prev_noise + self.theta * (self.mu - self.prev_noise) * self.dt +\
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.prev_noise = noise
        return noise

    # def get_batch(self, n_samples):
    #     """
    #     :param n_samples: Number of noise vectors to be sampled
    #     :return: batch of sampled noises [n_samples x mu_shape]
    #     """
    #     noise = self.prev_noise + self.theta * (self.mu - self.prev_noise) * self.dt +\
    #         self.sigma * np.sqrt(self.dt) * np.random.normal(size=(n_samples, self.mu.shape[0]))
    #     self.prev_noise = noise
    #     return noise

    def reset(self):
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.prev_noise = self.initial_noise if self.initial_noise is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

