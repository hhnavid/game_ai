import torch
from typing import Union

import numpy as np
import random
import gymnasium as gym


def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to torch.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


class LinearSchedule:
    """
    LinearSchedule interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).

    :param start: value to start with if ``progress_remaining`` = 1
    :param end: value to end with if ``progress_remaining`` = 0
    :param end_fraction: fraction of ``progress_remaining``  where end is reached e.g 0.1
        then end is reached after 10% of the complete training process.
    """

    def __init__(self, start: float, end: float, end_fraction: float) -> None:
        self.start = start
        self.end = end
        self.end_fraction = end_fraction

    def __call__(self, progress_remaining: float) -> float:
        if (1 - progress_remaining) > self.end_fraction:
            return self.end
        else:
            return self.start + (1 - progress_remaining) * (self.end - self.start) / self.end_fraction

    def __repr__(self) -> str:
        return f"LinearSchedule(start={self.start}, end={self.end}, end_fraction={self.end_fraction})"
    
    

def set_global_seeds(seed, is_cuda_available):
    """
    set the seed for python random, pytorch, numpy and gym_custom spaces

    :param seed: (int) the seed
    """
    torch.manual_seed(seed)
    if is_cuda_available: 
        torch.cuda.manual_seed(123)

    np.random.seed(seed)
    random.seed(seed)
    # prng was removed in latest gym version
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)
        
def get_random_generators_state():
    """
    return pseudo random generators state which will be used to resume training
    """
    return torch.get_rng_state(), np.random.get_state(), random.getstate()

def set_random_generators_state(torch_state, np_state, py_rnd_state):
    """
    resume pseudo random generators state for resuming
    """
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(py_rnd_state)