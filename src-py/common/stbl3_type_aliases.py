import torch as th
from typing import NamedTuple, Optional

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    # For n-step replay buffer
    discounts: Optional[th.Tensor] = None