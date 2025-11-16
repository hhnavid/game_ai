# Code adapted from Gym baseline DDPG
# ------------------------------------
import os
import gymnasium as gym
import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        """
        This is a cyclic buffer which means if it becomes full, overwriting existing
        items will begin from index 0 and so on.
        :param maxlen: Maximum capacity of buffer
        :param shape: (Tuple) Dimension of each item in the buffer
        :param dtype: Type of item in the buffer
        """
        self.maxlen = maxlen
        self.start = 0                                         # Specifies current start index of cyclic buffer
        self.length = 0                                        # Current number of items in the buffer
        self.data = np.zeros((maxlen,) + shape).astype(dtype)  # [maxlen x shape]
        self.last_append_absolute_index = None

    def __len__(self):
        """
        :return: Current number of items in the buffer
        """
        return self.length

    def __getitem__(self, idx):
        """
        :param idx: Index of item to be fetched
        :return: item with index `idx`
        """
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs, relative_indexing=True, return_idxs=False):
        """
        :param idxs:
        :param relative_indexing: (Bool) if True, the indexes are considered to be relative to current buffer start index.
        If False, the indexes are considered absoulte and are used directly to access the buffer. DDPG original code uses
        relative indexing to access the RingBuffer data
        :param return_idxs: (Bool) if True, the absolute indexex of the drawn sample are returned as well
        """
        if relative_indexing:
            idxs = (self.start + idxs) % self.maxlen
        # else: the input indexes are directly used to access the buffer

        if return_idxs:
            return self.data[idxs], idxs
        else:
            return self.data[idxs]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.last_append_absolute_index = (self.start + self.length - 1) % self.maxlen
        self.data[self.last_append_absolute_index] = v

    def append_batch(self, data_batch):
        """
        :param data_batch: [batch_size x data_dim]
        :return:
        """
        batch_size = data_batch.shape[0]
        new_length = self.length + batch_size

        # case 1: enough space available to store the batch
        if new_length <= self.maxlen:
            # self.start -> no change
            store_idexes = self.length + np.arange(start=0, stop=batch_size)  # arange: [0, batch_size-1]
            self.length = new_length

        # case 2: buffer is partially full => some of the existing items of the buffer must be overwritten
        elif self.length < self.maxlen < new_length:
            available_space = self.maxlen - self.length
            overwritten_space = batch_size - available_space
            self.start = (self.start + overwritten_space) % self.maxlen

            idx1 = self.length + np.arange(start=0, stop=available_space)    # arange: [0, available_space-1]
            idx2 = np.arange(start=0, stop=overwritten_space)                # arange: [0, overwritten_space-1]
            store_idexes = np.concatenate((idx1, idx2))

            self.length = self.maxlen

        # case 3: buffer if already full => we will have batch_size overwrites
        elif self.length == self.maxlen:
            # self.length -> no change
            self.start = (self.start + self.length + batch_size) % self.maxlen
            store_idexes = self.start - np.arange(start=batch_size, stop=0, step=-1)  # arange: [batch_size, 1]
            store_idexes[store_idexes < 0] += self.maxlen
        else:
            # This should never happen
            raise RuntimeError()
        self.data[store_idexes, :] = data_batch

    def clear(self):
        """
        Clears all items from the buffer
        """
        self.start = 0
        self.length = 0

    def save(self, save_path):
        """
        Save buffer data to binary file '.dat'
        """
        np.save(save_path, self.data[0:self.length, :])

    def load(self, load_path):
        """
        Load buffer data from a file
        """
        data = np.load(load_path)
        self.append_batch(data)

def array_min2d(x):
    """
    If x is 1D, makes it a 2D (column) vector. Otherwise x is returned as it is.
    :param x: numpy ndarray
    :return:
    """
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class ReplayBuffer(object):
    def __init__(self, limit, obs_shape, obs_dtype, 
                 action_shape, action_dtype):
        """
        :param limit: Max capacity of replay buffer
        :param action_shape: (Tuple) Action space dimensions
        :param observation_shape: (Tuple) Observation space dimensions
        """
        self.limit = limit

        self.prev_states = RingBuffer(limit, shape=obs_shape, dtype=obs_dtype)  # [length x state_dim]
        self.actions = RingBuffer(limit, shape=action_shape, dtype=action_dtype)           # [length x action_dim]
        self.rewards = RingBuffer(limit, shape=(1,))                   # [length x 1]
        self.done = RingBuffer(limit, shape=(1,))                      # [length x 1]
        self.nex_states = RingBuffer(limit, shape=obs_shape, dtype=obs_dtype)   # [length x state_dim]

    def sample(self, batch_size):
        """
        :param batch_size: # of samples to sample from buffer
        :return: A dictionary containing batch of samples from replay buffer
        """
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(self.nb_entries - 2, size=batch_size)  # rand ints between [0,self.nb_entries-2)

        prev_state_batch = self.prev_states.get_batch(batch_idxs)
        next_state_batch = self.nex_states.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)
        done_batch = self.done.get_batch(batch_idxs)

        result = {
            'prev_states': array_min2d(prev_state_batch),
            'next_states': array_min2d(next_state_batch),
            'rewards': array_min2d(reward_batch),
            'actions': array_min2d(action_batch),
            'done': array_min2d(done_batch),
        }
        return result

    def append(self, prev_state, action, reward, next_state, done, training=True):
        if not training:
            return

        self.prev_states.append(prev_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.nex_states.append(next_state)
        self.done.append(done)

    def append_batch(self, prev_states, actions, rewards, next_states, dones):
        self.prev_states.append_batch(prev_states)
        self.actions.append_batch(actions)
        self.rewards.append_batch(rewards)
        self.nex_states.append_batch(next_states)
        self.done.append_batch(dones)

    def clear(self):
        """
        Clears all entries from the buffer
        """
        self.prev_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.done.clear()
        self.nex_states.clear()

    def save(self, save_path):
        """
        Saves the buffer content to a file
        :param save_path: (Str) The path to save the buffer content e.g. '/my_path/'
        """
        self.prev_states.save(os.path.join(save_path, 'prev_states.npy'))
        self.actions.save(os.path.join(save_path, 'actions.npy'))
        self.rewards.save(os.path.join(save_path, 'rewards.npy'))
        self.nex_states.save(os.path.join(save_path, 'nex_states.npy'))
        self.done.save(os.path.join(save_path, 'done.npy'))

    def load(self, load_path):
        """
        Loads buffer data from file
        :param load_path: (Str) the path to the file from which buffer data are loaded
        """
        self.prev_states.load(os.path.join(load_path, 'prev_states.npy'))
        self.actions.load(os.path.join(load_path, 'actions.npy'))
        self.rewards.load(os.path.join(load_path, 'rewards.npy'))
        self.nex_states.load(os.path.join(load_path, 'nex_states.npy'))
        self.done.load(os.path.join(load_path, 'done.npy'))

    @property
    def nb_entries(self):
        return len(self.prev_states)

    @property
    def last_append_absolute_idx(self):
        """
        :return: The absolute index at which the last append is performed
        """
        return self.prev_states.last_append_absolute_index


if __name__ == "__main__":

    env = gym.make("Acrobot-v1", render_mode="human")
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_min = float(env.action_space.low[0])
    action_max = float(env.action_space.high[0])
    memory = ReplayBuffer(limit=35,
                          action_shape=env.action_space.shape,
                          observation_shape=env.observation_space.shape)
    # Test appending
    n_steps = 72
    env.reset()
    for _ in range(n_steps):
        s = np.random.uniform(low=-10, high=10, size=state_dim)
        a = np.random.uniform(low=-1, high=1, size=action_dim)
        sp, r, done_, info = env.step(action_max * a)
        memory.append(prev_state=s, next_state=sp, action=a, reward=r, done=done_)

    # Test sampling
    batch = memory.sample(batch_size=10)
    print('Done.')
