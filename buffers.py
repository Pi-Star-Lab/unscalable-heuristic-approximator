from collections import deque
import numpy as np
import random
import torch
import pickle

class ReplayBufferSearch(object):
    """
    Replay Buffer for search-TD based problems
    """
    def __init__(self, max_len):
        """
        :param max_len: Maximum length of buffer
        """
        self.len = max_len
        self.buffer_x = deque(maxlen=max_len)
        self.memory = deque(maxlen=max_len)
        self.buffer_target = deque(maxlen=max_len) #just to save computational speed!

    def __len__(self):
        return len(self.buffer_x)

    def append(self, x_value, memory, target):
        """
        :param x_value: state representation
        :param memory: state cost representation
        :param target: Target value for the states
        """
        self.buffer_x.append(x_value)
        self.memory.append(memory)
        self.buffer_target.append(target)

    def sample(self, sample_size):
        """
        :param sample_size: Number of desired samples
        """
        idxes = [random.randint(0, len(self) - 1) for _ in range(sample_size)]
        x, y = np.array(self.buffer_x), np.array(self.buffer_target)
        return x[idxes], y[idxes]

    def save(self, path):
        f = open(path + ".pkl", "wb")
        pickle.dump(self.memory, f)
        f = open(path + "_x.pkl", "wb")
        pickle.dump(self.buffer_x, f)
        f = open(path + "_target.pkl", "wb")
        pickle.dump(self.buffer_target, f)

    def load(self, path):
        f = open(path + ".pkl", "rb")
        self.memory = pickle.load(f)
        f = open(path + "_x.pkl", "rb")
        self.buffer_x = pickle.load(f)
        f = open(path + "_target.pkl", "rb")
        self.buffer_target = pickle.load(f)

    def update_target_buffer(self, target_values):
        self.buffer_target = deque(target_values, maxlen=self.len)

class PrioritizedReplayBufferSearch(ReplayBufferSearch):
    """
    Priortized Replay Buffer for search-TD based problems
    Uses MSE by default
    """
    def set_predict_function(self, fn):
        """
        :param fn: Function to used to predict NN values
        """
        self.predict_fn = fn

    def sample(self, sample_size):
        """
        :param sample_size: Number of desired samples
        """
        predicted_values = self.predict_fn(np.array(self.buffer_x)).cpu()
        loss_fn = torch.nn.MSELoss(reduction = 'none')
        bt = torch.Tensor(self.buffer_target).unsqueeze(1)
        priorities = loss_fn(predicted_values, bt)
        idxes = torch.multinomial(priorities.squeeze(1), sample_size, replacement=True)
        idxes = list(np.array(idxes))
        x, y = np.array(self.buffer_x), np.array(self.buffer_target)
        return x[idxes], y[idxes]
