from collections import deque
import numpy as np
import random

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

    def update_target_values(self, get_target_value):
        """
        :param get_target_value: A function that would give you the target value
        """
        i = 0
        for x, y in self.memory:
            if x == y[1]:
                self.buffer_target[i] = 0
            else:
                min_target = get_target_value(x, y[1])
                self.buffer_target[i] = y[0] + min_target
                """ remove the code below """
                if i < 45:
                    print(y[0], min_target, end = ",   ")
            if i < 45:
                pass
                #print(i, self.buffer_target[i], "actual", self.get_h(x, y[1]))
            i += 1

    def sample(self, sample_size):
        """
        :param sample_size: Number of desired samples
        """
        idxes = [random.randint(0, len(self) - 1) for _ in range(sample_size)]
        x, y = np.array(self.buffer_x), np.array(self.buffer_target)
        return x[idxes], y[idxes]
