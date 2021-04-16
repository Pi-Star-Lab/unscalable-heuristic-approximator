from abc import ABC, abstractmethod
import numpy as np
import random


class AbstractState(ABC):
    w = 1

    def __init__(self, g):
        self.g = g
        self.h = -1
        self.parent = None

    def set_h(self, value):
        self.h = value

    def random_step(self):
        successors = self.get_successors()

        # Generates a random successor index
        r1 = random.randint(0, len(successors) - 1)
        return successors[r1]

    def get_f(self):
        return self.g + AbstractState.w * self.h

    def __lt__(self, other):
        return self.get_f() < other.get_f()

    # def as_tensor(self):
    #     x = self.as_numpy()
    #     x = np.reshape(self.as_numpy(), [1, len(x)])
    #     return x

    def as_tensor(self):
        return np.array(self.as_list())

    def get_path(self):
        if self.parent is None:
            path = [self]
        else:
            path = self.parent.get_path()
            path.append(self)
        return path

    @abstractmethod
    def get_h(self, goal):
        pass

    @abstractmethod
    def is_solution(self):
        pass


    @abstractmethod
    def get_successors(self):
        pass

    @abstractmethod
    def as_list(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @staticmethod
    def get_name():
        pass

    @staticmethod
    def parse_state(string):
        pass

    @staticmethod
    def get_goal_dummy(size):
        pass
