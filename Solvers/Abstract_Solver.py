from abc import ABC, abstractmethod
from enum import Enum


class AbstractSolver(ABC):

    def __init__(self):
        self.statistics = [0] * len(Statistics)

    def get_stat(self,problem):
        ans = '{},{}'.format(str(problem.index), str(self))
        for s in Statistics:
            ans += ',' + str(self.statistics[s.value])
        return ans

    @abstractmethod
    def solve(self,problem):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def train(self, options):
        pass

    @staticmethod
    def get_out_header():
        ans = "Episode,Problem index,solver"
        for s in Statistics:
            ans += ","+s.name
        return ans

    def load(**kwargs):
        raise Exception("No Learning, Cannot load")

    def save(**kwargs):
        print("No Learning, Nothing to save")



class Statistics(Enum):
    Expanded = 0
    Generated = 1
    Distance = 2
    Solution = 3
    TrustRadius = 4
