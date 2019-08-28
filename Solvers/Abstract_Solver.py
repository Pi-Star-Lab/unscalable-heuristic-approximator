from abc import ABC, abstractmethod
from enum import Enum


class AbstractSolver(ABC):

    def __init__(self,problem=None,options=None):
        self.statistics = [0] * len(Statistics)
        if options:
            self.use_h = options.use_heuristic

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

    @staticmethod
    def get_out_header():
        ans = "Episode,Problem index,solver"
        for s in Statistics:
            ans += ","+s.name
        return ans


class Statistics(Enum):
    Expanded = 0
    Generated = 1
    Distance = 2
    Solution = 3
