from Domains.Abstract_State import AbstractState
import Utils
import copy
import math
import numpy as np

class BlocksWorld(AbstractState):
    def __init__(self, table, g):
        self.table = table
        self.dim = len(table)
        super(BlocksWorld,self).__init__(g)

    # Return "[1,2,3,0]" from table '1', '2', '3', 'blank' (blank is '0')
    def __str__(self):
        ans="["
        for n in self.table:
            ans += str(n) + ","
        ans = ans[:-1] + "]"
        return ans

    def get_successors(self):
        #TODO
        raise NotImplementedError

    def index_for_tile(self, tile):
        return self.table.index(tile)

    def as_list(self):
        return self.table

    def __eq__(self, other):
        return str(self.table) == str(other.table)

    def __hash__(self):
        return hash(str(self.table))

    # manhattan distance heuristic
    def get_h(self, goal):
        #TODO
        raise NotImplementedError

    def as_tensor(self):
        """
        Return the one-hot encoding of the tile puzzle
        return np.eye(len(self.table))[self.table].reshape(-1)
        """
        #return np.eye(len(self.table))[self.table].reshape(self.dim ** 2, self.dim, self.dim)
        return np.eye(len(self.table))[self.table].reshape(-1)

    def is_solution(self):

        raise NotImplementedError

    @staticmethod
    def parse_state(string):
        string = string[1:-1].split(',') # Change "[0,1,2,3]" to '0', '1', '2', '3'
        table = []
        for n in string:
            table.append(int(n))
        return BlocksWorld(table,0)

    @staticmethod
    def get_goal_dummy(size):
        table = [0 for i in range(size)]
        return BlocksWorld(table, 0)

    @staticmethod
    def get_name():
        return "BlocksWorld"
