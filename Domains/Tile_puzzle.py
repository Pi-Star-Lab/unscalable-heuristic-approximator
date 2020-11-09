from Domains.Abstract_State import AbstractState
import Utils
import copy
import math
import numpy as np

class Tile(AbstractState):

    def __init__(self, board, g):
        self.board = board
        self.dim = int(math.sqrt(len(board)))
        super(Tile,self).__init__(g)

    # Return "[1,2,3,0]" from board '1', '2', '3', 'blank' (blank is '0')
    def __str__(self):
        ans="["
        for n in self.board:
            ans += str(n) + ","
        ans = ans[:-1] + "]"
        return ans

    def get_successors(self):
        successors = []
        blank = self.index_for_tile(0)

        for op in range(4):
            c_board = copy.deepcopy(self.board)
            n = self.neighbor_index(blank,op)
            if n is not None:
                c_board[blank], c_board[n] = c_board[n], c_board[blank]
                child = Tile(c_board, self.g + 1)
                successors.append(child)

        return successors

    def index_for_tile(self, tile):
        return self.board.index(tile)

    def neighbor_index(self, index, move):
        ans = None
        if move == 0: #up
            ans = index - self.dim
            if ans < 0:
                ans = None
        elif move == 1: #right
            ans = index + 1
            if index % self.dim == self.dim - 1:
                ans = None
        elif move == 2: #down
            ans = index + self.dim
            if ans >= len(self.board):
                ans = None
        elif move == 3: #left
            ans = index - 1
            if index % self.dim == 0:
                ans = None
        return ans

    def as_list(self):
        return self.board

    def __eq__(self, other):
        return str(self.board) == str(other.board)

    def __hash__(self):
        return hash(str(self.board))

    # manhattan distance heuristic
    def get_h(self, goal):
        h = 0
        for tile in range(1,len(self.board)):
            i1 = self.index_for_tile(tile)
            i2 = goal.index_for_tile(tile)
            dx = abs(i1 % self.dim - i2 % self.dim)
            dy = abs(int(i1/self.dim) - int(i2/self.dim))
            h += dx + dy
        return h

    def as_tensor(self):
        """
        Return the one-hot encoding of the tile puzzle
        """
        return np.eye(len(self.board))[self.board].reshape(-1)

    @staticmethod
    def parse_state(string):
        string = string[1:-1].split(',') # Change "[0,1,2,3]" to '0', '1', '2', '3'
        board = []
        for n in string:
            board.append(int(n))
        return Tile(board,0)

    @staticmethod
    def get_goal(size):
        tiles = size**2
        board = [i for i in range(tiles)]
        return Tile(board, 0)

    @staticmethod
    def get_name():
        return "tile"
