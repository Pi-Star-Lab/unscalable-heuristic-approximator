from Domains.Abstract_State import AbstractState
import Utils
import math
import copy
import numpy as np

class Rubik(AbstractState):

    #{F(0), U(1), B(2), D(3), R(4), L(5)}
    rotate_front = [[31, 32, 33], [45, 46, 47], [9, 10, 11], [36, 37, 38]]
    rotate_up = [[4, 5, 6], [51, 52, 45], [18, 19, 20], [38, 39, 40]]
    rotate_back = [[13, 14, 15], [49, 50, 51], [27, 28, 29], [40, 41, 42]]
    rotate_down = [[22, 23, 24], [47, 48, 49], [0, 1, 2], [42, 43, 36]]
    rotate_right = [[6, 7, 0], [15, 16, 9], [24, 25, 18], [33, 34, 27]]
    rotate_left = [[2, 3, 4], [29, 30, 31], [20, 21, 22], [11, 12, 13]]
    rotate_indices = [rotate_front, rotate_up, rotate_back, rotate_down, rotate_right, rotate_left]

    def __init__(self, cube, g):
        self.cube = cube
        self.dim = int(math.sqrt(len(cube)/6))
        assert self.dim==3, "Rubik's cube with dimensionality of 3 i.e., 3x3x3 is the nly one supported at the moment"
        super(Rubik,self).__init__(g)

    # Return a single vector from cube
    def __str__(self):
        ans="["
        for n in self.cube:
            ans += str(n) + ","
        ans = ans[:-1] + "]"
        return ans

    def get_successors(self):
        successors = []

        for direction in range(2):
            for face in range(6):
                c_cube = copy.deepcopy(self.cube)
                # A cube move rotates one of the 6 faces 90 degrees
                # face from {F(0), U(1), B(2), D(3), R(4), L(5)}
                for i in range(8):
                    if direction == 0:
                        c_cube[i + face * 9] = self.cube[(i + 2) % 8 + face * 9]
                    if direction == 1:
                        c_cube[i + face * 9] = self.cube[(i - 2) % 8 + face * 9]
                for side in range(4):
                    d = 1 if direction == 0 else -1
                    for i1, i2 in zip(Rubik.rotate_indices[face][d * side], Rubik.rotate_indices[face][d * (side + 1) % 4]):
                        c_cube[i1] = self.cube[i2]
                child = Rubik(c_cube, self.g + 1)
                successors.append(child)
        return successors

    def as_list(self):
        return self.cube

    def __eq__(self, other):
        return str(self.cube) == str(other.cube)

    def __hash__(self):
        return hash(str(self.cube))

    def get_misplaced_edges(self, goal):
        """
        Return the number of misplaced edges per face
        """
        me = []
        for face in range(6):
            index = face * 9
            misplaced_edges = 0
            for cube in range(9):
                if face % 2 == 0:
                    #Even faces (starting 0)
                    if index % 2 == 1 and self.cube[index] != goal.cube[index]: # if an edge and misplaced
                        misplaced_edges += 1
                else:
                    if index % 2 == 0 and self.cube[index] != goal.cube[index]:
                        misplaced_edges += 1
                index += 1
            me.append(misplaced_edges)
        return me

    def get_misplaced_cornors(self, goal):
        """
        Return the number of misplaced cornors per face
        """
        mc = []
        for face in range(6):
            index = face * 9
            misplaced_cornor = 0
            for cube in range(9):
                if face % 2 == 0:
                    #Even faces (starting 0)
                    if index % 2 == 0 and self.cube[index] != goal.cube[index]: # if an edge and misplaced
                        misplaced_cornor += 1
                else:
                    if index % 2 == 1 and self.cube[index] != goal.cube[index]:
                        misplaced_cornor += 1
                index += 1
            mc.append(misplaced_cornor)
        return mc

    def get_misplaced_cornors_2(self, goal):
        """
        Return the number of misplaced cornors per face (as per report)
        """
        mc = []
        next_idx = lambda face, idx: idx + 1
        prev_idx = lambda face, idx: idx - 1 if face * 9 < idx - 1 else (face + 1) * 9 - 2
        for face in range(6):
            index = face * 9
            misplaced_cornor = 0
            for cube in range(9):
                if cube == 8:
                    index += 1
                    continue
                if face % 2 == 0:
                    #Even faces (starting 0)
                    if index % 2 == 0 and \
                            (self.cube[index] != self.cube[next_idx(face, index)] and \
                            self.cube[index] != self.cube[prev_idx(face, index)]):
                        misplaced_cornor += 1
                else:
                    if index % 2 == 1 and \
                            (self.cube[index] != self.cube[next_idx(face, index)] and \
                            self.cube[index] != self.cube[prev_idx(face, index)]):
                        misplaced_cornor += 1
                index += 1
            mc.append(misplaced_cornor)
        return mc


    def get_h(self, goal):
        me = self.get_misplaced_edges(goal)
        mc = self.get_misplaced_cornors_2(goal)
        g_f =  [mc[x] + me[x] for x in range(len(me))]
        return (max(g_f) + min(g_f)) / 4

    @staticmethod
    def parse_state(string):
        string = string[1:-1].split(',') # Change "[0,1,2,3]" to '0', '1', '2', '3'
        cube = []
        for n in string:
            cube.append(int(n))
        return Rubik(cube,0)

    @staticmethod
    def get_goal(size):
        cube = []
        for f in range(6):
            for i in range(size**2):
                cube.append(f)
        return Rubik(cube, 0)

    def as_tensor(self):
        """
        Return the one-hot encoding of the rubik's cube problem
        """
        return np.eye(6)[self.cube].reshape(-1)


    @staticmethod
    def get_name():
        return "rubik"
