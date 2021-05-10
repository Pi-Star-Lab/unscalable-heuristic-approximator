import numpy as np
from Domains.Abstract_State import AbstractState
import Utils
"""
Credits: levilelis Git repo: https://github.com/levilelis/h-levin/
Modifications: sumedhpendurkar
"""

class Sokoban(AbstractState):

    _channel_walls = 0
    _channel_goals = 1

    _channel_boxes = 2
    _channel_man = 3

    _goal = '.'
    _man = '@'
    _wall = '#'
    _box = '%'
    _box_on_goal = "*"

    _E = 0
    _W = 1
    _N = 2
    _S = 3

    _number_channels = 4
    _height = 10
    _width = 10

    def __init__(self, state, g):

        # how to get model state in parsing (big issue)
        # what is state representation?
        self._boxes = state[0]
        self._maze = state[1]
        self._x_man = state[2][0]
        self._y_man = state[2][1]

        super(Sokoban, self).__init__(g) ## define goal

    def copy(self):

        copy_state = Sokoban((self._boxes.copy(), self._maze.copy(), (self._x_man, self._y_man)), self.g)
        copy_state._width = self._width
        copy_state._height = self._height
        #copy_state._maze = self._maze

        #copy_state._x_man = self._x_man
        #copy_state._y_man = self._y_man

        #copy_state._boxes = self._boxes.copy()

        return copy_state

    def __hash__(self):
        return hash((str(self._boxes), str(self._man)))

    def __eq__(self, other):
        return np.array_equal(self._boxes, other._boxes) and self._x_man == other._x_man and self._y_man == other._y_man

    def get_successors(self):
        actions = []

        if self._x_man + 1 < self._width:
            if (self._maze[self._y_man][self._x_man + 1][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man + 1] == 0):

                actions.append(self._E)
            elif (self._maze[self._y_man][self._x_man + 1][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man + 1] == 1 and
                self._x_man + 2 < self._width and
                self._maze[self._y_man][self._x_man + 2][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man + 2] == 0):

                actions.append(self._E)

        if self._x_man - 1 > 0:
            if (self._maze[self._y_man][self._x_man - 1][Sokoban._channel_walls] == 0 and
                    self._boxes[self._y_man][self._x_man - 1] == 0):

                actions.append(self._W)

            elif (self._maze[self._y_man][self._x_man - 1][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man - 1] == 1 and
                self._x_man - 2 > 0 and
                self._maze[self._y_man][self._x_man - 2][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man - 2] == 0):

                actions.append(self._W)

        if self._y_man + 1 < self._height:
            if (self._maze[self._y_man + 1][self._x_man][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man + 1][self._x_man] == 0):

                actions.append(self._S)
            elif (self._maze[self._y_man + 1][self._x_man][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man + 1][self._x_man] == 1 and
                self._y_man + 2 < self._height and
                self._maze[self._y_man + 2][self._x_man][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man + 2][self._x_man] == 0):

                actions.append(self._S)

        if self._y_man - 1 > 0:
            if (self._maze[self._y_man - 1][self._x_man][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man - 1][self._x_man] == 0):

                actions.append(self._N)
            elif (self._maze[self._y_man - 1][self._x_man][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man - 1][self._x_man] == 1 and
                self._y_man - 2 > 0 and
                self._maze[self._y_man - 2][self._x_man][Sokoban._channel_walls] == 0 and
                self._boxes[self._y_man - 2][self._x_man] == 0):

                actions.append(self._N)

        return [self.copy().apply_action(a) for a in actions]

    def apply_action(self, action):

        if action == self._N:
            if self._boxes[self._y_man - 1][self._x_man] == 1:
                self._boxes[self._y_man - 1][self._x_man] = 0
                self._boxes[self._y_man - 2][self._x_man] = 1
            #self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            #self._puzzle[self._y_man - 1][self._x_man][self._channel_man] = 1
            self._y_man = self._y_man - 1

        if action == self._S:
            if self._boxes[self._y_man + 1][self._x_man] == 1:
                self._boxes[self._y_man + 1][self._x_man] = 0
                self._boxes[self._y_man + 2][self._x_man] = 1
            #self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            #self._puzzle[self._y_man + 1][self._x_man][self._channel_man] = 1
            self._y_man = self._y_man + 1

        if action == self._E:
            if self._boxes[self._y_man][self._x_man + 1] == 1:
                self._boxes[self._y_man][self._x_man + 1] = 0
                self._boxes[self._y_man][self._x_man + 2] = 1
            #self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            #self._puzzle[self._y_man][self._x_man + 1][self._channel_man] = 1
            self._x_man = self._x_man + 1

        if action == self._W:
            if self._boxes[self._y_man][self._x_man - 1] == 1:
                self._boxes[self._y_man][self._x_man - 1] = 0
                self._boxes[self._y_man][self._x_man - 2] = 1
#             self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
#             self._puzzle[self._y_man][self._x_man - 1][self._channel_man] = 1
            self._x_man = self._x_man - 1
        self.g += 1
        return self

    def is_solution(self):
        # change goal check for everything else, as this might have multiple goal states
        for i in range(self._height):
            for j in range(self._width):
                if self._boxes[i][j] == 1 and self._maze[i][j][self._channel_goals] == 0:
                    return False
        return True

    def as_tensor(self):
        image = np.zeros((Sokoban._height, Sokoban._width, Sokoban._number_channels))
        for i in range(Sokoban._height):
            for j in range(Sokoban._width):
                image[i][j][Sokoban._channel_goals] = self._maze[i][j][Sokoban._channel_goals]
                image[i][j][Sokoban._channel_walls] = self._maze[i][j][Sokoban._channel_walls]
                image[i][j][Sokoban._channel_boxes] = self._boxes[i][j]

        image[self._y_man][self._x_man][Sokoban._channel_man] = 1

        #flatten the image here (else use a CNN
        return image.transpose(2, 0, 1)
        #return image.reshape(1)

    def get_h(self,goal):
        h = 0
        h_man = self._width + self._height

        for i in range(self._height):
            for j in range(self._width):
                if self._boxes[i][j] == 1 and self._maze[i][j][Sokoban._channel_goals] == 0:
                    h_box = self._width + self._height
                    for l in range(self._height):
                        for m in range(self._width):
                            if self._maze[l][m][Sokoban._channel_goals] == 1:
                                dist_to_goal = abs(l - i) + abs(m - j)
                                if dist_to_goal < h_box:
                                    h_box = dist_to_goal
                    h += h_box
                if self._boxes[i][j] == 1:
                    dist_to_man = abs(self._y_man - i) + abs(self._x_man - j) - 1
                    if dist_to_man < h_man:
                        h_man = dist_to_man
        h += h_man
        return h

    def __str__(self):
        string = ""
        for i in range(self._height):
            for j in range(self._width):
                if self._maze[i][j][Sokoban._channel_goals] == 1 and self._boxes[i][j] == 1:
                    string += '*'
                elif i == self._y_man and j == self._x_man:
                    string += Sokoban._man
                elif self._maze[i][j][Sokoban._channel_goals] == 1:
                    string += Sokoban._goal
                elif self._maze[i][j][Sokoban._channel_walls] == 1:
                    string += Sokoban._wall
                elif self._boxes[i][j] == 1:
                    string += Sokoban._box
                else:
                    string += ' '
            string += "\n"
        return string

    def is_solution(self):
        for i in range(self._height):
            for j in range(self._width):
                if self._boxes[i][j] == 1 and self._maze[i][j][self._channel_goals] == 0:
                    return False
        return True

    @staticmethod
    def parse_state(string_state):
        if len(string_state) > 0:
            _width = 10 #len(string_state[0])
            _height = 10 #len(string_state)
            width = _width
            _maze = np.zeros((_height, _width, 2))
            _boxes = np.zeros((_height, _width))
            _x_man, _y_man = 0, 0
            for i in range(_height):
                for j in range(_width):
                    if string_state[i * width + j] == Sokoban._goal:
                        _maze[i][j][Sokoban._channel_goals] = 1

                    if string_state[i * width  + j] == Sokoban._man:
                        _y_man = i
                        _x_man = j

                    if string_state[i * width + j] == Sokoban._wall:
                        _maze[i][j][Sokoban._channel_walls] = 1

                    if string_state[i * width + j] == Sokoban._box:
                        _boxes[i][j] = 1

                    if string_state[i * width + j] == Sokoban._box_on_goal:
                        _boxes[i][j] = 1
                        _maze[i][j][Sokoban._channel_goals] = 1
        return Sokoban((_boxes, _maze, (_x_man, _y_man)), 0)


    @staticmethod
    def get_goal_dummy(size = 10):
        # Dummy get goal, because it needs to be specific to the problem, thus cannot be static
        string = "".join([''.join(["#" for x in range(10)]) for y in range(10)])
        return Sokoban.parse_state(string)

    @staticmethod
    def get_name():
        return "sokoban"

    def as_list(self):
        """
        Not valid
        """
        return []
