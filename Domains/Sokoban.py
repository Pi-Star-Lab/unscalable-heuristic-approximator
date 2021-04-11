import numpy as np
from domains.environment import Environment

"""
Credits: levilelis Git repo: https://github.com/levilelis/h-levin/
Modifications: sumedhpendurkar
"""

class Sokoban(Environment):

    self._channel_walls = 0
    self._channel_goals = 1

    self._channel_boxes = 2
    self._channel_man = 3

    self._goal = '.'
    self._man = '@'
    self._wall = '#'
    self._box = '$'

    self._E = 0
    self._W = 1
    self._N = 2
    self._S = 3

    self._number_channels = 4

    def __init__(self, state, goal):

        # how to get model state in parsing (big issue)
        # what is state representation?
        super(Sokoban, self).__init__(g) ## define goal

    @staticmethod
    def parse_state(string_state):
        if len(string_state) > 0:
           self._width = len(string_state[0])
            self._height = len(string_state)

            self._maze = np.zeros((self._height, self._width, 2))
            self._boxes = np.zeros((self._height, self._width))

            for i in range(self._height):
                for j in range(self._width):
                    if string_state[i][j] == self._goal:
                        self._maze[i][j][self._channel_goals] = 1

                    if string_state[i][j] == self._man:
                        self._y_man = i
                        self._x_man = j

                    if string_state[i][j] == self._wall:
                        self._maze[i][j][self._channel_walls] = 1

                    if string_state[i][j] == self._box:
                        self._boxes[i][j] = 1
        #return a Sokoban object

    def copy(self):
        copy_state = Sokoban()

        copy_state._width = self._width
        copy_state._height = self._height
        copy_state._maze = self._maze

        copy_state._x_man = self._x_man
        copy_state._y_man = self._y_man

        copy_state._boxes = self._boxes.copy()

        return copy_state

    def __hash__(self):
        return hash((str(self._boxes), str(self._man)))

    def __eq__(self, other):
        return np.array_equal(self._boxes, other._boxes) and self._x_man == other._x_man and self._y_man == other._y_man

    def get_successors(self):
        actions = []

        if self._x_man + 1 < self._width:
            if (self._maze[self._y_man][self._x_man + 1][self._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man + 1] == 0):

                actions.append(self._E)
            elif (self._maze[self._y_man][self._x_man + 1][self._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man + 1] == 1 and
                self._x_man + 2 < self._width and
                self._maze[self._y_man][self._x_man + 2][self._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man + 2] == 0):

                actions.append(self._E)

        if self._x_man - 1 > 0:
            if (self._maze[self._y_man][self._x_man - 1][self._channel_walls] == 0 and
                    self._boxes[self._y_man][self._x_man - 1] == 0):

                actions.append(self._W)

            elif (self._maze[self._y_man][self._x_man - 1][self._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man - 1] == 1 and
                self._x_man - 2 > 0 and
                self._maze[self._y_man][self._x_man - 2][self._channel_walls] == 0 and
                self._boxes[self._y_man][self._x_man - 2] == 0):

                actions.append(self._W)

        if self._y_man + 1 < self._height:
            if (self._maze[self._y_man + 1][self._x_man][self._channel_walls] == 0 and
                self._boxes[self._y_man + 1][self._x_man] == 0):

                actions.append(self._S)
            elif (self._maze[self._y_man + 1][self._x_man][self._channel_walls] == 0 and
                self._boxes[self._y_man + 1][self._x_man] == 1 and
                self._y_man + 2 < self._height and
                self._maze[self._y_man + 2][self._x_man][self._channel_walls] == 0 and
                self._boxes[self._y_man + 2][self._x_man] == 0):

                actions.append(self._S)

        if self._y_man - 1 > 0:
            if (self._maze[self._y_man - 1][self._x_man][self._channel_walls] == 0 and
                self._boxes[self._y_man - 1][self._x_man] == 0):

                actions.append(self._N)
            elif (self._maze[self._y_man - 1][self._x_man][self._channel_walls] == 0 and
                self._boxes[self._y_man - 1][self._x_man] == 1 and
                self._y_man - 2 > 0 and
                self._maze[self._y_man - 2][self._x_man][self._channel_walls] == 0 and
                self._boxes[self._y_man - 2][self._x_man] == 0):

                actions.append(self._N)

        return [self.apply_action(a) for a in actions]

    def successors_parent_pruning(self, op):
        actions = []

        return self.successors()

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

    def is_solution(self):
        for i in range(self._height):
            for j in range(self._width):
                if self._boxes[i][j] == 1 and self._maze[i][j][self._channel_goals] == 0:
                    return False
        return True

    def as_tensor(self):
        image = np.zeros((self._height, self._width, self._number_channels))
        for i in range(self._height):
            for j in range(self._width):
                image[i][j][self._channel_goals] = self._maze[i][j][self._channel_goals]
                image[i][j][self._channel_walls] = self._maze[i][j][self._channel_walls]
                image[i][j][self._channel_boxes] = self._boxes[i][j]

        image[self._y_man][self._x_man][self._channel_man] = 1

        #flatten the image here (else use a CNN
        return image.reshape(-1)

    def get_h(self):
        h = 0
        h_man = self._width + self._height

        for i in range(self._height):
            for j in range(self._width):
                if self._boxes[i][j] == 1 and self._maze[i][j][self._channel_goals] == 0:
                    h_box = self._width + self._height
                    for l in range(self._height):
                        for m in range(self._width):
                            if self._maze[l][m][self._channel_goals] == 1:
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
                if self._maze[i][j][self._channel_goals] == 1 and self._boxes[i][j] == 1:
                    str += '*'
                elif i == self._y_man and j == self._x_man:
                    str += self._man
                elif self._maze[i][j][self._channel_goals] == 1:
                    str += self._goal
                elif self._maze[i][j][self._channel_walls] == 1:
                    str += self._wall
                elif self._boxes[i][j] == 1:
                    str += self._box
                else:
                    str += ' '
            print()

    @staticmethod
    def get_goal(size):

    @staticmethod
    def get_name():
        return "sokoban"

