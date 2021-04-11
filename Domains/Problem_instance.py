from Domains.Pancake import Pancake
from Domains.Tile_puzzle import Tile
from Domains.Rubiks_cube import Rubik
from Domains.Sokoban import Sokoban

class ProblemInstance:

    separator = '$'
    _random_steps = 998
    random_steps_domain = {'pancake' : 998,'rubik': 10,'tile': 998}
    domains = ['pancake','rubik','tile', 'sokoban']

    def __init__(self):
        self.index = -1
        self.edge_cost = 1
        self.start = None
        self.goal = None
        self.cls = None

    def __str__(self):
        return str(self.index)+ProblemInstance.separator+self.cls.get_name()+ProblemInstance.separator+ \
               str(self.start)+ProblemInstance.separator+str(self.goal)+ProblemInstance.separator

    def read_in(self, string):
        string = string.split(ProblemInstance.separator)
        self.index = string[0]
        self.cls = ProblemInstance.get_domain_class(string[1])
        self.start = self.cls.parse_state(string[2])
        self.goal = self.cls.parse_state(string[3])

    @property
    def random_steps(self):
        return self._random_steps

    @random_steps.setter
    def random_steps(self, val):
        self._random_steps = val

    def generate_random(self, index, cls, goal):
        self.index = index
        self.cls = cls
        self.goal = goal
        start = goal
        while start == goal:
            for x in range(ProblemInstance.random_steps_domain[self.cls.get_name()]):
                start = start.random_step()
        self.start = start

    def get_state_size(self):
        return len(self.start.as_tensor())

    @staticmethod
    def get_domain_class(name):
        if name == ProblemInstance.domains[0]:
            return Pancake
        elif name == ProblemInstance.domains[1]:
            return Rubik
        elif name == ProblemInstance.domains[2]:
            return Tile
        elif name == ProblemInstance.domains[3]:
            return Sokoban
        else:
            assert False, "unknown domain name as input. domain must be from " + str(ProblemInstance.domains)
