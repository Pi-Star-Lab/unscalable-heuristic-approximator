from Solvers.Abstract_Solver import AbstractSolver, Statistics
from Domains.Problem_instance import ProblemInstance as prob
from statistics import mean
from collections import defaultdict
import pickle
from Solvers.A_Star import AStar
import copy

class TMC(AbstractSolver):

    def __init__(self, problem = None, options = None):

        super(TMC, self).__init__()
        self.greedy_solver = AStar()
        self.greedy_solver.__init__(problem, options, return_expanded = False)

        #do we wanna use it?
        self.w = 0

    def init_h(self, input_dim, options):

        self.lr = options.learning_rate
        self.H = defaultdict(int)
        self.counter = defaultdict(int)

    def solve(self, problem, expansion_bound = 100, dw = 0.02):

        self.greedy_solver.__init__(return_expanded = True)
        self.greedy_solver.h_func = self.weighted_h

        path, expanded = self.greedy_solver.solve(problem)

        print("Path found. Length of Path: {}".format(len(path)))

        path.reverse()

        for x, state in enumerate(path):
            self.H[state] = (1 - self.lr) * self.H[state] + \
                    self.lr * x
        self.statistics = copy.deepcopy(self.greedy_solver.statistics)
        self.statistics[Statistics.Weights.value] = self.w
        self.greedy_solver.noise_std = self.greedy_solver.noise_std * AStar.noise_decay
        if self.statistics[0] < expansion_bound:
            self.w = min(1, self.w + dw)
        else:
            self.w = max(0, self.w - dw)

    def get_h(self, state, goal):
        return self.H[state]

    def weighted_h(self, start, goal):
        if start == goal:
            return 0
        return self.w * self.get_h(start, goal) + (1 - self.w) * start.get_h(goal)

    def train(self, options):
        cls = prob.get_domain_class(options.training_domain)
        self.init_h(len(cls.get_goal(options.training_size).as_tensor()), options)
        return

    def load(self, path):
        f = open(path + '_H_vals.pkl', "rb")
        self.H = pickle.load(f)
        f = open(path + '_counts.pkl', "rb")
        self.counter = pickle.load(f)

    def save(self, path):

        f = open(path + "_H_vals.pkl", "wb")
        pickle.dump(self.H, f)
        f = open(path + "_counts.pkl", "rb")
        pickle.dump(self.counter,  f)

    def __str__(self):
        return "TMC"
