from Solvers.Abstract_Solver import AbstractSolver, Statistics
import random
from keras.models import Sequential
import keras.models
from keras.layers import Dense
from keras.optimizers import Adam,SGD
import numpy as np
import Utils
from collections import deque
from Solvers.A_Star import AStar
import copy
from Domains.Problem_instance import ProblemInstance as prob
from statistics import mean
import keras.backend as K


def weighted_loss(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)/(y_true + 1)) , axis=-1)

# Deep Learning Greedy Monte-Carlo
class DLMC(AbstractSolver):
    buffer_size = 1000000
    batch_size = 2 ** 10

    def __init__(self, problem=None, options=None):
        super(DLMC, self).__init__()
        self.greedy_solver = AStar()
        self.greedy_solver.__init__(problem,options)
        self.w = 0
        print(self.greedy_solver)

    def weighted_h(self, start, goal):
        return self.w * self.get_h(start, goal) + (1 - self.w) * start.get_h(goal)

    def init_h(self, input_dim, options):
        try:
            layers = Utils.parse_list(options.layers)
        except ValueError:
            raise Exception('layers argument doesnt follow int array conventions i.e., [<int>,<int>,<int>,...]')
        except:
            raise Exception('must specify hidden layers for deep network. e.g., "-l [10,32]"')
        learning_rate = options.learning_rate
        # Neural Net for h values
        model = Sequential()
        model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
        for l in layers[1:]:
            model.add(Dense(l, activation='relu'))
        model.add(Dense(1))
        #model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        model.compile(loss="mse", optimizer=Adam(lr=learning_rate))
        self.h = model

    def train(self, options, dw = 0.05):
        self.greedy_solver.h_func = None
        self.buffer_x = deque(maxlen=DLMC.buffer_size)
        self.buffer_y = deque(maxlen=DLMC.buffer_size)
        self.memory = deque(maxlen=DLMC.buffer_size)
        cls = prob.get_domain_class(options.training_domain)
        self.init_h(len(cls.get_goal(options.training_size).as_tensor()), options)

        return
        """
        for i in range(options.training_episodes):
            p = prob()
            p.generate_random(i, cls, cls.get_goal(options.training_size))
            self.greedy_solver.__init__(p,options)
            path = self.greedy_solver.solve(p)
            print("length of Path: {}".format(len(path)))
            path.reverse()
            for x in range(len(path)):
                #loop through all next_states
                self.remember(path[x], min(x, path[x].get_h(p.goal)))
            print("training episode {}/{}".format(i,options.training_episodes))
        self.h.fit(x=np.array(self.buffer_x), y=np.array(self.buffer_y), batch_size=DLMC.batch_size, epochs=100, verbose=1)
        print("training complete")
        #self.h.compile(loss='mse', optimizer=Adam(lr=0.00001))
        """
    def solve(self, problem, dw = 0.05, expansion_bound = 500):
        # A single run
        self.greedy_solver.__init__()
        self.greedy_solver.h_func = self.weighted_h
        path = self.greedy_solver.solve(problem)
        print("Path found. Length of Path: {}".format(len(path)))
        path.reverse()
        for x in range(len(path)):
            if path[x].get_h(problem.goal) > x:
                import sys
                print("Not admissible H")
                sys.exit(1)
            next_states = path[x].get_successors()
            if path[x] == problem.goal:
                target = 0
            else:
                target = float("inf")
            for next_state in next_states:
                cost = 1 + self.get_h(next_state, problem.goal)
                target = min(cost, target)
            #self.remember(path[x],min(x,path[x].get_h(problem.goal)))
            self.remember(path[x], target)
            print(self.get_h(path[x], problem.goal))
        self.replay()
        self.statistics = copy.deepcopy(self.greedy_solver.statistics)
        print(self.statistics, "W:", self.w)
        if self.statistics[0] < expansion_bound:
            self.w = min(1, self.w + dw)
        else:
            self.w = max(0, self.w - dw)
        self.greedy_solver.noise_std = self.greedy_solver.noise_std * AStar.noise_decay

    def get_h(self, state, goal):
        if state == goal:
            return 0
        h = np.asscalar(self.h.predict(self.reshape(state)))
        return h

    def remember(self, state, h):
        self.buffer_x.append(state.as_tensor())
        self.buffer_y.append(np.array([h]))
        self.memory.append([self.reshape(state), np.array([h])])

    def replay(self):
        self.h.fit(x=np.array(self.buffer_x), y=np.array(self.buffer_y), batch_size=DLMC.batch_size, epochs=1, verbose=1)

    # def weighted_reply(self):
    #     if len(self.memory) < DLMC.batch_size:
    #         return
    #
    #     samples = random.sample(self.memory, 32)
    #     for sample in samples:
    #         state, h = sample
    #         self.h.optimizer.lr = 0.000001 / (h+1)
    #         #self.h.optimizer.lr = 0.000001
    #         self.h.fit(state, h, epochs=1, verbose=0)

    def reshape(self, state):
        x = state.as_tensor()
        return np.reshape(state.as_tensor(), [1, len(x)])

    def __str__(self):
        return "DLMC"
