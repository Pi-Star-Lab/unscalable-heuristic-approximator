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
        print(self.greedy_solver)

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
        model.compile(loss=weighted_loss, optimizer=Adam(lr=learning_rate))
        self.h = model

    def train(self,options):
        self.greedy_solver.h_func = None
        self.buffer_x = deque(maxlen=DLMC.buffer_size)
        self.buffer_y = deque(maxlen=DLMC.buffer_size)
        self.memory = deque(maxlen=DLMC.buffer_size)
        cls = prob.get_domain_class(options.training_domain)
        self.init_h(len(cls.get_goal(options.training_size).as_tensor()), options)
        for i in range(options.training_episodes):
            p = prob()
            p.generate_random(i, cls, cls.get_goal(options.training_size))
            self.greedy_solver.__init__(p,options)
            path = self.greedy_solver.solve(p)
            print("length of Path: {}".format(len(path)))
            path.reverse()
            for x in range(len(path)):
                print(x, path[x].get_h(p.goal))
                self.remember(path[x], min(x, path[x].get_h(p.goal)))
            print("training episode {}/{}".format(i,options.training_episodes))
        self.h.fit(x=np.array(self.buffer_x), y=np.array(self.buffer_y), batch_size=DLMC.batch_size, epochs=100, verbose=1)
        print("training complete")
        #self.h.compile(loss='mse', optimizer=Adam(lr=0.00001))

    def solve(self, problem):
        # A single run
        self.greedy_solver.__init__()
        self.greedy_solver.h_func = self.get_h
        path = self.greedy_solver.solve(problem)
        print("Path found. Length of Path: {}".format(len(path)))
        path.reverse()
        for x in range(len(path)):
            if path[x].get_h(problem.goal) > x:
                import sys
                print("Not admissible H")
                sys.exit(1)
            #self.remember(path[x],min(x,path[x].get_h(problem.goal)))
            self.remember(path[x],x)
        #self.weighted_reply()
        self.replay()
        self.statistics = copy.deepcopy(self.greedy_solver.statistics)
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
