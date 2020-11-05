from Solvers.Abstract_Solver import AbstractSolver, Statistics
import random
import numpy as np
import Utils
from collections import deque
from Solvers.A_Star import AStar
import copy
from Domains.Problem_instance import ProblemInstance as prob
from statistics import mean
from FCNN import FCNN
from buffers import ReplayBufferSearch, PrioritizedReplayBufferSearch

import logging

import pickle
import os

def weighted_loss(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)/(y_true + 1)) , axis=-1)

# Deep Learning Greedy Monte-Carlo
class DLMC(AbstractSolver):
    buffer_size = 20000 * 5
    batch_size = int(1e10)
    sample_size = 10 ** 4

    def __init__(self, problem=None, options=None):
        super(DLMC, self).__init__()
        self.greedy_solver = AStar()
        self.greedy_solver.__init__(problem,options, return_expanded = True)
        self.w = 0
        self.counter = 0

        if options is not None:
            self.update_target = options.update_target
            self.expansion_bound = options.expansion_bound
        else:
            self.update_target = 100

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
        model = FCNN([input_dim] + layers + [1])
        target_model = FCNN([input_dim] + layers + [1])

        target_model.set_weights(model.get_weights())
        #model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        model.compile(lr=learning_rate)
        self.h = model

        self.target_model = target_model

    def train(self, options):
        self.greedy_solver.h_func = None
        self.buffer = PrioritizedReplayBufferSearch(self.buffer_size)

        cls = prob.get_domain_class(options.training_domain)
        self.init_h(len(cls.get_goal(options.training_size).as_tensor()), options)
        self.save_path = options.save_path
        return

    def solve(self, problem, dw = 0.02):
        # A single run
        self.greedy_solver.__init__(return_expanded = True)
        self.greedy_solver.h_func = self.weighted_h
        path,expanded = self.greedy_solver.solve(problem, expansion_bound = 8 * self.expansion_bound)

        self.statistics = copy.deepcopy(self.greedy_solver.statistics)
        self.statistics[Statistics.Weights.value] = self.w

        if path:
            print("Path found. Length of Path: {}".format(len(path)))
        else:
            print("Couldn't find path under {} expansions".format(8 * self.expansion_bound))
            print("Dropping this episode and reducing the weight")
            self.w -= dw
            return
        expanded.reverse()

        for x in range(len(expanded)):
            if expanded[x] == problem.goal:
                cost = (0, problem.goal)
            else:
                cost = (1, problem.goal)
            self.remember(expanded[x], cost)
        self.replay()
        if self.statistics[0] < self.expansion_bound:
            self.w = min(1, self.w + dw)
        else:
            self.w = max(0, self.w - dw)
        self.greedy_solver.noise_std = self.greedy_solver.noise_std * AStar.noise_decay

    def get_h(self, state, goal):
        if state == goal:
            return 0
        h = np.asscalar(self.h.predict(self.reshape(state)))
        return h

    def target_predict(self, state, goal):
        if state == goal:
            return 0
        h = np.asscalar(self.target_model.predict(self.reshape(state)))
        return h

    def target_weighted_predict(self, state, goal):
        return self.w * self.target_predict(state, goal) + \
                (1 - self.w) * state.get_h(goal)

    def target_bounded_predict(self, state, goal):
        return max(self.target_predict(state, goal), state.get_h(goal))

    def get_target_value(self, state, goal):
        new_states = state.get_successors()
        return min([self.target_bounded_predict(state, goal) for state in new_states])

    def remember(self, state, h):
        if h[0] == 0:
            target = 0
        else:
            target = h[0] + self.get_target_value(state, h[1])
        self.buffer.append(state.as_tensor(), [state, h], target)

    def replay(self):
        self.counter += 1
        if self.counter % self.update_target == 0:# or \
                #len(self.buffer_target) != len(self.memory):
            print("Updating Target Weights...")
            self.target_model.set_weights(self.h.get_weights())

            path = os.path.join(self.save_path, "solver_{:07d}".format(self.counter))
            self.save(path)

            self.buffer.update_target_values(self.get_target_value)
        self.buffer.set_predict_function(self.h.predict)
        x, y = self.buffer.sample(self.sample_size)
        self.h.run_epoch(x=np.array(x), y=np.array(y), batch_size=DLMC.batch_size, verbose=1)

    def reshape(self, state):
        x = state.as_tensor()
        return np.reshape(state.as_tensor(), [1, len(x)])

    def save(self, path):
        f = open(path + '.pkl', "wb")
        pickle.dump(self, f)

    def load(self, path):
        if not os.path.exists(path):
            raise("Wrong Solver path")
        f = open(path, "rb")
        self = pickle.load(f)

    def save_weights_memory(self, model_path, memory):
        self.h.save(model_path + '.pkl')
        f = open(memory + ".pkl", "wb")
        pickle.dump(self.memory, f)
        f = open(memory + "_x.pkl", "wb")
        pickle.dump(self.buffer_x, f)
        f = open(memory + "_target.pkl", "wb")
        pickle.dump(self.buffer_target, f)

    def load_weights_memory(self, model_path, memory):

        print("Loading and resuming weights")
        self.h = keras.models.load_model(model_path + ".pkl")
        self.target_model = keras.models.load_model(model_path + ".pkl")
        f = open(memory + '.pkl', "rb")
        self.memory = pickle.load(f)
        f = open(memory + "_x.pkl", "rb")
        self.buffer_x = pickle.load(f)
        f = open(memory + "_target.pkl", "rb")
        self.buffer_target = pickle.load(f)

    def __str__(self):
        return "DLMC"
