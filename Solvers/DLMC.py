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
from ResNN import ResNN
from buffers import ReplayBufferSearch, PrioritizedReplayBufferSearch

import logging

import pickle
import os

def weighted_loss(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)/(y_true + 1)) , axis=-1)

# Deep Learning Greedy Monte-Carlo
class DLMC(AbstractSolver):
    buffer_size = 20000 * 5
    batch_size = int(2**14)
    sample_size = 10 ** 5
    boostrap_update_size = 3 * buffer_size

    def __init__(self, problem=None, options=None):
        super(DLMC, self).__init__()
        self.greedy_solver = AStar()
        self.greedy_solver.__init__(problem, options, return_expanded = True)
        self.w = 0
        self.counter = 0
        self.update_target = 100
        self.current_problem = None

        if options is not None:
            self.update_target = options.update_target
            self.expansion_bound = options.expansion_bound
            self.train = options.train
            self.counter = options.resume if options.resume is not None else 0
            self.w = 0 if self.train else 1

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
        model = ResNN([input_dim] + layers + [1])
        target_model = ResNN([input_dim] + layers + [1])

        target_model.set_weights(model.get_weights())
        #model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        model.compile(lr=learning_rate)
        self.h = model

        self.target_model = target_model

    def train(self, options):
        self.greedy_solver.h_func = None
        self.buffer = ReplayBufferSearch(self.buffer_size)

        cls = prob.get_domain_class(options.training_domain)
        self.init_h(len(cls.get_goal(options.training_size).as_tensor()), options)
        self.save_path = options.save_path
        if not os.path.exists(os.path.join(self.save_path, 'weights')):
            os.makedirs(os.path.join(self.save_path, 'weights'))
            os.makedirs(os.path.join(self.save_path, 'buffer'))
        return

    def solve(self, problem, dw = 0.02):
        # A single run
        self.current_problem = problem

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
            self.w = max(0, self.w - dw) if self.train else 1
            return
        expanded.reverse()

        if not self.train:
            return

        # Update weight
        self.remember(expanded)
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

    def remember(self, states):
        target_values = self.get_target_values(states)
        for i, state in enumerate(states):
            if state == self.current_problem.goal:
                cost = 0
            else:
                cost = 1
            self.buffer.append(state.as_tensor(), [state, cost], target_values[i])

    def replay(self):
        self.counter += 1
        if self.counter % self.update_target == 0:
            print("Updating Target Weights...")
            self.target_model.set_weights(self.h.get_weights())

            self.save_weights_memory()
            target_values = self.get_target_values([m[0] for m in self.buffer.memory])
            self.buffer.update_target_buffer(target_values)
        #self.buffer.set_predict_function(self.h.predict)
        x, y = self.buffer.sample(self.sample_size)
        self.h.run_epoch(x=np.array(x), y=np.array(y), batch_size=DLMC.batch_size, verbose=1)

    def get_target_value(self, state, goal):
        """
        Depricated
        """
        new_states = state.get_successors()
        return min([self.target_bounded_predict(state, goal) for state in new_states])

    def get_target_values(self, nodes):
        target_values = np.zeros((len(nodes)))
        X = set()
        update_idx = 0
        for i, x in enumerate(nodes):
            if x != self.current_problem.goal:
                for state in x.get_successors():
                    X.add(tuple(state.as_tensor()))
            if len(X) > self.boostrap_update_size:
                X = list(X)
                self.update_target_fn(nodes, list(X), target_values, update_idx, i + 1)
                update_idx = i + 1
                X = set()
        if len(nodes) != update_idx:
            self.update_target_fn(nodes, list(X), target_values, update_idx, len(nodes))
        return target_values

    def update_target_fn(self, nodes, X, target_values, start, end):
        vals = self.h.predict(np.array(X))
        table = dict(zip(X, vals))
        del X, vals
        for i in range(start, end):
            x = nodes[i]
            if x == self.current_problem.goal:
                target_values[i] = 0
                cost = 0
            else:
                cost = 1 # Hard code vaues! Consider storing costs in an array
                min_vals = []
                for state in x.get_successors():
                    hand_crafted_h = state.get_h(self.current_problem.goal)
                    if hand_crafted_h == 0:
                        min_vals.append(cost)
                    else:
                        min_vals.append(cost + max(table[tuple(state.as_tensor())], hand_crafted_h))
                #min_target = self.get_target_value(x, self.current_problem.goal)
                target_values[i] = min(min_vals)
            i += 1
        return target_values

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

    def save_weights_memory(self):
        path = self.save_path
        episode = self.counter
        print("Saving Weights")
        self.h.save(os.path.join(path, 'weights', 'solver_{:07d}.pkl'.format(episode)))
        self.buffer.save(os.path.join(path, 'buffer', 'solver_{:07d}'.format(episode)))

    def load_weights_memory(self, episode):

        print("Loading and resuming weights")
        path = self.save_path
        self.load_model(os.path.join(path, 'weights', 'solver_{:07d}.pkl'.format(episode)))
        self.buffer.load(os.path.join(path, 'buffer', 'solver_{:07d}'.format(episode)))
        self.target_model.load_model(os.path.join(path, 'weights', 'solver_{:07d}.pkl'.format(episode - self.update_target)))

    def load_model(self, model_path):
        self.h.load_model(model_path)

    def __str__(self):
        return "DLMC"
