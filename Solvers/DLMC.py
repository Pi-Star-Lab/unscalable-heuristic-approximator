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

import tensorflow as tf
import logging
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
os.environ['KMP_WARNINGS'] = '0'

def weighted_loss(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)/(y_true + 1)) , axis=-1)

# Deep Learning Greedy Monte-Carlo
class DLMC(AbstractSolver):
    buffer_size = 20000 * 5
    batch_size = 2 ** 10

    def __init__(self, problem=None, options=None):
        super(DLMC, self).__init__()
        self.greedy_solver = AStar()
        self.greedy_solver.__init__(problem,options, return_expanded = True)
        self.w = 0
        self.counter = 0

        ## Debuging code ###
        self.optimal_states = []
        self.initial_state_hval = []
        self.final_state_hval = []
        ## end ####

        if options is not None:
            self.update_target = options.update_target
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
        model = Sequential()
        target_model = Sequential()
        model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
        target_model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
        for l in layers[1:]:
            model.add(Dense(l, activation='relu'))
            target_model.add(Dense(l, activation='relu'))
        model.add(Dense(1))
        target_model.add(Dense(1))
        target_model.set_weights(model.get_weights())
        #model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        model.compile(loss="mse", optimizer=SGD(lr=learning_rate))
        self.h = model

        model.save("seed_2_model_0.pkl")
        self.target_model = target_model
        model.summary()

    def train(self, options):
        self.greedy_solver.h_func = None
        self.buffer_x = deque(maxlen=DLMC.buffer_size)
        self.memory = deque(maxlen=DLMC.buffer_size)
        self.buffer_target = deque(maxlen=DLMC.buffer_size) #just to save computational speed!
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
    def solve(self, problem, dw = 0.02, expansion_bound = 500):
        # A single run
        self.greedy_solver.__init__(return_expanded = True)
        self.greedy_solver.h_func = self.weighted_h
        path,expanded = self.greedy_solver.solve(problem)


        ###### Debug Code ######
        """
        if len(path) == 5:
            self.optimal_states = path
        """
        ########
        print("Path found. Length of Path: {}".format(len(path)))

        expanded.reverse()

        for x in range(len(expanded)):
            if expanded[x] == problem.goal:
                cost = (0, problem.goal)
            else:
                cost = (1, problem.goal)
            #self.remember(path[x],min(x,path[x].get_h(problem.goal)))
            self.remember(expanded[x], cost)
            #print(self.get_h(path[x], problem.goal), x)
        self.replay()
        self.statistics = copy.deepcopy(self.greedy_solver.statistics)
        self.statistics[Statistics.Weights.value] = self.w
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

    def target_predict(self, state, goal):
        if state == goal:
            return 0
        h = np.asscalar(self.target_model.predict(self.reshape(state)))
        return h

    def target_weighted_predict(self, state, goal):
        return self.w * self.target_predict(state, goal) + \
                (1 - self.w) * state.get_h(goal)

    def get_target_value(self, state, goal):
        new_states = state.get_successors()
        return min([self.target_weighted_predict(state, goal) for state in new_states])

    def remember(self, state, h):
        self.buffer_x.append(state.as_tensor())
        self.memory.append([state, h])
        min_target_val = self.get_target_value(state, h[1])
        self.buffer_target.append(h[0] + min_target_val)

    def replay(self):
        self.counter += 1
        if self.counter % self.update_target == 0 or \
                len(self.buffer_target) != len(self.memory):
            #self.initial_state_hval.append(self.get_h(self.optimal_states[0],\
            #        self.optimal_states[-1]))
            #self.final_state_hval.append(self.get_h(self.optimal_states[-2], \
            #        self.optimal_states[-1]))
            print("Updating Target Weights...")
            self.target_model.set_weights(self.h.get_weights())
            self.save("Models/pancake7_seed_test_1/weights/model_dump_{}".format(self.counter),
                    "Models/pancake7_seed_test_1/buffer/memory_10k_5_{}".format(self.counter))
            #self.h.save("Models/model_dump_" + str(self.counter) + ".pkl")
            i = 0
            for x,y in self.memory:
                if x == y[1]:
                    self.buffer_target[i] = 0
                else:
                    min_target_val = self.get_target_value(x, y[1])
                    self.buffer_target[i] = y[0] + min_target_val
                    if i < 45:
                        print(y[0], min_target_val, end = ",   ")
                if i < 45:
                    print(i, self.buffer_target[i], "actual", self.get_h(x, y[1]))
                i += 1

        self.h.fit(x=np.array(self.buffer_x), y=np.array(self.buffer_target), batch_size=DLMC.batch_size, epochs=1, verbose=1, shuffle=False)

    def reshape(self, state):
        x = state.as_tensor()
        return np.reshape(state.as_tensor(), [1, len(x)])

    def save(self, model_path, memory):
        self.h.save(model_path + '.pkl')
        f = open(memory + ".pkl", "wb")
        pickle.dump(self.memory, f)
        f = open(memory + "_x.pkl", "wb")
        pickle.dump(self.buffer_x, f)
        f = open(memory + "_target.pkl", "wb")
        pickle.dump(self.buffer_target, f)
        #f = open(memory + "_y.pkl", "wb")
        #pickle.dump(self.buffer_y, f)

    def load(self, model_path, memory):

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
