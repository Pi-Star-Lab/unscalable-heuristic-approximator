from keras.models import Sequential
import keras.models
from keras.layers import Dense
from keras.optimizers import Adam
from Solvers.LRTA import LRTA
import numpy as np
import random
import Utils


# DLRTA is similar to LRTA but instead of updating and consulting a heuristic table it uses a deep ANN
class DLRTA(LRTA):

    buffer_size = 100
    batch_size = 30
    clone_model_every = 100

    def __init__(self,problem=None,options=None):
        self.buffer = set()
        super(DLRTA, self).__init__()
        self.epsilon_greedy = 0.1

    def init_h(self,problem,options):
        input_dim = problem.get_state_size()
        try:
            layers = Utils.parse_list(options.layers)
        except ValueError:
            raise Exception('layers argument doesnt follow int array conventions i.e., [<int>,<int>,<int>,...]')
        except:
            raise Exception('must specify hidden layers for deep network. e.g., "-l [10,32]"')
        learning_rate = options.learning_rate
        # Neural Net for Deep-LRTA
        model = Sequential()
        model.add(Dense(layers[0], input_dim = input_dim, activation='relu'))
        for l in layers[1:]:
            model.add(Dense(l, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        self.h = model
        self.h_target = None
        self.count_set = 0

    def get_h(self,state,use_h,goal):
        if state == goal:
            return 0
        h=0
        if self.h_target:
            h = np.asscalar(self.h_target.predict(self.to_tensor(state)))
        if use_h:
            h = max(h,state.get_h(goal))
        else:
            h = max(h, 0)
        return h

    def set_h(self,state,h,problem):
        if state not in self.buffer:
            if len(self.buffer) == DLRTA.buffer_size:
                self.buffer.pop()
            self.buffer.add(state)
        self.count_set += 1
        target_h = h
        self.h.fit(x=self.to_tensor(state), y=np.array([target_h]), epochs=1, verbose=0)
        self.h.fit(x=self.to_tensor(problem.goal), y=np.array([0]), epochs=1, verbose=0)
        self.replay(problem)
        if self.count_set % self.clone_model_every == 0:
            self.h_target = keras.models.clone_model(self.h)
            self.h_target.set_weights(self.h.get_weights())

    def replay(self, problem):
        batch_size = min(DLRTA.batch_size,len(self.buffer))
        minibatch = random.sample(self.buffer, batch_size)
        for state in minibatch:
            if state != problem.goal:
                # if state == problem.start:
                #     print("Start: before {}".format(self.get_h(state,self.use_h,problem.goal)))
                # else:
                #     print("before {}".format(self.get_h(state,self.use_h,problem.goal)))
                target_h = self.get_best_successor(state, problem)[1]
            else:
                # print("Goal: before {}".format(self.get_h(state,self.use_h,problem.goal)))
                target_h = 0
            # print("target = {}".format(target_h))
            self.h.fit(x=self.to_tensor(state), y=np.array([target_h]), epochs=1, verbose=0)
            # print("After {}".format(self.get_h(state,self.use_h,problem.goal)))
            # print("-----------------")

    def to_tensor(self,state):
        x = state.as_tensor()
        return np.reshape(state.as_tensor(), [1, len(x)])

    def __str__(self):
        return "DLRTA*"