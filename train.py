import optparse
import sys
import random
import os
from Domains.Problem_instance import ProblemInstance as prob

import numpy as np
import math
from ResNN import ResNN
from FCNN import FCNN
from Solvers.Abstract_Solver import Statistics
from Solvers.A_Star import AStar
import torch
indir = "dataset/"

LOSS_THRESHOLD = 0.1
MAX_EPOCHS = 300

class scaled_MSE_loss(torch.nn.Module):
    def __init__(self):
        super(scaled_MSE_loss, self).__init__()

    def forward(self, target, output):
        return torch.mean((1 / (target[:, 0] ** 2) + 1 / (output[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)
        #return torch.mean((1 / (target[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)


def readCommand(argv):
    parser = optparse.OptionParser(description = 'Create problem instances and store them to file in dir'
                                                 '"Problem_instances".')
    parser.add_option("-o", "--outfile", dest="outfile", type="string",
                      help="write output to FILE", metavar="FILE")
    parser.add_option("-s", "--seed", type="int", dest="seed", default=random.randint(0, 9999999999),
                      help = 'seed integer for random stream')
    parser.add_option("-d", "--domain", dest="domain", type="string",
                      help='problem domain from ' + str(prob.domains))
    parser.add_option("-n", "--max-size", type="int", dest="max_size",
                      help='max problem size. E.g., number of pancakes in pancake domain or '
                           'dimensionality in tile and rubik')
    parser.add_option("-m", "--min-size", type="int", dest="min_size",
                      help='min problem size. E.g., number of pancakes in pancake domain or '
                           'dimensionality in tile and rubik')
    parser.add_option("-i", "--max_steps", type="int", dest="max_steps", default=100,
                      help='Maxiumum number of steps')
    (options, args) = parser.parse_args(argv)
    return options

def parse_line(p, line):
    string = line
    string = string.split(prob.separator)
    p.index = string[0]
    p.cls = prob.get_domain_class(string[1])
    p.start = p.cls.parse_state(string[2])
    p.goal = p.cls.parse_state(string[3])
    label = int(string[-1][:-1])

    return p.start.as_tensor(), label

def mse(X, Y):
    return np.mean((X-Y) ** 2, axis = 0)

def scaled_mse(target, output):
    return np.mean((1 / (target ** 2) + 1 / (output ** 2)) * (target-output) ** 2, axis = 0)

class Tester:
    def __init__(self, options):
        assert options.outfile and options.domain and options.max_size and options.min_size, "arguments must include: outfile, domain, " \
                                                                    "and problem size"
        random.seed(options.seed)
        self.cls = prob.get_domain_class(options.domain)
        self.max_steps = options.max_steps
        self.train_test_split = 0.8
        self.breaking_point_logger = open("breaking_points_binary.txt", "w")
        print("[", end = " ",file=self.breaking_point_logger)
        self.breaking_point_logger.flush()
        self.saved_breaking_points = {} # neurons to problem size mapping
        self.outfile = options.outfile
        #self.neuron_range = [2, 800000]
        self.neuron_range = [2, 5000]
        self.layer_range = [0, 8]
        self.layer_size = 500
        #self.neuron_range = [600, 100000]
        #self.neuron_range = [400, 1000000]

    def search_width(self, problem_size):
        keys = list(sorted(self.saved_breaking_points.keys()))
        #mini_idx, maxi_idx = keys[0], keys[-1]
        mini_idx, maxi_idx = self.neuron_range

        while maxi_idx > mini_idx + 1:
            mid = (maxi_idx + mini_idx) // 2
            print(mini_idx, mid, maxi_idx, self.saved_breaking_points)

            input_dim = len(self.cls.get_goal_dummy(problem_size).as_tensor())
            model = FCNN([input_dim] + [mid] + [1], use_batchnorm=False)
            model.compile(lr=2e-3)
            #model.compile(lr=2e-3, loss = scaled_MSE_loss())
            #percent_factor = 0.2
            print("=" * 40, mid, "=" * 40, problem_size)
            num_samples = self.max_steps #TODO: or (percent_factor / 100) * math.factorial(problem_size)
            does_fit = self.does_fit(problem_size, model, num_samples)
            if does_fit:
                maxi_idx = mid
                self.saved_breaking_points[maxi_idx] = problem_size
            else:
                mini_idx = mid
        return maxi_idx

    def search_depth(self, problem_size):
        keys = list(sorted(self.saved_breaking_points.keys()))
        #mini_idx, maxi_idx = keys[0], keys[-1]
        mini_idx, maxi_idx = self.layer_range

        while maxi_idx > mini_idx + 1:
            mid = (maxi_idx + mini_idx) // 2
            print(mini_idx, mid, maxi_idx, self.saved_breaking_points)

            input_dim = len(self.cls.get_goal_dummy(problem_size).as_tensor())
            print([input_dim] + [self.layer_size] * mid + [1])
            model = FCNN([input_dim] + [self.layer_size] * mid + [1], use_batchnorm=False)
            model.compile(lr=2e-3)
            #model.compile(lr=2e-3, loss = scaled_MSE_loss())
            #percent_factor = 0.2
            print("=" * 40, mid, "=" * 40, problem_size)
            num_samples = self.max_steps #TODO: or (percent_factor / 100) * math.factorial(problem_size)
            does_fit = self.does_fit(problem_size, model, num_samples)
            if does_fit:
                maxi_idx = mid
                self.saved_breaking_points[maxi_idx] = problem_size
            else:
                mini_idx = mid
        return maxi_idx

    def get_data(self, problem_size, num):
        with open(os.path.join(indir, self.outfile + '_{}.txt'.format(problem_size)), 'r') as problem_file:
            samples = 0
            X, Y = [], []
            while samples < num:
                p = prob()
                line = problem_file.readline()
                x, y = parse_line(p, line)
                X.append(x)
                Y.append(y)
                samples += 1

            X = np.array(X)
            Y = np.array(Y)

        return X, Y


    def does_fit(self, problem_size, model, num_samples):

        X, Y = self.get_data(problem_size, num_samples)
        len_data = len(X)

        slice_idx = int(self.train_test_split * len_data)

        train_X = X[:slice_idx,:]
        train_Y = Y[: slice_idx]
        test_X = X[slice_idx:,:]
        test_Y = Y[slice_idx:]
        epoch = 1
        training_loss = float("inf")
        test_loss = float("inf")
        while training_loss > LOSS_THRESHOLD and epoch < MAX_EPOCHS:

            batch_size = 3000
            training_loss =  model.run_epoch(train_X, train_Y, batch_size=batch_size)
            prediction_value = model.predict(test_X, batch_size = batch_size)
            print(prediction_value.min(), prediction_value.max())
            test_loss = mse(test_Y, prediction_value)
            epoch += 1
            print("Epoch where fit:", epoch, "Training loss:", training_loss, "Test Loss:", test_loss)
        return False if epoch == MAX_EPOCHS else True


if __name__ == '__main__':
    options = readCommand(sys.argv)
    t = Tester(options)
    breaking_point = []

    for i in range(options.min_size, options.max_size + 1):
            breaking_point.append(t.search_depth(i))
            print(i, ", ", end="", file=t.breaking_point_logger)
            t.breaking_point_logger.flush()
            #print("Problem_Size: ", i,  "=" * 40 + "Epochs: ", epoch, "="  * 40)
            k = list(sorted(t.saved_breaking_points.keys()))

    print(breaking_point, list(range(options.min_size, options.max_size + 1)))
