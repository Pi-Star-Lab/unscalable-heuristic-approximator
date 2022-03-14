import optparse
import sys
import random
import os
from Domains.Problem_instance import ProblemInstance as prob

import numpy as np
import math
from ResNN import ResNN
from FCNN import FCNN
from ResNN import ResNN
from Solvers.Abstract_Solver import Statistics
from Solvers.A_Star import AStar
import torch
from sklearn.utils import shuffle
indir = "dataset/"

LOSS_THRESHOLD = 0.1
MAX_EPOCHS = 300

class scaled_MSE_loss(torch.nn.Module):
    def __init__(self):
        super(scaled_MSE_loss, self).__init__()

    def forward(self, target, output):
        #return torch.mean((1 / (target[:, 0] ** 2) + 1 / (output[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)
        return torch.mean((1 / (target[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)

class pth_norm_loss(torch.nn.Module):
    def __init__(self):
        super(pth_norm_loss, self).__init__()
        self.p = 10

    def forward(self, target, output):
        #return torch.mean((1 / (target[:, 0] ** 2) + 1 / (output[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)
        return torch.mean((target[:, 0] - output[:, 0]) ** self.p)

class weighted_MSE_loss(torch.nn.Module):
    def __init__(self):
        super(weighted_MSE_loss, self).__init__()

    def forward(self, target, output, weights):
        #return torch.mean((1 / (target[:, 0] ** 2) + 1 / (output[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)
        return torch.mean(weights * (target[:, 0] - output[:, 0]) ** 2)

class SE_loss(torch.nn.Module):
    def __init__(self):
        super(SE_loss, self).__init__()

    def forward(self, target, output):
        #return torch.mean((1 / (target[:, 0] ** 2) + 1 / (output[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)
        return torch.sum((target[:, 0] - output[:, 0]) ** 2)

class sigmod_based_MSE_loss(torch.nn.Module):
    def __init__(self):
        super(sigmod_based_MSE_loss, self).__init__()

    def forward(self, target, output):
        #return torch.mean((1 / (target[:, 0] ** 2) + 1 / (output[:, 0] ** 2)) * (target[:, 0] - output[:, 0]) ** 2)
        se = (target[:, 0] - output[:, 0]) ** 2
        epsilon2 = 1
        return torch.mean(torch.sigmoid(((4 * se) / epsilon2) - 1) * se)


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

def weighted_mse(labels,predicted):
    idx, counts = np.unique(labels, return_counts=True)
    mapping = np.zeros((idx.max()))
    for i,index in enumerate(idx):
        mapping[index] = counts[i]
    weights = (len(labels) - mapping[labels]) / len(labels)
    return np.mean(weights * (labels-predicted) ** 2, axis = 0)

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
        self.neuron_range = [2, 48000]
        self.layer_range = [0, 160]
        self.layer_size = 500
        #self.neuron_range = [600, 100000]
        #self.neuron_range = [400, 1000000]
        self.last_test_loss = None

    def search_width(self, problem_size):
        keys = list(sorted(self.saved_breaking_points.keys()))
        #mini_idx, maxi_idx = keys[0], keys[-1]
        mini_idx, maxi_idx = self.neuron_range

        while maxi_idx > mini_idx + 1:
            mid = (maxi_idx + mini_idx) // 2
            print(mini_idx, mid, maxi_idx, self.saved_breaking_points)

            input_dim = len(self.cls.get_goal_dummy(problem_size).as_tensor())
            model = FCNN([input_dim] + [mid] + [1], use_batch_norm=True)
            #model.compile(lr=2e-3)
            #model.compile(lr=2e-3, loss = scaled_MSE_loss())
            model.compile(lr=2e-3, loss = sigmod_based_MSE_loss())
            #percent_factor = 0.2
            print("=" * 40, mid, "=" * 40, problem_size)
            num_samples = self.max_steps #TODO: or (percent_factor / 100) * math.factorial(problem_size)
            does_fit = self.does_fit_2(problem_size, model, num_samples, 0.5)
            if does_fit:
                maxi_idx = mid
                self.saved_breaking_points[maxi_idx] = problem_size
            else:
                mini_idx = mid
        self.model = model
        return maxi_idx

    def search_depth(self, problem_size):
        keys = list(sorted(self.saved_breaking_points.keys()))
        #mini_idx, maxi_idx = keys[0], keys[-1]
        mini_idx, maxi_idx = self.layer_range
        layer_size = self.layer_size
        layer_size = problem_size ** 2 + 3  ## TODO:extra addition here!
        while maxi_idx > mini_idx + 1:
            mid = (maxi_idx + mini_idx) // 2
            print(mini_idx, mid, maxi_idx, self.saved_breaking_points)

            input_dim = len(self.cls.get_goal_dummy(problem_size).as_tensor())
            #model = FCNN([input_dim] + [layer_size] * mid + [1], use_batch_norm=True)
            if mid % 2 == 0:
                model = ResNN([input_dim] + [layer_size, layer_size, 1], (mid-1) // 2, use_batch_norm=False)
            else:
                model = ResNN([input_dim] + [layer_size, 1], (mid-1) // 2, use_batch_norm = False)
            #model.compile(lr=2e-3)
            model.compile(lr=2e-3, loss = scaled_MSE_loss())
            percent_factor = 0.2
            print("=" * 40, mid, "num layers", "=" * 40, problem_size)
            num_samples = self.max_steps #TODO: or (percent_factor / 100) * math.factorial(problem_size)
            does_fit = self.does_fit_2(problem_size, model, num_samples, 0.5)
            if does_fit:
                maxi_idx = mid
                self.saved_breaking_points[maxi_idx] = problem_size
            else:
                mini_idx = mid
        self.model = model
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

        #return shuffle(X, Y)
        return X, Y

    def split_data(self, X, Y):
        len_data = len(X)

        slice_idx = int(self.train_test_split * len_data)

        train_X = X[:slice_idx,:]
        train_Y = Y[: slice_idx]
        test_X = X[slice_idx:,:]
        test_Y = Y[slice_idx:]
        return train_X, train_Y, test_X, test_Y

    def does_fit(self, problem_size, model, num_samples):

        X, Y = self.get_data(problem_size, num_samples)

        train_X, train_Y, test_X, test_Y = self.split_data(X, Y)
        epoch = 1
        training_loss = float("inf")
        test_loss = float("inf")
        while training_loss > LOSS_THRESHOLD and epoch < MAX_EPOCHS:

            batch_size = 10000
            training_loss =  model.run_epoch(train_X, train_Y, batch_size=batch_size, verbose=0)
            prediction_value = model.predict(test_X, batch_size = batch_size)
            print([prediction_value.min(), prediction_value.max()], [test_Y.min(), test_Y.max()])
            test_loss = mse(test_Y, prediction_value)
            self.last_test_loss = test_loss
            epoch += 1
            print("Epoch:", epoch, "Training loss:", training_loss, "Test Loss:", test_loss)
        return False if epoch == MAX_EPOCHS else True

    def does_fit_2(self, problem_size, model, num_samples, threshold):
        """
        threshold = \epsilon / 2
        """

        X, Y = self.get_data(problem_size, num_samples)

        train_X, train_Y, test_X, test_Y = self.split_data(X, Y)
        epoch = 1
        training_loss = float("inf")
        test_loss = float("inf")
        mis_classified = float('inf')
        #while training_loss > 0 and epoch < MAX_EPOCHS:
        while mis_classified > 0 and epoch < MAX_EPOCHS:

            batch_size = 10000
            predict_trainset = model.predict(train_X, batch_size = batch_size)
            #print(train_X.shape, predict_trainset.shape)
            incorrect_idxs = np.abs(predict_trainset - train_Y) >= threshold
            #weights = incorrect_idxs.astype('float64')
            #incorrect_train_X, incorrect_train_Y = train_X[incorrect_idxs], train_Y[incorrect_idxs]
            #weights[weights < 0.01] = 0.1
            #weights = torch.Tensor(weights)
            #print(sum(weights), model.loss_fn.weights.shape, model.loss_fn.weights.min(), model.loss_fn.weights.max())
            #training_loss = model.run_epoch_weighted(train_X, train_Y, weights, batch_size=batch_size)
            training_loss = model.run_epoch(train_X, train_Y, batch_size=batch_size)
            prediction_value = model.predict(test_X, batch_size = batch_size)
            mis_classified = sum(incorrect_idxs)
            print(mis_classified, [prediction_value.min(), prediction_value.max()], [test_Y.min(), test_Y.max()])
            test_loss = mse(test_Y, prediction_value)
            self.last_test_loss = test_loss
            epoch += 1
            print("Epoch:", epoch, "Training loss:", training_loss, "Test Loss:", test_loss)
        return False if epoch == MAX_EPOCHS else True


    def get_test_results(self, layer_size, problem_size, num_samples):
        input_dim = len(self.cls.get_goal_dummy(problem_size).as_tensor())
        model = FCNN([input_dim] + [layer_size] + [1], use_batch_norm=False)
        model.compile(lr=2e-3)
        self.does_fit(problem_size, model, num_samples)
        return self.testset_results(model, problem_size, num_samples)

    def get_test_width_results(self, num_layer, problem_size, num_samples):
        input_dim = len(self.cls.get_goal_dummy(problem_size).as_tensor())
        layer_size = problem_size ** 2 + 3  ## TODO:extra addition here!
        mid = num_layer
        if mid % 2 == 0:
            model = ResNN([input_dim] + [layer_size, layer_size, 1], (mid-1) // 2, use_batch_norm=True)
        else:
            model = ResNN([input_dim] + [layer_size, 1], (mid-1) // 2, use_batch_norm = True)

        model.compile(lr=2e-3)
        for i in range(10):
            if self.does_fit(problem_size, model, num_samples):
                return self.last_test_loss
        #return self.testset_results(model, problem_size, num_samples)
        return False

    def testset_results(self, model, problem_size, num_samples):
        X, Y = self.get_data(problem_size, num_samples)
        len_data = len(X)

        slice_idx = int(self.train_test_split * len_data)

        train_X = X[:slice_idx,:]
        train_Y = Y[: slice_idx]
        test_X = X[slice_idx:,:]
        test_Y = Y[slice_idx:]
        batch_size = 3000
        prediction_value = model.predict(test_X, batch_size = batch_size)
        test_loss = mse(test_Y, prediction_value)
        return test_loss

    def get_search_results(self, layer_size, problem_size, num_samples):
        input_dim = len(self.cls.get_goal_dummy(problem_size).as_tensor())
        model = FCNN([input_dim] + [layer_size] + [1], use_batch_norm=True)
        model.compile(lr=2e-3)
        self.does_fit(problem_size, model, num_samples)
        return self.search_results(model, problem_size, num_samples)

    def search_results(self, model, problem_size, num_samples):
        X, Y = self.get_data(problem_size, num_samples)
        greedy_solver = AStar()
        greedy_solver.__init__(return_expanded = True)
        greedy_solver.h_func = self.get_h
        len_data = len(X)

        slice_idx = int(self.train_test_split * len_data)

        num = 120
        problems = self.get_states(problem_size, slice_idx, slice_idx + num)
        results = []
        self.model = model
        costs = []
        expansions = []
        for p in problems:
            sol, expanded = greedy_solver.solve(p)
            results.append((len(sol), len(expanded)))
            costs.append(len(sol))
            expansions.append(len(expanded))
        return costs, expansions

    def get_h(self, state, goal) :
        if state.is_solution():
            return 0
        h = np.asscalar(self.model.predict(self.reshape(state)))
        return h

    def reshape(self, state):
        x = state.as_tensor()
        #return np.reshape(state.as_tensor(), [1, len(x)])
        return np.expand_dims(state.as_tensor(), axis = 0)

    def get_states(self, problem_size, min_num, max_num):
        with open(os.path.join(indir, self.outfile + '_{}.txt'.format(problem_size)), 'r') as problem_file:
            samples = 0
            X = []
            while samples < max_num:
                samples += 1
                line = problem_file.readline()
                if samples < min_num:
                    continue
                p = prob()
                p.read_in(line[:line.rfind('$')])
                X.append(p)

        return X

def get_test_losses(tester, layer_sizes, problem_sizes):

    results = []
    logger = open("logger.txt", 'w')
    c_mean, c_std, e_mean, e_std = [], [], [], []
    for i in range(len(layer_sizes)-1, -1, -1):
        costs, expansions = (tester.get_search_results(layer_sizes[i], problem_sizes[i], tester.max_steps))
        print(problem_sizes[i], costs, expansions, file = logger)
        c_mean.append(np.mean(costs))
        c_std.append(np.std(costs))
        e_mean.append(np.mean(expansions))
        e_std.append(np.std(expansions))
    return c_mean, c_std, e_mean, e_std

if __name__ == '__main__':
    options = readCommand(sys.argv)
    t = Tester(options)
    breaking_point = []
    costs = []
    expansions = []
    test_losses = []
    #### TODO: delete this block######
    """
    #layer_sizes = [2, 5, 10, 29, 117, 672, 2191, 2709, 2636, 2504, 2243, 2064, 1862, 1670]
    layer_sizes = [2, 5, 10, 29, 117, 672, 2191, 2709, 2636, 2504, 2243, 2064, 1862, 1670]
    layer_sizes, problem_sizes = [1, 2, 4, 4, 5, 12, 13, 11, 9, 7, 4], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    test_set_results = []
    import sys
    for i in range(len(layer_sizes)):
        test_set_results.append(t.get_test_width_results(layer_sizes[i], problem_sizes[i], t.max_steps))
        if not test_set_results[-1]:
            print("ERRRROROOROR")
            sys.exit(1)
        print("=" * 150)
    print(test_set_results, problem_sizes)
    sys.exit(0)
    layer_sizes = [2, 4, 6, 12, 24, 51, 150, 441, 742, 931, 1002, 889, 789, 688]
    problem_sizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    avg_cost, std_cost, avg_expansions, std_expansions = get_test_losses(t, layer_sizes, problem_sizes)
    print("AVG COST     :", avg_cost)
    print("AVG EXPANSION:", avg_expansions)
    print("STD COST     :", std_cost)
    print("STD EXPANSION:", std_expansions)
    print("Problem Size :", problem_sizes)
    print("Layer Size   :", layer_sizes)
    """
    ##################################

    for i in range(options.min_size, options.max_size + 1):
            breaking_point.append(t.search_width(i))
            test_losses.append(t.testset_results(t.model, i, t.max_steps))
            print(i, ", ", end="", file=t.breaking_point_logger)
            t.breaking_point_logger.flush()
            #print("Problem_Size: ", i,  "=" * 40 + "Epochs: ", epoch, "="  * 40)
            k = list(sorted(t.saved_breaking_points.keys()))

    print(breaking_point, list(range(options.min_size, options.max_size + 1)), test_losses)
