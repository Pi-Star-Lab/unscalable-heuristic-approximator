import optparse
import sys
import os
import random
from Solvers.Abstract_Solver import AbstractSolver as sol
from Domains.Problem_instance import ProblemInstance as prob
import Solvers.Available_solvers as avs
import random

resultdir = "Results/"
problemdir = "Problem_instances/"

# register arguments and set default values
def readCommand(argv):
    parser = optparse.OptionParser(description = 'Run experiments with the DLRTA* algorithm.')
    parser.set_defaults(heuristic=None, seed=random.random())
    parser.add_option("-p", "--problems", dest="probfile",
                      help="the file containing the problem instances (must be located in " + problemdir)
    parser.add_option("-o", "--outfile", dest="outfile",
                      help="write results to FILE", metavar="FILE")
    parser.add_option("-t", "--heuristic",
                      action="store_true", dest="use_heuristic", default=False,
                      help="use informative heuristic function (defined for each domaing, e.g., "
                           "gap heuristic for pancake problem")
    parser.add_option("-s", "--solver", dest="solver", type="string",
                      help='solver from ' + str(avs.solvers))
    parser.add_option("-e", "--episodes", type="int", dest="episodes", default=1,
                      help='number of episodes for solving each problem instance. relevant for h learning algorithms')
    parser.add_option("-l", "--layers", dest="layers", type="string", default="[30]",
                      help='size of hidden layers in a Deep neural net. e.g., "[10,15]" creates a net where the'
                           'input layer is connected to a layer of size 10 that is connected to a layer of size 15'
                           ' that is connected to the output')
    parser.add_option("-r", "--learning_rate", dest="learning_rate", type="float", default=1,
                      help='the learning rate for training the deep neural net')
    parser.add_option("-d", "--seed", type="int", dest="seed", default=random.randint(0, 9999999999),
                      help='seed integer for random stream')

    (options, args) = parser.parse_args(argv)
    return options


if __name__ == '__main__':
    options = readCommand(sys.argv)

    # Create result file if one doesn't exist
    if not os.path.exists(os.path.join(resultdir, options.outfile + '.csv')):
        with open(os.path.join(resultdir, options.outfile + '.csv'), 'w+') as result_file:
            result_file.write(sol.get_out_header()+'\n')

    random.seed(options.seed)
    solver = avs.get_solver_class(options.solver)()
    with open(os.path.join(problemdir, options.probfile + '.txt'), 'r') as problem_file:
        with open(os.path.join(resultdir, options.outfile + '.csv'), 'a+') as result_file:
            line = problem_file.readline()
            p = prob()
            while line:
                p.read_in(line)
                print(p)
                solver.__init__(p,options)
                for e in range(options.episodes):
                    solver.solve(p)
                    result_file.write(str(e) + ',' + solver.get_stat(p) + '\n')
                line = problem_file.readline()
