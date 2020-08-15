import optparse
import sys
import os
import random
from Solvers.Abstract_Solver import AbstractSolver as sol
from Solvers.Abstract_Solver import Statistics
from Solvers.A_Star import AStar
from Domains.Problem_instance import ProblemInstance as prob
import Solvers.Available_solvers as avs
import random
import tqdm
from tqdm import tqdm
import numpy as np
import Plotting

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
    # parser.add_option("-t", "--heuristic",
    #                   action="store_true", dest="use_heuristic", default=True,
    #                   help="use informative heuristic function (defined for each domaing, e.g., "
    #                        "gap heuristic for pancake problem")
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
    parser.add_option("-w", "--weight", type="float", dest="weight", default=1,
                      help='weight for weighted A* (w=1 for A* and w=0 for UCS)')
    parser.add_option("-n", "--noise", type="float", dest="noise", default=0,
                      help='nosie factor for the heuristic value so A* will diversify its solutions. '
                           'h=h*random.gauss(1, sigma=noise)')
    parser.add_option("-t", "--train", type="int", dest="training_episodes", default=0,
                      help='number of training episodes')
    parser.add_option("-a", "--tdomain", dest="training_domain", type="string",
                      help='problem domain to train on from ' + str(prob.domains))
    parser.add_option("-b", "--tsize", type="int", dest="training_size",
                      help='problem size to tran on. E.g., number of pancakes in pancake domain or '
                           'dimensionality in tile and rubik')
    parser.add_option("-c", "--smooth", type="int", dest="smoothing_window", default=30,
                      help='smoothen observations over')

    (options, args) = parser.parse_args(argv)
    return options


def run_baseline(blsolver, problem, options, blstats, l):
    blsolver.__init__(problem, options)
    blsolver.solve(problem)
    blstats.solution_cost[l] = blsolver.statistics[Statistics.Solution.value]
    blstats.expanded[l] = blsolver.statistics[Statistics.Expanded.value]
    blstats.generated[l] = blsolver.statistics[Statistics.Generated.value]

def update_ratio(stats, blstats, l, rstats):
    rstats.solution_cost[l] = float(stats.solution_cost[l])/blstats.solution_cost[l]
    rstats.expanded[l] = float(stats.expanded[l])/blstats.expanded[l]
    rstats.generated[l] = float(stats.generated[l])/blstats.generated[l]

# -p tile4 -o res -s dlmc -w 10 -e 100 -r 0.1 -l [24,24] -t 100 -a tile -b 4
if __name__ == '__main__':
    options = readCommand(sys.argv)

    # Create result file if one doesn't exist
    if not os.path.exists(os.path.join(resultdir, options.outfile + '.csv')):
        with open(os.path.join(resultdir, options.outfile + '.csv'), 'w+') as result_file:
            result_file.write(sol.get_out_header()+'\n')

    random.seed(options.seed)
    solver = avs.get_solver_class(options.solver)()
    if options.training_episodes > 0:
        solver.train(options)
    blsolver = AStar()

    # Keeps track of useful statistics
    problem_count = min(len(open(os.path.join(problemdir, options.probfile + '.txt')).readlines()), options.training_episodes)
    stats = Plotting.EpisodeStats(
        solution_cost=np.zeros(problem_count*options.episodes),
        expanded=np.zeros(problem_count*options.episodes),
        generated=np.zeros(problem_count*options.episodes))
    blstats = Plotting.EpisodeStats(
        solution_cost=np.zeros(problem_count * options.episodes),
        expanded=np.zeros(problem_count * options.episodes),
        generated=np.zeros(problem_count * options.episodes))
    rstats = Plotting.EpisodeStats(
        solution_cost=np.zeros(problem_count * options.episodes),
        expanded=np.zeros(problem_count * options.episodes),
        generated=np.zeros(problem_count * options.episodes))

    with open(os.path.join(problemdir, options.probfile + '.txt'), 'r') as problem_file:
        with open(os.path.join(resultdir, options.outfile + '.csv'), 'a+') as result_file:
            line = problem_file.readline()
            p = prob()
            l = 0
            while line and l < options.training_episodes:
                p.read_in(line)
                print("Solving problem #{}: {}".format(l,p))
                solver.__init__(p,options)
                for e in (range(options.episodes)):
                    print("Running episode {}".format(e+1))
                    solver.solve(p)
                    result_file.write(str(e) + ',' + solver.get_stat(p) + '\n')

                    # Update statistics
                    stats.solution_cost[l] = solver.statistics[Statistics.Solution.value]
                    stats.expanded[l] = solver.statistics[Statistics.Expanded.value]
                    stats.generated[l] = solver.statistics[Statistics.Generated.value]
                    print("cost={}, expanded={}".format(stats.solution_cost[l], stats.expanded[l]))
                    # Update baseline
                    run_baseline(blsolver, p, options, blstats, l)
                    update_ratio(stats,blstats,l,rstats)
                    l = l + 1

                line = problem_file.readline()
    Plotting.plot_stats(stats, blstats, rstats, smoothing_window = options.smoothing_window)


