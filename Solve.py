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
import pickle
import torch

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
    parser.add_option("-l", "--layers", dest="layers", type="string",
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

    parser.add_option("-m", "--solver_dump_path", type="str", dest="save_path",
            default=".", help='Path to save checkpoints')
    parser.add_option("-u", "--update_target", type="int", dest="update_target", default=100,
                      help='No. of episodes after which you should sync target with model')
    parser.add_option("-f", "--expansion_bound", type="int", dest="expansion_bound", default=500,
                      help='Maximum number of expansions that should take place')
    parser.add_option("--resume", type="int", dest="resume", default=None,
                      help='Which episode to resume from?')
    parser.add_option("--path", type="int", dest="resume", default=None,
                      help='Path of Solver from where to resume?')
    (options, args) = parser.parse_args(argv)
    return options


def run_baseline(blsolver, problem, options, blstats, idx):
    blsolver.__init__(problem, options)
    blsolver.solve(problem)
    blstats.solution_cost[idx] = blsolver.statistics[Statistics.Solution.value]
    blstats.expanded[idx] = blsolver.statistics[Statistics.Expanded.value]
    blstats.generated[idx] = blsolver.statistics[Statistics.Generated.value]

def update_ratio(stats, blstats, idx, rstats):
    rstats.solution_cost[idx] = float(stats.solution_cost[idx])/blstats.solution_cost[idx]
    rstats.expanded[idx] = float(stats.expanded[idx])/blstats.expanded[idx]
    rstats.generated[idx] = float(stats.generated[idx])/blstats.generated[idx]

# -p tile4 -o res -s dlmc -w 10 -e 100 -r 0.1 -l [24,24] -t 100 -a tile -b 4
if __name__ == '__main__':
    options = readCommand(sys.argv)

    # Create result file if one doesn't exist
    if not os.path.exists(os.path.join(resultdir, options.outfile + '.csv')):
        with open(os.path.join(resultdir, options.outfile + '.csv'), 'w+') as result_file:
            result_file.write(sol.get_out_header()+'\n')

    random.seed(options.seed)
    os.environ['PYTHONHASHSEED']=str(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    solver = avs.get_solver_class(options.solver)()
    if options.training_episodes > 0:
        solver.train(options)
    blsolver = AStar()

    # Keeps track of useful statistics
    problem_count = min(len(open(os.path.join(problemdir, options.probfile + '.txt')).readlines()), options.training_episodes)
    stats = Plotting.EpisodeStats(
        solution_cost=np.zeros(problem_count*options.episodes),
        expanded=np.zeros(problem_count*options.episodes),
        weights=np.zeros(problem_count*options.episodes),
        generated=np.zeros(problem_count*options.episodes))
    blstats = Plotting.BaselineStats(
        solution_cost=np.zeros(problem_count * options.episodes),
        expanded=np.zeros(problem_count * options.episodes),
        generated=np.zeros(problem_count * options.episodes))
    rstats = Plotting.BaselineStats(
        solution_cost=np.zeros(problem_count * options.episodes),
        expanded=np.zeros(problem_count * options.episodes),
        generated=np.zeros(problem_count * options.episodes))

    skip_problems = 0
    if options.resume is not None:
        skip_problems = options.resume
    if options.path:
        solver = pickle.load(open(options.path, 'rb'))
    problem_no = 0

    with open(os.path.join(problemdir, options.probfile + '.txt'), 'r') as problem_file:
        with open(os.path.join(resultdir, options.outfile + '.csv'), 'a+') as result_file:
            line = problem_file.readline()
            p = prob()

            if options.path is None:
                solver.__init__(p,options)

            l = 0
            while line and l < options.training_episodes * options.episodes:
                p.read_in(line)
                print("Solving problem #{}: {}".format(l,p))
                line = problem_file.readline()
                if l < skip_problems * options.episodes:
                    l += options.episodes
                    continue
                for e in range(0, (options.episodes)):
                    print("Running episode {}".format(e+1))
                    solver.solve(p)
                    result_file.write(str(e) + ',' + solver.get_stat(p) + '\n')

                    # Update statistics
                    stats.solution_cost[l] = solver.statistics[Statistics.Solution.value]
                    stats.expanded[l] = solver.statistics[Statistics.Expanded.value]
                    stats.generated[l] = solver.statistics[Statistics.Generated.value]
                    stats.weights[l] = solver.statistics[Statistics.Weights.value]
                    print("cost={}, expanded={}, weight={}".format(stats.solution_cost[l], stats.expanded[l], stats.weights[l]))
                    # Update baseline
                    run_baseline(blsolver, p, options, blstats, l)
                    update_ratio(stats,blstats,l,rstats)
                    l = l + 1
            problem_no += 1
    Plotting.plot_stats(stats, blstats, rstats, smoothing_window = options.smoothing_window)

    #f = open("optimal_states.list", "wb")
    #pickle.dump(solver.optimal_states, f)

    #f = open("optimal_states.list", "rb")
    #opt_states = solver.optimal_states#pickle.load(f)
    #for state in opt_states:
    #    print(state.as_tensor(), solver.get_h(state, opt_states[-1]))
    #solver.save("Models_tiles/10k_4", "memory_10k_4")
    #print(solver.initial_state_hval)
    #print(solver.final_state_hval)
    #f.write(json.dumps(solver.optimal_states))
