import optparse
import sys
import random
import os
from Domains.Problem_instance import ProblemInstance as prob
from Domains.TSP import TSP
from Solvers.Abstract_Solver import Statistics
from Solvers.A_Star import AStar
from Solvers.TSP_Solver import TSPSolver
outdir = "dataset/tsp/"

def print_path(path):
    if not path:
        print([])
        return
    for s in path[1:]:
        print(s)
# register arguments and set default values
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


if __name__ == '__main__':
    options = readCommand(sys.argv)
    assert options.outfile and options.domain and options.max_size and options.min_size, "arguments must include: outfile, domain, " \
                                                                "and problem size"
    random.seed(options.seed)
    cls = prob.get_domain_class(options.domain)
    solver = AStar()
    if cls == TSP:
        solver = TSPSolver()
    max_steps = 1000
    for i in range(options.min_size, options.max_size + 1):
        with open(os.path.join(outdir, options.outfile + '_{}.txt'.format(i)), 'w+') as out:
            samples = 0
            while samples < options.max_steps:
                p = prob()

                p.random_walk(cls.get_goal_dummy(i),cls, max_steps)
                sol_len = len(solver.solve(p))
                # do a search, if equals? print samples += 1
                print(p, i, sol_len - 1)
                if sol_len - 1 == i:
                    pass # solution found
                    #print(p)
                samples += 1
                out.write(str(p)+str(sol_len - 1) + '\n')

