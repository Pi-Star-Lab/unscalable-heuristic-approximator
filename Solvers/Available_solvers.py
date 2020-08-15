from Solvers.A_Star import AStar
from Solvers.DLRTA import DLRTA
from Solvers.LRTA import LRTA
from Solvers.DLMC import DLMC

solvers = ['a', 'lrta', 'dlrta', 'dlmc']

def get_solver_class(name):
    if name == solvers[0]:
        return AStar
    elif name == solvers[1]:
        return LRTA
    elif name == solvers[2]:
        return DLRTA
    elif name == solvers[3]:
        return DLMC
    else:
        assert False, "unknown solver name as input. solver must be from " + str(solvers)
