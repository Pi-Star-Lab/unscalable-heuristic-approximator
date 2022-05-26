from Solvers.A_Star import AStar

solvers = ['a']

def get_solver_class(name):
    if name == solvers[0]:
        return AStar
    else:
        assert False, "unknown solver name as input. solver must be from " + str(solvers)
