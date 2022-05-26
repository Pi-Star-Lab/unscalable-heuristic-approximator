from Solvers.Abstract_Solver import AbstractSolver, Statistics
from python_tsp.exact import solve_tsp_dynamic_programming


class AssumptionViolationException(Exception):
    pass

class TSPSolver(AbstractSolver):

    def __init__(self,problem=None):
        super(TSPSolver,self).__init__()


    def solve(self, problem):
        """
        Important: Takes graph with NO VISITED elements
        """
        if True in problem.start.visited:
            print(problem.start.visited)
            raise AssumptionViolationException("TSP constraint violated")
        path, distance = solve_tsp_dynamic_programming(problem.start.graph)

        problem.visited = [True for x in range(problem.start.num_nodes)]

        #TODO: fix hack
        return [x for x in range(distance)]

    def set_h(self, start, goal):
        state.set_h(lambda x: 0)

    def __str__(self):
        return "TSP Solver"

