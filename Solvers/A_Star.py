from Solvers.Abstract_Solver import AbstractSolver, Statistics
from sortedcontainers import SortedDict
from Utils import MappedQueue


class AStar(AbstractSolver):

    def __init__(self,problem=None,options=None):
        super(AStar,self).__init__(problem,options)

    def solve(self,problem):
        open = MappedQueue()
        closed = set()
        start = problem.start
        goal = problem.goal

        if self.use_h:
            start.set_h(goal)

        open.push(start)
        best_cost = float('inf')

        while len(open) > 0:
            current = open.pop()
            pr = current.get_f()
            if pr >= best_cost:
                self.statistics[Statistics.Distance.value] = best_cost
                self.statistics[Statistics.Solution.value] = best_cost
                return True

            self.statistics[Statistics.Expanded.value] += 1
            successors = current.get_successors()

            if successors:
                for s in successors:

                    if s in closed:
                        continue

                    if self.use_h:
                        s.set_h(goal)

                    prs = s.get_f()
                    if s in open:
                        if open[s].get_f() <= prs:
                            continue
                        else:
                            open.remove(s)

                    if s == problem.goal and s.g < best_cost:
                        best_cost = s.g

                    s.parent = current
                    self.statistics[Statistics.Generated.value] += 1
                    open.push(s)

            closed.add(current)

        self.statistics[Statistics.Distance.value] = -1
        self.statistics[Statistics.Solution.value] = -1
        return False

    def __str__(self):
        return "A*"
