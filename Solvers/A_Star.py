from Solvers.Abstract_Solver import AbstractSolver, Statistics
from sortedcontainers import SortedDict
from Utils import MappedQueue
from Domains.Abstract_State import AbstractState
import random


class AStar(AbstractSolver):
    noise_decay = 0.97

    def __init__(self,problem=None,options=None, return_expanded = False):
        super(AStar,self).__init__()
        self.return_expanded = return_expanded
        try:
            self.h_func
        except:
            self.h_func = None
        if options:
            try:
                AbstractState.w = float(options.weight)
            except ValueError:
                raise Exception('Weight must be a valid number')
            except:
                raise Exception('must specify weight for WA* (w=1 for A* and w=0 for UCS)')
            assert AbstractState.w >= 0, "weight must be non negative"
            try:
                self.noise_std = float(options.noise)
            except ValueError:
                raise Exception('noise must be a valid number')
            except:
                raise Exception('must specify noise for WA* (n=0 for deterministic A*)')
            assert self.noise_std >= 0 and self.noise_std <= 0.5, "noise std must be between 0 and 0.5"

    def solve(self,problem):
        open = MappedQueue()
        closed = set()
        start = problem.start
        goal = problem.goal

        expanded = []

        self.set_h(start, goal)

        open.push(start)
        best_cost = float('inf')

        while len(open) > 0:
            current = open.pop()
            if self.return_expanded:
                expanded.append(current)
            pr = current.get_f()
            if pr >= best_cost:
                self.statistics[Statistics.Distance.value] = best_cost
                self.statistics[Statistics.Solution.value] = best_cost

                if self.return_expanded:
                    return goal.get_path(), expanded
                else:
                    return goal.get_path()

            self.statistics[Statistics.Expanded.value] += 1
            successors = current.get_successors()

            if successors:
                for s in successors:

                    if s in closed:
                        continue

                    self.set_h(s, goal)

                    prs = s.get_f()
                    if s in open:
                        if open[s].get_f() <= prs:
                            continue
                        else:
                            open.remove(s)

                    if s == problem.goal and s.g < best_cost:
                        best_cost = s.g
                        goal = s

                    s.parent = current
                    self.statistics[Statistics.Generated.value] += 1
                    open.push(s)

            closed.add(current)

        self.statistics[Statistics.Distance.value] = -1
        self.statistics[Statistics.Solution.value] = -1
        return False

    def set_h(self,state,goal):
        if self.h_func:
            h = self.h_func(state,goal)
        else:
            h = state.get_h(goal)
        if self.noise_std > 0:
            h = h * random.gauss(1, self.noise_std)
        state.set_h(h)

    def __str__(self):
        if AbstractState.w == 0:
            return "UCS"
        ans = ""
        if AbstractState.w != 1:
            ans += "{}h".format(AbstractState.w)
        return ans + "A*"
