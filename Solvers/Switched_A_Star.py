from Solvers.Abstract_Solver import AbstractSolver, Statistics
from sortedcontainers import SortedDict
from Utils import MappedQueue
from Domains.Abstract_State import AbstractState
import random
from Solvers.A_Star import AStar
import numpy as np

class SwitchedAStar(AStar):
    """
    A Star Search which takes another heuristic and quickly checks if its accurate and returs
    """
    noise_decay = 0.97

    def __init__(self,problem=None,options=None, return_expanded = False):
        super(SwitchedAStar,self).__init__(problem, options, return_expanded)

    def solve(self,problem, h_theta, expansion_bound = None):
        open = MappedQueue()
        closed = set()
        start = problem.start
        goal = problem.goal

        expanded = []
        self.trust_radius = float("inf")
        self.statistics[Statistics.Expanded.value] = 0

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
                self.statistics[Statistics.TrustRadius.value] = 0

                if self.return_expanded:
                    return goal.get_path(), expanded
                else:
                    return goal.get_path()

            self.statistics[Statistics.Expanded.value] += 1
            if expansion_bound is not None and self.statistics[Statistics.Expanded.value] > expansion_bound:
                return False, expanded

            sub_path = self.quick_search(current, h_theta, goal, self.trust_radius)
            if sub_path is not None:
                path = current.get_path() + sub_path
                self.statistics[Statistics.Distance.value] = len(path)
                self.statistics[Statistics.Solution.value] = len(path)
                self.statistics[Statistics.Expanded.value] += len(sub_path)
                self.statistics[Statistics.Generated.value] += len(sub_path) #### Highly Incorrect!!!!!!!!!!
                self.statistics[Statistics.TrustRadius.value] = len(sub_path)
                print("r3", self.trust_radius)
                return path, expanded + path
                # return path and expanded


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

    def quick_search(self, state, h, goal, trust_radius):
        h_theta_val = h(state, goal)
        if h_theta_val > trust_radius:
            return None
        init_state_h_val = h_theta_val
        nodes = []
        for x in range(round(h_theta_val)):
            successors = state.get_successors()
            state = successors[np.argmin([h(s, goal) for s in successors])]
            nodes.append(state)

        if state == goal:
            return nodes
        else:
            self.trust_radius = min(self.trust_radius, h_theta_val)
            return None

    def __str__(self):
        if AbstractState.w == 0:
            return "Switched UCS"
        ans = ""
        if AbstractState.w != 1:
            ans += "{}h".format(AbstractState.w)
        return ans + "Switched A*"
