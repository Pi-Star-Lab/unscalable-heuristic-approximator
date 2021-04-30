from Solvers.Abstract_Solver import AbstractSolver, Statistics
from sortedcontainers import SortedDict
from Utils import MappedQueue
from Domains.Abstract_State import AbstractState
import random


class MultipleAStar(AbstractSolver):
    noise_decay = 0.97
    update_buffer_size = 300
    def __init__(self,problem=None,options=None,return_expanded=False):
        super(MultipleAStar,self).__init__()
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

    def solve(self,problem, expansion_bound = None):
        open = MappedQueue()
        closed = set()
        start = problem.start
        goal = problem.goal

        self.statistics[Statistics.Expanded.value] = 0

        self.set_h(start, goal)

        open.push(start)
        best_cost = float('inf')

        final_path = None
        to_update = {}

        while len(open) > 0 and len(to_update) < MultipleAStar.update_buffer_size:
            current = open.pop()
            pr = current.get_f()
            if pr >= best_cost:

                path = goal.get_path()
                path.reverse()
                if final_path is None:
                    print("found final_path", best_cost)
                    self.statistics[Statistics.Distance.value] = best_cost
                    self.statistics[Statistics.Solution.value] = best_cost
                    final_path = path

                for i, x in enumerate(path):
                    if x in to_update:
                        to_update[x] = min(to_update[x], i)
                    else:
                        to_update[x] = i

                best_cost = float("inf")


            self.statistics[Statistics.Expanded.value] += 1
            if expansion_bound is not None and self.statistics[Statistics.Expanded.value] > expansion_bound:
                print("expansion bound reached")
                if final_path is None:
                    return False, []
                else:
                    return final_path, to_update

            successors = current.get_successors()

            if successors:
                for s in successors:
                    if s in to_update:
                        path = current.get_path()
                        path.reverse()
                        for i, x in enumerate(path):
                            if x in to_update:
                                to_update[x] = min(to_update[x], i + to_update[s] + 1)
                            else:
                                to_update[x] = i + to_update[s] + 1

                    if s in closed:
                        continue

                    self.set_h(s, goal)

                    prs = s.get_f()
                    if s in open:
                        if open[s].get_f() <= prs:
                            continue
                        else:
                            open.remove(s)

                    if s.is_solution() and s.g < best_cost:
                        best_cost = s.g
                        goal = s

                    s.parent = current
                    self.statistics[Statistics.Generated.value] += 1
                    open.push(s)

            closed.add(current)


        print("done with search")
        if final_path is None:
            self.statistics[Statistics.Distance.value] = -1
            self.statistics[Statistics.Solution.value] = -1
            return False, []
        else:
            return final_path, to_update

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
