from Solvers.Abstract_Solver import AbstractSolver, Statistics
from sortedcontainers import SortedDict
from Utils import MappedQueue, MappedFIFOQueue
from Domains.Abstract_State import AbstractState
import random


class MultipleIDAStar(AbstractSolver):
    noise_decay = 0.97
    update_buffer_size = 30

    def __init__(self,problem=None,options=None, return_expanded = False):
        super(MultipleIDAStar,self).__init__()
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

    def solve(self,problem, expansion_bound = None):
        start = problem.start
        goal = problem.goal

        self.statistics[Statistics.Expanded.value] = 0

        self.set_h(start, goal)

        BOUND_MAX = 60000
        final_path = None
        to_update = {}

        for bound in range(int(start.get_f()), BOUND_MAX):
        #for bound in range(100, 101):

            open = MappedFIFOQueue()
            open.push(start)

            while len(open) > 0:

                if len(to_update) > MultipleIDAStar.update_buffer_size:
                    break

                current = open.pop()

                if current in to_update:
                    path = current.get_path()
                    path.reverse()
                    for i, x in enumerate(path):
                        if x in to_update:
                            to_update[x] = min(to_update[x], i + to_update[current] + 1)
                        else:
                            to_update[x] = i + to_update[current] + 1


                if current.is_solution(): #  current is goal? remove best_cost var

                    path = current.get_path()
                    for i, p in enumerate(path):
                        print(i, p, end=" ")
                    print(" ")
                    path.reverse()
                    if final_path is None:
                        self.statistics[Statistics.Distance.value] = len(path)
                        self.statistics[Statistics.Solution.value] = len(path)
                        final_path = path

                    for i, x in enumerate(path):
                        if x in to_update:
                            to_update[x] = min(to_update[x], i)
                        else:
                            to_update[x] = i

                self.statistics[Statistics.Expanded.value] += 1
                if expansion_bound is not None and self.statistics[Statistics.Expanded.value] > expansion_bound:
                    if final_path is None:
                        return False, []
                    else:
                        return final_path, to_update

                successors = current.get_successors()

                if successors:
                    for s in successors:

                        path = s.get_path()
                        if s in path[1:]:
                            continue

                        self.set_h(s, goal)

                        prs = s.get_f()
                        if prs > bound:
                            continue

                        s.parent = current
                        self.statistics[Statistics.Generated.value] += 1
                        open.push(s)


            if len(to_update) > MultipleIDAStar.update_buffer_size:
                break

        if final_path is None:
            self.statistics[Statistics.Distance.value] = -1
            self.statistics[Statistics.Solution.value] = -1
            return False, []
        else:
            return final_path, to_update

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
