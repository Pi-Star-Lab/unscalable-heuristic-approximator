from Solvers.Abstract_Solver import AbstractSolver, Statistics
import random

class LRTA(AbstractSolver):

    def __init__(self,problem=None,options=None):
        if problem:
            self.init_h(problem,options)
        self.epsilon_greedy = 0
        super(LRTA,self).__init__()

    def init_h(self,problem,options):
        self.h = {}

    def solve(self, problem):
        self.statistics = [0] * len(Statistics)
        current = problem.start
        while current != problem.goal:
            next_state, new_h = self.get_best_successor(current,problem)
            # print("Gap vs learned {} : {}".format(current.get_h(problem.goal), self.get_h(current,self.use_h,problem.goal)))
            # print("Start h-value {}".format(self.get_h(problem.start)))
            # print("Goal h-value {}".format())
            self.set_h(current,new_h,problem)
            print("Gap vs learned {} : {}".format(current.get_h(problem.goal),
                                                  self.get_h(current, self.use_h, problem.goal)))
            current = next_state
            if random.random() < self.epsilon_greedy:
                current = current.random_step()
            self.statistics[Statistics.Distance.value] += problem.edge_cost
            self.statistics[Statistics.Expanded.value] += problem.edge_cost


    def get_h(self,state,use_h,goal):
        #if in self.h then return self.h else if use_h then return h function value else return 0
        if state in self.h:
            return self.h[state]
        elif use_h:
          return state.get_h(goal)
        else:
            return 0

    def set_h(self,state,h,problem):
        self.h[state] = h

    def get_best_successor(self,current,problem):
        next_state = None
        hc = float("inf")
        successors = current.get_successors()
        for s in successors:
            hs = self.get_h(s, self.use_h, problem.goal)
            if hs + problem.edge_cost < hc:
                hc = hs + problem.edge_cost
                next_state = s
        if self.use_h:
            hc = max(current.get_h(problem.goal), hc)
        return next_state, hc

    def __str__(self):
        return "LRTA*"
