from Domains.Abstract_State import AbstractState
import Utils
import numpy as np

class TSP(AbstractState):

    def __init__(self, graph, start_node, g, visited = None):

        self.num_nodes = len(graph)
        self.graph = graph
        #self.graph = np.random.randint(low=1, high=50, size=(self.num_nodes, self.num_nodes))
        #np.diagonal_fill(self.graph, 0)
        if visited:
            self.visited = visited
        else:
            self.visited = [False for x in range(self.num_nodes)]
        self.destination = start_node       # As start and end are same

        super(TSP, self).__init__(g)

    def __str__(self):
        """
        Print the array as it as
        TODO: write a better way to print graph
        """
        ans="["
        for x in self.graph:
            for y in x:
                ans += str(y) + ","
        for x in self.visited:
            if x:
                ans +=  '-1' + ','
            else:
                ans +=  '-2' + ','
        ans += str(self.destination) + "]"

        return ans

    def get_successors(self):
        #TODO
        pass

    def __eq__(self, other):
        return self.graph == other.graph and self.destination == other.destination and self.visited == other.visited #basically everything is equal

    def is_solution(self):
        return False if False in self.visited else True

    @staticmethod
    def parse_state(string):
        list_conversion = Utils.parse_list(string)
        start_node = list_conversion[-1]

        idx = -2
        while True:
            if list_conversion[idx] >= 0:
                break
            idx -= 1
        #print(list_conversion[idx], idx)
        idx += 1
        visited = [True if list_conversion[x] == -1 else False for x in range(idx, -1)]
        #print(len(list_conversion) + idx, list_conversion[len(list_conversion) + idx])
        num_nodes = int(np.sqrt(len(list_conversion) + idx))
        graph = np.array(list_conversion[:idx]).reshape(num_nodes, num_nodes) / 10
        return TSP(graph, start_node, 0, visited)


    @staticmethod
    def get_goal_dummy(size):
        graph = np.random.randint(low=1, high=50, size=(size, size))
        np.fill_diagonal(graph, 0)
        ans="["
        destination = np.random.randint(0, len(graph) - 1)
        for x in graph:
            for y in x:
                ans += str(y) + ","
        for x in range(size):
            ans +=  '-1' + ','
        ans += str(destination) + "]"

        return TSP.parse_state(ans)

    @staticmethod
    def get_random_problem_state(size):
        graph = np.random.randint(low=1, high=50, size=(size, size))
        np.fill_diagonal(graph, 0)
        destination = np.random.randint(0, len(graph) - 1)
        ans="["
        for x in graph:
            for y in x:
                ans += str(y) + ","
        for x in range(size):
            ans +=  '-2' + ','
        ans += str(destination) + "]"

        return TSP.parse_state(ans)


    def get_h(self, goal):
        # TODO
        return 0

    def __hash__(self):
        return hash(str(self))

    def as_tensor(self):
        array = self.graph.reshape(-1)
        other_e = [0 if x else 1 for x in self.visited] + [self.destination]
        return np.array(list(array) + other_e)

    @staticmethod
    def get_name():
        return "tsp"

    def as_list():
        raise NotImplementedError
