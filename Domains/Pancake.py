from Domains.Abstract_State import AbstractState
import Utils


class Pancake(AbstractState):

    def __init__(self, stack, g):
        self.stack = stack
        super(Pancake,self).__init__(g)

    # Return "[0,1,2]" from stack '0', '1', '2'
    def __str__(self):
        ans="["
        for n in self.stack:
            ans += str(n) + ","
        ans = ans[:-1] + "]"
        return ans

    def get_successors(self):
        successors = []

        for i in range(0, len(self.stack) - 1):
            child = Pancake(Utils.flip_tail(self.stack,i), self.g + 1)
            successors.append(child)

        return successors

    def __eq__(self, other):
        return str(self.stack) == str(other.stack)

    def __hash__(self):
        return hash(str(self.stack))

    # Gap heuristic
    def get_h(self, goal):
        # Currently Goal is assumed to be [0,1,2,3,4,...]
        cost = 0
        for i in range(len(self.stack) - 1):
            if abs(self.stack[i + 1] - self.stack[i]) > 1:
                cost += 1
        if self.stack[-1] != len(self.stack) - 1:
            cost += 1
        return cost

    def as_list(self):
        return self.stack

    @staticmethod
    def parse_state(string):
        stack = Utils.parse_list(string)
        return Pancake(stack,0)

    @staticmethod
    def get_goal(size):
        stack = [i for i in range(size)]
        return Pancake(stack,0)

    @staticmethod
    def get_name():
        return "pancake"
