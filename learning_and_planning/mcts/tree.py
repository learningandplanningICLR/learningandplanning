# graph node
class GraphNode(object):
    def __init__(self, value_acc,
                 state=None,
                 terminal=False,
                 solved=False,
                 nedges=4):
        self.value_acc = value_acc
        self.rewards = [None] * nedges
        self.state = state
        self.terminal = terminal
        self.solved = solved

# tree node
class TreeNode(object):
    def __init__(self, node: object) -> object:
        self.node = node
        self.children = {}  # {valid_action: Node}

    @property
    def rewards(self):
        return self.node.rewards

    @property
    def value_acc(self):
        return self.node.value_acc

    @property
    def state(self):
        return self.node.state

    @state.setter
    def state(self, state):
        self.node.state = state

    def expanded(self):
        return True if self.children else False

    @property
    def terminal(self):
        return self.node.terminal

    @property
    def solved(self):
        return self.node.solved

    @terminal.setter
    def terminal(self, terminal):
        self.node.terminal = terminal
