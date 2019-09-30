from collections import Counter

import numpy as np
import gin.tf

from learning_and_planning.mcts.mcts_planner import MCTSBase, game_evaluator_new, _softmax_sample, td_backup
from learning_and_planning.mcts.tree import TreeNode, GraphNode


@gin.configurable
class MCTSValueWithVoting(MCTSBase):

    def __init__(self,
                 model,
                 value,
                 episode_max_steps,
                 node_value_mode,
                 gamma=0.99,
                 value_annealing=1.,
                 num_sampling_moves=0,
                 num_mcts_passes=10,
                 avoid_loops=True,
                 avoid_traversal_loop_coeff=0.0,
                 avoid_history_coeff=0.0,
                 history_process_fn = lambda x, solved: (x, {}),
                 differ_final_rating=False
                 ):
        super().__init__(num_mcts_passes=num_mcts_passes)
        self._value = value  # generalized value, e.g. could be ensemble
        self._gamma = gamma
        self._value_annealing = value_annealing
        self._num_sampling_moves = num_sampling_moves
        self._model = model
        self._avoid_loops = avoid_loops
        self._state2node = {}
        self.history = []
        self.avoid_traversal_loop_coeff = avoid_traversal_loop_coeff
        if callable(avoid_history_coeff):
            self.avoid_history_coeff = avoid_history_coeff
        else:
            self.avoid_history_coeff = lambda: avoid_history_coeff
        self.episode_max_steps = episode_max_steps
        self._node_value_mode = node_value_mode
        self.history_process_fn = history_process_fn
        self.differ_final_rating = differ_final_rating
        assert value_annealing == 1., "Annealing temporarily not supported."  # TODO(pm): reenable

    def tree_traversal(self, root):
        node = root
        seen_states = set()
        search_path = []
        while node.expanded():
            seen_states.add(node.state)
            node.value_acc.add_auxiliary(self.avoid_traversal_loop_coeff)
            #  Avoiding visited states in the fashion of https://openreview.net/pdf?id=Hyfn2jCcKm

            # INFO: if node Dead End, (new_node, action) = (None, None)
            # INFO: _select_child can SAMPLE an action (to break tie)
            states_to_avoid = seen_states if self._avoid_loops else set()
            new_node, action = self._select_child(node, states_to_avoid)  #
            search_path.append((node, action))
            node = new_node
            if new_node is None:  # new_node is None iff node has no unseen children, i.e. it is Dead End
                break
        # at this point node represents a leaf in the tree (and is None for Dead End).
        # node does not belong to search_path.
        return node, search_path

    def _backpropagate(self, search_path, value):
        # Note that a pair
        # (node, action) can have the following form:
        # (Terminal node, None),
        # (Dead End node, None),
        # (TreeNode, action)
        for node, action in reversed(search_path):
            value = td_backup(node, action, value, self._gamma)  # returns value if action is None
            node.value_acc.add(value)
            node.value_acc.add_auxiliary(-self.avoid_traversal_loop_coeff)

    def _get_value(self, obs, states):
        value = self._value(obs=obs, states=states)
        return self._value_annealing * value
        # return self._value_annealing * value

    def _initialize_graph_node(self, initial_value, state, done, solved):
        value_acc = self._value.create_accumulator(initial_value, state)
        new_node = GraphNode(value_acc,
                             state=state,
                             terminal=done,
                             solved=solved,
                             nedges=self._model.num_actions)
        self._state2node[state] = new_node  # store newly initialized node in _state2node
        return new_node

    def preprocess(self, root):
        if root is not None and not root.terminal:
            root.value_acc.add_auxiliary(self.avoid_history_coeff())
            return root

        # 'reset' mcts internal variables: _state2node and _model
        self._state2node = {}
        obs = self._model.reset()
        state = self._model.state()
        value = self._get_value([obs], [state])[0]
        new_node = self._initialize_graph_node(initial_value=value, state=state, done=False, solved=False)
        new_root = TreeNode(new_node)
        new_root.value_acc.add_auxiliary(self.avoid_history_coeff())
        return new_root

    def initialize_root(self):
        raise NotImplementedError("should not happen")
        # 'reset' mcts internal variables: _state2node and _model
        self._state2node = {}
        obs = self._model.reset()
        state = self._model.state()
        value = self._get_value([obs], [state])[0]

        new_node = self._initialize_graph_node(initial_value=value, state=state, done=False, solved=False)
        return TreeNode(new_node)

    def expand_leaf(self, leaf: TreeNode):
        if leaf is None:  # Dead End
            return self._value.traits.dead_end

        if leaf.terminal:  # Terminal state
            return self._value.traits.zero

        # neighbours are ordered in the order of actions: 0, 1, ..., _model.num_actions
        obs, rewards, dones, solved, states = self._model.neighbours(leaf.state)

        value_batch = self._get_value(obs=obs, states=states)

        for idx, action in enumerate(self._model.legal_actions()):
            leaf.rewards[idx] = rewards[idx]
            new_node = self._state2node.get(states[idx], None)
            if new_node is None:
                child_value = value_batch[idx] if not dones[idx] else self._value.traits.zero
                new_node = self._initialize_graph_node(
                    child_value, states[idx], dones[idx], solved=solved[idx]
                )
            leaf.children[action] = TreeNode(new_node)

        return leaf.value_acc.get()


    def _child_index(self, parent, action, final_index=False):
        accumulator = parent.children[action].value_acc
        if final_index:
            value = accumulator.final_index(parent.value_acc, action)
        else:
            value = accumulator.index(parent.value_acc, action)
        return td_backup(parent, action, value, self._gamma)

    def _rate_children(self, node, states_to_avoid, final_rating=False):
        final_index = final_rating and self.differ_final_rating
        assert self._avoid_loops or len(states_to_avoid) == 0, "Should not happen. There is a bug."
        return [
            (self._child_index(node, action, final_index=final_index), action)
            for action, child in node.children.items()
            if child.state not in states_to_avoid
        ]

    # Select the child with the highest score
    def _select_child(self, node, states_to_avoid):
        values_and_actions = self._rate_children(node, states_to_avoid, final_rating=False)  # [(ensemble, action)]
        if not values_and_actions:
            return None, None
        # INFO: if a given ensemble has multiple choices with the same value, we take first. should not happen often!
        action = self._majority_vote(values_and_actions)
        return node.children[action], action

    # here UNLIKE alphazero, we choose final action from the root according to value
    def _select_next_node(self, root):
        # INFO: below line guarantees that we do not perform one-step loop (may be considered slight hack)
        states_to_avoid = {root.state} if self._avoid_loops else set()
        values_and_actions = self._rate_children(root, states_to_avoid, final_rating=False)
        if not values_and_actions:
            # when there are no children (e.g. at the bottom states of ChainEnv)
            return None, None
        ## TODO: can we do below more elegantly
        #if len(self.history) < self._num_sampling_moves:
        #    chooser = _softmax_sample
        #else:
        #    chooser = max
        ## todo: HERE
        #(_, action) = chooser(values_and_actions)
        action = self._majority_vote(values_and_actions)
        return root.children[action], action

    def _majority_vote(self, values_and_actions):
        max_values = np.max(list(zip(*values_and_actions))[0], axis=0)  # max_values[i] = max value for ith ensemble
        #argmax = np.argmax([vaa[0] for vaa in values_and_actions], axis=0)
        argmax = [
            action for idx, mv in enumerate(max_values) for value, action in values_and_actions if np.isclose(value[idx], mv)
        ]
        return majority_vote(argmax)  # , num_actions=self._model.num_actions)

    def run_one_episode(self):
        new_root = None
        history = []
        game_steps = 0

        while True:
            old_root, new_root, action = self.run_one_step(new_root)

            history.append((old_root, action, old_root.rewards[action]))
            game_steps += 1

            # action required if the end of the game (terminal or step limit reached)
            if new_root.terminal or game_steps >= self.episode_max_steps:

                game_solved = new_root.solved
                # if game_steps < self.episode_max_steps:  # hence new_roots[idx].terminal == True
                #     history.append((new_roots, -1))  # INFO: for Terminal state 'action' = -1
                history, evaluator_kwargs = self.history_process_fn(history, game_solved)
                # give each state of the trajectory a value
                values = game_evaluator_new(history, self._node_value_mode, self._gamma, game_solved, **evaluator_kwargs)
                game = [(node.state, value, action) for (node, action, _), value in zip(history, values)]
                game = [(state.get_np_array_version(), value, action) for state, value, action in game]

                # self._curriculum.apply(game_solved)
                nodes = [elem[0] for elem in history]
                return game, game_solved, dict(nodes=nodes, graph_size=len(self._state2node))

# majority vote for general, mulitple repetitions (quite fast)
def majority_vote(argmax, num_actions=4):
    frequencies = [0] * num_actions
    for a in argmax:
        frequencies[a] += 1
    max_frequency = frequencies[a]
    for a in argmax[:-1]:
        if frequencies[a] > max_frequency:
            max_frequency = frequencies[a]
    #for frequency in frequencies:
    #    if frequency > max_frequency:
    #        max_frequency = frequency
    argmax = [action for action in range(num_actions) if frequencies[action] == max_frequency]
    if len(argmax) > 1:
        action = np.random.choice(argmax)
    else:
        action = argmax[0]
    return action

#def majority_vote(argmax):
#    frequencies = Counter(argmax)  # frequency = [(action, frequency)]
#    max_frequency = frequencies.most_common(1)[0][1]  # max frequency
#    argmax = [action for action, frequency in frequencies.items() if frequency == max_frequency]
#    if len(argmax) > 1:
#        action = np.random.choice(argmax)
#    else:
#        action = argmax[0]
#    return action


## optimized for num_actions=4
#def majority_vote(argmax, num_actions=4):
#    frequencies = [0] * num_actions
#    len_argmax = len(argmax)
##    # optimization for num_actions=4
##    if len_argmax == 1:
##        return argmax[0]
##    elif len_argmax == 2:
##        if argmax[0] == argmax[1]:
##            return argmax[0]
##        else:
##            return np.random.choice(argmax)
##    elif len_argmax == 3:
##        if argmax[0] == argmax[1] or argmax[0] == argmax[2]:
##            return argmax[0]
##        elif argmax[1] == argmax[2]:
##            return argmax[1]
##        else:
##            return np.random.choice(argmax)
##    elif len_argmax == 4:
##        # there exists a majority vote (4 cases)
##        if argmax[0] == argmax[1] and argmax[0] == argmax[2]:
##            return argmax[0]
##        elif argmax[0] == argmax[1] and argmax[0] == argmax[3]:
##            return argmax[0]
##        elif argmax[0] == argmax[2] and argmax[0] == argmax[3]:
##            return argmax[0]
##        elif argmax[1] == argmax[2] and argmax[1] == argmax[3]:
##            return argmax[1]
##        # there is a two and 2x ones (6 cases)
##        elif argmax[0] == argmax[1] and argmax[2] != argmax[3]:
##            return argmax[0]
##        elif argmax[0] == argmax[2] and argmax[1] != argmax[3]:
##            return argmax[0]
##        elif argmax[0] == argmax[3] and argmax[1] != argmax[2]:
##            return argmax[0]
##        elif argmax[1] == argmax[2] and argmax[0] != argmax[3]:
##            return argmax[1]
##        elif argmax[1] == argmax[3] and argmax[0] != argmax[2]:
##            return argmax[1]
##        elif argmax[2] == argmax[3] and argmax[0] != argmax[1]:
##            return argmax[2]
##        # there are 2 twos (3 cases)
##        elif argmax[0] == argmax[1] and argmax[2] == argmax[3]:
##            return np.random.choice([argmax[0], argmax[2]])
##        elif argmax[0] == argmax[2] and argmax[1] == argmax[3]:
##            return np.random.choice([argmax[0], argmax[1]])
##        elif argmax[0] == argmax[3] and argmax[1] == argmax[2]:
##            return np.random.choice([argmax[0], argmax[1]])
##        # each different (1 case)
##        else:
##            return np.random.choice(argmax)
#    majority = len_argmax // 2
#    for a in argmax:
#        frequencies[a] += 1
#        if frequencies[a] > majority:  # unique majority
#            return a
#    max_frequency = frequencies[a]
#    for frequency in frequencies:
#        if frequency > max_frequency:
#            max_frequency = frequency
#    argmax = [action for action in range(num_actions) if frequencies[action] == max_frequency]
#    if len(argmax) > 1:
#        action = np.random.choice(argmax)
#    else:
#        action = argmax[0]
#    return action
