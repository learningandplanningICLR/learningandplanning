import random

import numpy as np
import gin.tf
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional
from learning_and_planning.mcts.tree import TreeNode, GraphNode

class Planner(metaclass=ABCMeta):

    @abstractmethod
    def run_one_episode(self):
        pass


@gin.configurable
class MCTSBase(Planner):

    def __init__(self, num_mcts_passes=10):
        self._num_mcts_passes = num_mcts_passes

    @abstractmethod
    def tree_traversal(self, root) -> Tuple[TreeNode, List[Tuple[TreeNode, Optional[int]]]]: pass

    @abstractmethod
    def expand_leaf(self, leaf: TreeNode) -> float: pass

    @abstractmethod
    def initialize_root(self) -> GraphNode: pass

    @abstractmethod
    def _backpropagate(self, search_path: List[Tuple[TreeNode, Optional[int]]], value: float) -> None: pass

    @abstractmethod
    def _select_next_node(self, root) -> Tuple[TreeNode, Optional[int]]: pass

    def run_mcts_pass(self, root: TreeNode) -> None:
        # search_path = list of tuples (node, action)
        # leaf does not belong to search_path (important for not double counting its value)
        leaf, search_path = self.tree_traversal(root)
        value = self.expand_leaf(leaf)
        self._backpropagate(search_path, value)

    def preprocess(self, root: TreeNode) -> TreeNode: pass

    def run_one_step(self, root: TreeNode) -> Tuple[TreeNode, TreeNode, Optional[int]]:

        # if root is None (start of new episode) or Terminal (end of episode), initialize new root
        root = self.preprocess(root)

        # perform MCTS passes. each pass = tree traversal + leaf evaluation + backprop
        for _ in range(self._num_mcts_passes):
            self.run_mcts_pass(root)
        next, action = self._select_next_node(root)  # INFO: possible sampling for exploration

        return root, next, action

    @abstractmethod
    def run_one_episode(self):
        pass


@gin.configurable
class MCTS(MCTSBase):

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

    # here UNLIKE alphazero, we choose final action from the root according to value
    def _select_next_node(self, root):
        # INFO: below line guarantees that we do not perform one-step loop (may be considered slight hack)
        states_to_avoid = {root.state} if self._avoid_loops else set()
        values_and_actions = self._rate_children(root, states_to_avoid, final_rating=True)
        if not values_and_actions:
            # when there are no children (e.g. at the bottom states of ChainEnv)
            return None, None
        # TODO: can we do below more elegantly
        if len(self.history) < self._num_sampling_moves:
            chooser = _softmax_sample
        else:
            chooser = max
        (_, action) = chooser(values_and_actions)
        return root.children[action], action

    # Select the child with the highest score
    def _select_child(self, node, states_to_avoid):
        values_and_actions = self._rate_children(node, states_to_avoid)
        if not values_and_actions:
            return None, None
        (max_value, _) = max(values_and_actions)
        argmax = [
            action for value, action in values_and_actions if value == max_value
        ]
        # INFO: here can be sampling
        if len(argmax) > 1:  # PM: This works faster
            action = np.random.choice(argmax)
        else:
            action = argmax[0]
        return node.children[action], action

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
                nodes = [elem[0] for elem in history]
                # if game_steps < self.episode_max_steps:  # hence new_roots[idx].terminal == True
                #     history.append((new_roots, -1))  # INFO: for Terminal state 'action' = -1
                history, evaluator_kwargs = self.history_process_fn(history, game_solved)
                # give each state of the trajectory a value
                values = game_evaluator_new(history, self._node_value_mode, self._gamma, game_solved, **evaluator_kwargs)
                game = [(node.state, value, action) for (node, action, _), value in zip(history, values)]
                game = [(state.get_np_array_version(), value, action) for state, value, action in game]

                # self._curriculum.apply(game_solved)
                return game, game_solved, dict(nodes=nodes, graph_size=len(self._state2node))


def calculate_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards)+1)
    for i in np.arange(len(rewards), 0, -1):
        discounted_rewards[i-1] = gamma*discounted_rewards[i] + rewards[i-1]
    return discounted_rewards[:-1]


def game_evaluator_new(game, mode, gamma, solved, **kwargs):
    if mode == "bootstrap":
        return [node.value_acc.target() for node, _, _ in game]
    if "factual" in mode:
        rewards = np.zeros(len(game))
        if mode == "factual":  # PM: Possibly remove, kept for backward compatibility
            rewards[-1] = int(solved)
            gamma = 1
        elif mode == "factual_discount":
            rewards[-1] = int(solved)
        elif mode == "factual_hindsight":
            rewards[-1] = kwargs['hindsight_solved']
        elif mode == "factual_rewards":
            rewards = [reward for node, action, reward in game]
        else:
            raise NotImplementedError("not known mode", mode)

        return calculate_discounted_rewards(rewards, gamma)

    raise NotImplementedError("not known mode", mode)


@gin.configurable
class KC_MCTS(MCTS):
    def __init__(self, num_ensembles_per_game=None, ensemble_size=None, **kwargs):
        """
        Args:
            num_ensembles_per_game: number of ensemble members to use in single
                episode for index calculation. (this is passed to
                ValueAccumulator and does NOT changes dimension of values moved
                around inside this instance during run_one_episode)
            ensemble_size: size of ensemble, need to be set only if
                num_ensembles_per_game is not None.

        """
        self.num_ensembles_per_game = num_ensembles_per_game
        self.ensemble_size = ensemble_size
        # indices of ensemble members to use
        self._current_episode_ensemble_indices = None
        super(KC_MCTS, self).__init__(**kwargs)

    def run_one_episode(self):
        # Sample current episode ensemble members and run one episode.
        if self.num_ensembles_per_game is not None:
            self._current_episode_ensemble_indices = list(
                np.random.choice(
                    self.ensemble_size, self.num_ensembles_per_game,
                    replace=False
                )
            )
        return super(KC_MCTS, self).run_one_episode()

    def _initialize_graph_node(self, initial_value, state, done, solved):
        # Same as parent method, but set ensembles to use in value accumulator.
        new_node = super(KC_MCTS, self)._initialize_graph_node(
            initial_value, state, done, solved=solved
        )
        if self.num_ensembles_per_game is not None:
            new_node.value_acc.set_index_indices(
                index_indices=self._current_episode_ensemble_indices
            )
        return new_node

    def _select_next_node(self, root):
        # INFO: below line guarantees that we do not perform one-step loop (may be considered slight hack)
        states_to_avoid = {root.state} if self._avoid_loops else set()
        values_and_actions = self._rate_children(root, states_to_avoid, final_rating=True)
        # TODO: can we do below more elegantly
        if len(self.history) < self._num_sampling_moves:
            chooser = _softmax_sample
        else:
            chooser = max
        (_, action) = chooser(values_and_actions)
        return root.children[action], action


# def game_evaluator(game, mode, gamma, solved, **kwargs):
#
#     def get_value(node, distance):
#         if mode == "bootstrap":
#             value = node.value_acc.target()
#         elif mode in ("factual", "factual_discount"):
#             value = int(solved)
#             if mode == "factual_discount":
#                 value *= gamma ** distance
#         elif mode == "factual_hindsight": # That is not the best style, but we keep backward compatibility
#             value = kwargs['hindsight_solved']
#             value *= gamma ** distance
#         return value
#
#     values = []
#     for ((node, _), distance) in zip(game, reversed(range(len(game)))):
#         values.append(get_value(node, distance))
#     return values


def _softmax_sample(values_and_actions):
    # INFO: below for numerical stability,
    # see https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    max_values = max([v for v, _ in values_and_actions])
    sharpening_coeff = 2.  # TODO: make it a parameter
    prob = np.array([np.exp(sharpening_coeff * v - max_values) for v, _ in values_and_actions])
    total = np.sum(prob)
    prob /= total
    idx = np.random.choice(range(len(values_and_actions)), p=prob)
    return values_and_actions[idx]


def td_backup(node, action, value, gamma):
    if action is None:
        return value
    return node.rewards[action] + gamma * value
