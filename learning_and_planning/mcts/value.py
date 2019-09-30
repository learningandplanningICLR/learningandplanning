from abc import ABC
from collections import deque
import time
import unittest
from itertools import product

import numpy as np
import tensorflow as tf
import os
import gin
import heapq
from collections import deque
from tensorflow.python.keras.models import load_model

from ourlib.rl_misc import linearly_decaying_epsilon
from gym_sokoban.envs import SokobanEnv
from learning_and_planning.mcts.replay_buffer import circular_replay_buffer_mcts

try:
    from mpi4py import MPI
    import baselines.common.tf_util as U
except ImportError:
    MPI = None


class Value(object):

    additional_info_cmaps = ()
    traits = None

    def create_accumulator(self, initial_value):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def moments(self, inputs):
        raise NotImplementedError

    # TODO: should the input be inputs or state, obs?
    def __call__(self, inputs):
        raise NotImplementedError

    def additional_info(self, inputs):
        """Additional info for visualization.

        Returns a tuple of arrays of shape (batch_size,).
        """
        return ()

    def _restore_vf_for_root(self, data):
        raise NotImplementedError

    def _dump_vf_for_root(self, root):
        raise NotImplementedError

    def decompress(self, data):
        """

      Args:
        data: tuples with (env_state_bytes, env_seed, value)
      Returns:
        tuples with (flat_array_with_env_state_and_seed, value)
      """
        data = [(np.frombuffer(n1, dtype=np.uint8), n2, v) for n1, n2, v in data]
        data = [(n1.astype(np.uint32), n2, v) for n1, n2, v in data]
        data = [(np.concatenate((n1, np.asarray((n2,)))), v) for n1, n2, v in
                data]
        return data

    def compress(self, data):
        """

      Deprecated. For aggregated states from single board it is better to use
      zlib.compress.

      Args:
        data: tuples with (flat_array_with_env_state_and_seed, value)
      Returns:
        tuples with (env_state_bytes, env_seed, value)
      """
        return [(np.array(n[:-1], dtype=np.uint8).tobytes(), n[-1], v)
                for n, v in data]

    def dump_vf_for_root(self, root, compressed=False):
        data = self._dump_vf_for_root(root)
        if compressed:
            return self.compress(data)
        else:
            return data

    def load_vf_for_root(self, data, compressed=False):
        if compressed:
            data = self.decompress(data)

        self._restore_vf_for_root(data)

        return data[0][0]

    def dump_vf_for_root_visual_debug(self, root, filename_prefix, render_env):
        node_vf_tuples = self.dump_vf_for_root(root)
        self.dump_vf_for_visual_debug(node_vf_tuples, filename_prefix,
                                      render_env)

    def dump_vf_for_visual_debug(self, node_vf_tuples, filename_prefix,
                                 render_env):
        from PIL import Image, ImageDraw
        for i, (node, vf) in enumerate(node_vf_tuples):
            render_env.restore_full_state(np.array(node))
            arr = render_env.render('rgb_array')
            im = Image.fromarray(arr)
            draw = ImageDraw.Draw(im)
            draw.text((0, 0), "{0:.1f}".format(vf))
            file_name = filename_prefix + "{:04d}.png".format(i)
            im.save(file_name)


class ValueDictionary(Value, ABC):
    def __init__(self):
        self._value = None
        self.root = None

    def __call__(self, states=None, obs=None):
        assert self._value, "self._value = {} (load_vf_from_root has not been called)"
        assert self.root.any(), "self.root = None (load_vf_from_root has not been called)"
        assert states is not None, "States cannot be None"
        if not isinstance(states, list):  # if called on states
            states = [states]
        seeds = [state[-1] for state in states]
        assert len(set(seeds)) == 1, "States do not come from the same level"
        assert seeds[0] == self.root[-1], "States do not come from the same level as root"
        return np.array([self._value.get(tuple(state), -np.inf) for state in states])

    def _restore_vf_for_root(self, data):
        self._value = {tuple(state): value for state, value in data}
        self.root = data[0][0]

    def _dump_vf_for_root(self, root):
        return [(state, value) for state, value in self._value.items()]


# Needs a slightly different init than ValueEnsemble
class ValuePerfect(Value, ABC):
    """Calculate Value for optimal policy.

    This only consist sub-graph of states achievable from root to solution
    states. It does not consist some children states of solution states.
    """

    def __init__(self, env, root=None):
        """

        Args:
          env: Vectorized env.
        """
        self.root = root if root else env.clone_full_state(how_many=1)[0]
        self.root = tuple(self.root)  # for hash-ability
        self.env = env
        self.room_structure = None
        self._debug = False
        self._solve_game()

    # solves starting from self.root
    def _solve_game(self):
        self._parents = {self.root: set()}
        self._end_states = set()
        self._find_end_states(self.root)  # find end_states
        self._dist = []  # list of dicts, one for each end-states
        self._length = []
        self._states = set()
        for end_state in self._end_states:
            dist, length = self._shortest_path(end_state)
            self._states = self._states.union(set(dist.keys()))
            self._dist.append(dist)
            self._length.append(length)
        self._value = {vertex: max([length[vertex] * 11 - dist[vertex]
                                    for length, dist in zip(self._length, self._dist)
                                    if vertex in dist])
                       for vertex in self._states}
        # Add dead end states to value dictionary
        for node in self._parents:
            if node not in self._value:
                self._value[tuple(node)] = -np.inf

    # BFS: executed with argument self.root to find end_states
    def _find_end_states(self, root):
        Q = deque([(root, False)])
        seen = {root}
        while Q:
            vertex, done = Q.popleft()
            if done:
                continue
            # env step
            self.env.restore_full_state([np.array(vertex)] * self.env.nenvs)
            _, rewards, dones, _ = self.env.step(list(range(4)))  # self.game.legal_actions())  # TODO: change
            cloned_states = [tuple(cs) for cs in self.env.clone_full_state()]
            for idx, adj in enumerate(cloned_states):
                if adj in self._parents:
                    self._parents[adj].add((vertex, rewards[idx]))
                else:
                    self._parents[adj] = {(vertex, rewards[idx])}
                if adj not in seen:
                    Q.append((adj, dones[idx]))
                    seen.add(adj)
                    if dones[idx] and adj not in self._end_states:
                        self._end_states.add(adj)

    # Dijkstra (shortest path from solution to all states). Needs reward shift.
    # Run from the end states to produce distance to every node
    def _shortest_path(self, root):  # state either root or goal state
        Q = []
        heapq.heappush(Q, [0., root, 0])  # [priority, vertex, true reward]
        dist = {root: 0.}
        length = {root: 0}
        seen = {root}

        while Q:
            priority, vertex, vlen = heapq.heappop(Q)
            dist[vertex] = priority
            length[vertex] = vlen
            seen.add(vertex)
            for neighbour, reward in self._parents[vertex]:
                if neighbour in seen:
                    continue
                alt = dist[vertex] + (11 - reward)  # we use minheap, hence we min -rewards and shift to make it >=0
                if neighbour not in dist:
                    heapq.heappush(Q, [alt, neighbour, vlen + 1])
                else:  # if adj in dist and not seen => adj in Q
                    if alt < dist[neighbour]:
                        nlen = length[neighbour]  # if neighbour in dist => in length
                        idx = Q.index([dist[neighbour], neighbour, nlen])
                        Q.pop(idx)
                        heapq.heapify(Q)
                        heapq.heappush(Q, [alt, neighbour, vlen + 1])
                        dist[neighbour] = alt
                        length[neighbour] = vlen + 1
        return dist, length

    # everything is exact, hece 0. training loss
    def train_step(self):
        return 0.

    def moments(self):
        return 0.

    # INFO: guarantees that the root is 0'th position
    def _dump_vf_for_root(self, root):
        # This forces reclulation of the value functions if need, and calculated _parents.
        self.__call__(root)
        dump = [(root, self._value.get(tuple(root), -np.inf))]
        dump.extend([(node, self._value.get(tuple(node), -np.inf))
                     for node in self._parents if node != tuple(root)])

        return dump

    # obs.shape = (1+nenvs, frame)
    # usually states are batched (1+nenvs)
    # WARNING: __call__ assumes that states (or obs) come from the same sokoban level
    def __call__(self, states=None, obs=None):
        assert states is not None, "States cannot be None"
        if not isinstance(states, list):  # if called on states
            states = [states]
        if self.root is None or states[-1][-1] != self.root[-1]:  # check seed
            self.root = tuple(states[-1])  # [-1] = leaf
            self._solve_game()
        return np.array([self._value[tuple(state)] for state in states])


class StatesBatchValue(Value, ABC):
    def __call__(self, obs=None, states=None):
        assert states is not None, "Only support states, got None."
        if not isinstance(states, list):
            states = [states]
        return np.array([self._evaluate_state(state) for state in states])

    def _evaluate_state(self, state):
        # This should be defined in subclasses.
        raise NotImplementedError


class SparseValuePerfect(StatesBatchValue, ABC):
    def __init__(self, env, root=None):
        """ Wrapper for ValuePerfect, giving perfect value for sparse rewards."""
        self.perfect_dense = ValuePerfect(env, root)

    def _evaluate_state(self, state):
        value_dense = self.perfect_dense(states=[state])[0]
        assert isinstance(value_dense, float)
        if value_dense == -np.inf:
            return 0.
        else:
            return 1.


class NoisyValueWrapper(StatesBatchValue, ABC):
    """Value Function with iid gausian noise.

  Perturbations are cached when given state is seen for first time.
  """

    def __init__(self, value_fn, noise_std=0.1, seed=None, reset_cache=False):
        """

    Args:
      value_fn: callable (obs, states) -> values. Ignores obs.
      reset_cache: if reset cached noise, when called with state from new room.
        Use it only with Sokoban states.
    """
        self.random_state = np.random.RandomState(seed)
        self.value_fn = value_fn
        self.noise_std = noise_std
        self.room = None
        self._reset_cache_for_new_room = reset_cache
        self._evaluated = dict()

    def _reset_cache(self):
        print("NoisyValueWrapper: state from new room observed, "
              "reseting cached noise.")
        self._evaluated = dict()

    def _evaluate_state(self, state):
        if self._reset_cache_for_new_room:
            # Make sense only for Sokoban
            if state[-1] != self.room:
                self._reset_cache()
                self.room = state[-1]
        if not tuple(state) in self._evaluated:
            self._evaluated[tuple(state)] = \
                self.value_fn(states=[state]) + \
                self.random_state.normal(0, self.noise_std)
        return self._evaluated[tuple(state)]


class ValueMapWrapper(StatesBatchValue, ABC):
    def __init__(self, value_fn, map):
        self.value_fn = value_fn
        if isinstance(map, dict):
            self.map_fn = lambda value: map[value] if value in map else value
        else:
            assert hasattr(map, "__call__")
            self.map_fn = map

    def _evaluate_state(self, state):
        return self.map_fn(self.value_fn(states=[state])[0][0])


class ValueFromKerasNet(Value, ABC):
    def __init__(self, model, env_kwargs):
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        self.env = SokobanEnv(**env_kwargs)
        self.env.reset()

    def _network_prediction(self, state):
        self.env.restore_full_state(state)
        obs = self.env.render()
        return self.model.predict(np.expand_dims(obs, axis=0))

    def __call__(self, state):
        raise NotImplementedError


class ValueFromOneHeadNet(ValueFromKerasNet, ABC):
    def __init__(self, *args, **kwargs):
        super(ValueFromOneHeadNet, self).__init__(*args, **kwargs)
        assert len(self.model.outputs) == 1, \
            "Given model has {} outputs, expected 1".format(len(self.model.outputs))

    def __call__(self, states):
        if not isinstance(states, list):
            states = [states]
        return np.concatenate([
            self._network_prediction(state)
            for state in states
        ])




class QFromV(object):
    def __init__(self, value_function, env_kwargs,
                 nan_for_zero_value=True, copy_negative=True):
        self.value_function = value_function
        self.env = SokobanEnv(**env_kwargs)
        self.env.reset()
        self.nan_for_zero_value = nan_for_zero_value
        self.copy_negative_values = copy_negative

    @property
    def env_n_actions(self):
        return self.env.action_space.n

    def q_values(self, state):
        q_values = list()
        if self.nan_for_zero_value:
            # Value might not have children for Sokoban success states.
            if self.value_function(states=state) == 0:
                return [np.nan] * self.env_n_actions
        if self.copy_negative_values:
            # For speed-up
            val = self.value_function(states=state)[0]
            if val < 0:
                return [val] * self.env_n_actions

        for action in range(self.env_n_actions):
            self.env.restore_full_state(state)
            ob, reward, done, _ = self.env.step(action)
            value = reward
            child_state = self.env.clone_full_state()
            if not done:
                value += self.value_function(states=child_state)[0]
            q_values.append(float(value))
        return q_values


class Policy(object):
    def act(self, state, return_single_action=True):
        best_actions = self.best_actions(state)
        if return_single_action:
            return best_actions[0]
        else:
            return best_actions

    def best_actions(self, state):
        raise NotImplementedError


class PolicyFromValue(Policy):

    def __init__(self, value_function, env_kwargs):
        """

    Args:
      value_function: callable: state -> value
    """
        self.q_value = QFromV(value_function, env_kwargs)
        self.env_n_actions = self.q_value.env_n_actions

    def best_actions(self, state):
        """Choose best actions from state, according to value_function."""
        q_values = self.q_value.q_values(state)
        q_values = np.round(q_values, 6)
        if np.isnan(q_values).any():
            assert np.isnan(q_values).all()
            best_actions = np.arange(q_values.size)
        else:
            best_value = np.max(q_values)
            best_actions = np.where(q_values == best_value)[0]
        return best_actions


class PolicyFromNet(Policy):
    def __init__(self, model, env_kwargs):
        self.render_env = SokobanEnv(**env_kwargs)
        self.env_n_actions = self.render_env.action_space.n
        if isinstance(model, str):
            self.model = load_model(model)
        else:
            self.model = model
        self.env = SokobanEnv(**env_kwargs)
        self.env.reset()
        assert len(self.model.outputs) == 1

    def best_actions(self, state):
        self.env.restore_full_state(state)
        ob = self.env.render()
        policy = self.model.predict(np.expand_dims(ob, axis=0))[0]
        best_actions = [np.argmax(policy)]
        return best_actions


class PolicyFromFullTree(Policy):
    def __init__(self, value_fn, env_kwargs, depth=4):
        self.render_env = SokobanEnv(**env_kwargs)
        self.env_n_actions = self.render_env.action_space.n
        self.value_function = value_fn
        self.env = SokobanEnv(**env_kwargs)
        self.env.reset()
        self.depth = depth
        self.nodes = dict()

    def best_actions(self, state):
        # Produce all action sequences
        seq_ = [range(self.env.action_space.n)] * self.depth
        action_seq = list(product(*seq_))
        # print("len(action_seq) {}".format(len(action_seq)))
        for actions in action_seq:
            root_action = actions[0]
            self.env.restore_full_state(state)
            branch_reward = 0
            current_depth = 0
            for action in actions:
                current_depth += 1
                ob, reward, done, _ = self.env.step(action)
                branch_reward += reward
                node = tuple(self.env.clone_full_state())
                if node not in self.nodes:
                    value = self.value_function(
                        states=np.array(node))  # self.model.predict(np.expand_dims(ob, axis=0))[0]
                    if done:
                        value += 1000
                    self.nodes[node] = (value, branch_reward, current_depth, root_action, actions[:current_depth])
                else:
                    value, previous_reward, previous_depth, _, _ = self.nodes[node]
                    if previous_depth > current_depth:
                        # if previous_reward > branch_reward:
                        #   assert branch_reward > 10., "{} {}".format(previous_reward, branch_reward)
                        self.nodes[node] = (value, branch_reward, current_depth, root_action, actions[:current_depth])
                if done:
                    break
        # self.nodes.values()
        best_node = max(self.nodes.keys(), key=(lambda node: self.nodes[node][0] + self.nodes[node][1]))
        node_value, branch_reward, current_depth, root_action, actions = self.nodes[best_node]
        # print("Distinct leaves {}".format(len(self.nodes)))
        # print("Node value {}, reward {:.1f}, depth {}, action {}, actions {}".format(
        #     node_value, branch_reward, current_depth, root_action, actions))
        return [root_action]


class ValueLoader(Value, ABC):
    def __init__(self):
        self.data = None

    def _dump_vf_for_root(self, root):
        if self.data is None:
            raise Exception("Data should be loaded first with load_vf_for_root")
        return self.data

    def _restore_vf_for_root(self, data):
        self.data = data
        self._value = {tuple(state): value for state, value in data}

    def __call__(self, states=None, obs=None):
        assert states is not None, "States cannot be None"
        if not isinstance(states, list):  # if called on states
            states = [states]

        return np.array([self._value[tuple(state)] for state in states])

    def moments(self, inputs):
        raise NotImplementedError

    def get_all_value_reference(self):
        return self._value


class ValueCompressionTest(unittest.TestCase):

    def assert_decompressed_equal(self, c1, c2):
        # Decompressed tuples consist numpy arrays, so assertEqual would not work.
        self.assertEqual(len(c1), len(c2))
        for t1, t2 in zip(c1, c2):
            self.assertTrue((t1[0] == t2[0]).all())
            self.assertEqual(t1[1], t2[1])
            self.assertEqual(t1[1], t2[1])

    def test_compression(self):
        decompressed = [
            (np.array([0, 7, 4, 123456]), 10.2),
            (np.array([1, 0, 2, 54321]), -float('inf')),
        ]
        value = Value()
        compressed = value.compress(decompressed)
        decompressed2 = value.decompress(compressed)
        self.assert_decompressed_equal(decompressed, decompressed2)


if __name__ == '__main__':
    unittest.main()
