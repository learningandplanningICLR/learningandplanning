from abc import ABCMeta, abstractmethod

import numpy as np


class ModelBase(metaclass=ABCMeta):
    @abstractmethod
    def step(self, state, action):
        pass

    @abstractmethod
    def neighbours(self, state):
        pass

    @abstractmethod
    def state(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def state_space(self):
        pass

    @abstractmethod
    def observation_space(self):
        pass

    @abstractmethod
    def renderer(self):
        pass


class ModelEnvPerfect(ModelBase):
    def __init__(self, env, force_hashable=True):
        self.env = env
        self.num_actions = env.action_space.n
        self.force_hashable = force_hashable

    def step(self, state, action):
        _state = state.np_array if self.force_hashable else state
        self.env.unwrapped.restore_full_state(_state)
        return self.env.step(action)

    def neighbours(self, state):
        output = []
        _state = state.np_array if self.force_hashable else state
        for action in self.legal_actions():
            self.env.unwrapped.restore_full_state(_state)
            obs, rew, done, info = self.env.step(action)
            solved = info["solved"]
            next_state = self.env.unwrapped.clone_full_state()
            next_state = HashableNumpyArray(next_state) if self.force_hashable else next_state
            output.append((obs, rew, done, solved, next_state))
        return zip(*output)

    def state(self):
        state = self.env.unwrapped.clone_full_state()
        return HashableNumpyArray(state) if self.force_hashable else state

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def legal_actions(self):
        return list(range(self.num_actions))

    def state_space(self):
        return self.env.observation_space.shape, self.env.observation_space.dtype
        # state_ = np.array(self.env.clone_full_state())
        # return state_.shape, state_.dtype

    def observation_space(self):
        return self.env.observation_space.shape, self.env.observation_space.dtype

    def renderer(self):
        def _thunk(states):
            obs = []
            for state in states:
                self.env.restore_full_state_from_np_array_version(state, quick=True)
                obs.append(self.env.render())
            stack = np.stack(obs, axis=0)
            return stack

        return _thunk


class HashableNumpyArray:
    
    hash_key = np.random.normal(size=10000)
    
    def __init__(self, np_array):
        assert type(np_array) is np.ndarray, "This works only for np.array"
        self.np_array = np_array
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            flat_np = self.np_array.flatten()
            self._hash = int(np.dot(flat_np, HashableNumpyArray.hash_key[:len(flat_np)])*10e8)
        return self._hash

    def __eq__(self, other):
        return np.array_equal(self.np_array, other.np_array)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_np_array_version(self):
        return self.np_array
