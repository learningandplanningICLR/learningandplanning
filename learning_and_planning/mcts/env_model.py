from abc import ABCMeta, abstractmethod

import gin
import numpy as np

import tensorflow
from tensorflow.python.keras.models import load_model

from gym_sokoban_fast.sokoban_env_fast import HashableState, FieldStates


def image_with_embedded_action(img, action, action_space_size):
  """

  Canonical way of concatenating action and image for next frame prediction
  task.

  Args:
    img: array of shape (height, width, channels)

  Returns:
    embedded image: array of shape (height, width, channels + action_space_size)
      action is one hot encoded after original channels.
  """
  assert len(img.shape) == 3
  action_map_ohe = np.zeros(
    img.shape[:2] + (action_space_size,), dtype=np.uint8
  )
  action_map_ohe[:, :, action] = 1
  return np.concatenate([img, action_map_ohe], axis=2)



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


@gin.configurable
class SimulatedSokobanEnvModel(ModelBase):
    """

    Env model with Neural Networks. Implemented only for Sokoban.
    """

    def __init__(self, env, model_path, force_hashable=True):
        self.model = None
        self.env = env
        assert self.env.mode == "one_hot"
        assert tuple(self.env.dim_room) == (
        10, 10), f"{self.env.dim_room} {type(self.env.dim_room)}"
        assert self.env.reward_box_on_target == 0., f"{self.env.reward_box_on_target}"
        assert self.env.penalty_for_step == 0.
        self.num_actions = env.action_space.n
        self.force_hashable = force_hashable
        self.model_path = model_path

    def step(self, state, action):
        if self.model is None:
            # load model only after __init__ to avoid weights reinitialization
            self.model = load_model(self.model_path)

        _state = state.np_array if self.force_hashable else state
        # self.env.unwrapped.restore_full_state(_state)
        # ob = self.env.render(mode=self.env.mode)
        # real_output = self.env.step(action)
        x = image_with_embedded_action(state.one_hot, action,
                                       action_space_size=self.env.action_space.n)
        pred = self.model.predict(np.expand_dims(x, 0).astype(float))
        pred_frame, if_done = tuple(pred)
        middle_pred = (pred_frame > 0.1) & (pred_frame < 0.9)
        # if middle_pred.any():
        #     raise ValueError("Suspicious predictions of neural net!")
        pred_frame = pred_frame.round(0)[0]  # remove first "batch" dimension
        if_done = if_done.round(0)
        # print("HACK!!!")
        # if_done = True
        # if (pred_frame != real_output[0]).any():
        #     print("Wrong next_frame prediction")
        # if (if_done == 1)[0] != real_output[2]:
        #     print("Wrong if_done prediction")
        reward = 0
        if if_done:
            reward = self.env.reward_finished
        return pred_frame, reward, if_done, {"solved": if_done}

    def neighbours(self, state):
        output = []
        _state = state.np_array if self.force_hashable else state
        for action in self.legal_actions():
            obs, rew, done, info = self.step(_state, action)
            solved = info["solved"]
            # next_state = self.env.unwrapped.clone_full_state()
            # next_state = HashableNumpyArray(next_state) if self.force_hashable else next_state

            state_np = obs
            agent_pos = np.unravel_index(
                np.argmax(state_np[..., FieldStates.player] +
                          state_np[..., FieldStates.player_target]),
                dims=state_np.shape[:2])
            unmatched_boxes = int(np.sum(state_np[..., FieldStates.box]))

            next_state_sim = HashableState(one_hot=obs, agent_pos=agent_pos,
                                           unmached_boxes=unmatched_boxes)
            output.append((obs, rew, done, solved, next_state_sim))
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
        # TODO(pm):  Do this better
        return self.env.state_space.shape, self.env.state_space.dtype
        # state_ = np.array(self.env.clone_full_state())
        # return state_.shape, state_.dtype

    def observation_space(self):
        return self.env.observation_space.shape, self.env.observation_space.dtype

    def renderer(self):
        def _thunk(states):
            obs = []
            for state in states:
                self.env.restore_full_state_from_np_array_version(state,
                                                                  quick=True)
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
