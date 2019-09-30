from collections import deque
from typing import List, Iterable

import numpy as np
import pyglet
from PIL import Image, ImageDraw
from gym.core import Wrapper

from baselines import logger
from gym_sokoban.envs import SokobanEnv
from gym_sokoban_fast import SokobanEnvFast
from learning_and_planning.utils.gym_utils import KeysToActionMapping

HUMAN_INFO_KEY = 'human_info'
NOOP_ACTION = -1


class SokobanTransparentWrapperMixin:
    def clone_full_state(self):
        return self.env.clone_full_state()

    def restore_full_state(self, state):
        return self.env.restore_full_state(state)


class InfoDisplayWrapper(Wrapper, SokobanTransparentWrapperMixin):
    """
    Wrapper which displays debug info for human.

    NOTE: This wrapper only displays information stored under HUMAN_INFO_KEY key of `info` dictionary.
    It must be used in conjunction with HumanPrintWrapper which stores some information under this key.
    """
    def __init__(
            self,
            env,
            augment_observations=False,
            reset_info_frequency='step',
            min_text_area_width=300,
            font_color=(255, 0, 0),
            viewer=None
    ):
        """
        Arguments:
            env: environment to wrap
            augment_observations: if True, human info will be included in observations. Recommended for playing mode.
            reset_info_frequency: if set to 'step', human info will be reset before taking every step
            min_text_area_width: min width of text area in pixels
            font_color: color of printed text
        """
        Wrapper.__init__(self, env=env)
        self.augment_observations = augment_observations
        self.reset_info_frequency = reset_info_frequency
        self.human_info = []
        self.font_color = font_color
        self.viewer = viewer
        self.min_text_area_width = min_text_area_width

        if self.augment_observations:
            ob_space = self.observation_space
            text_area_shape = self._get_text_area_shape(ob_space.shape)
            text_area_shape[1] += ob_space.shape[1]
            ob_space.shape = tuple(text_area_shape)

    def step(self, action):
        if self.reset_info_frequency == 'step':
            self.reset_human_info()
        obs, reward, done, info = self.env.step(action)
        self.human_info = info.get(HUMAN_INFO_KEY, [])

        if self.augment_observations:
            obs = self.display_human_info(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.reset_human_info()
        obs = self.env.reset(**kwargs)
        if self.augment_observations:
            obs = self.display_human_info(obs)
        return obs

    def render(self, mode='rgb_array', **kwargs):
        if not mode or ('human' not in mode and 'rgb_array' not in mode):
            return self.env.render(mode, **kwargs)

        obs = self.env.render(mode='rgb_array', **kwargs)
        obs = self.display_human_info(obs)

        if 'rgb_array' in mode:
            return obs

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer(maxwidth=obs.shape[1])
        self.viewer.imshow(obs)
        return self.viewer.isopen

    def display_human_info(self, obs):
        text_area_shape = self._get_text_area_shape(obs.shape)
        img = Image.fromarray(np.zeros(text_area_shape, dtype='uint8'))
        draw = ImageDraw.Draw(img)

        multiline_text = '\n'.join(self.human_info)
        draw.text((0, 0), multiline_text, self.font_color)

        text_area = np.asarray(img)
        return np.concatenate((obs, text_area), axis=1)

    def reset_human_info(self):
        self.human_info = []

    def _get_text_area_shape(self, observation_shape):
        shape = list(observation_shape)
        text_area_width = max(shape[1], self.min_text_area_width)
        shape[1] = text_area_width
        return shape


class HumanPrintWrapper(Wrapper, SokobanTransparentWrapperMixin):
    """
    Wrapper which stores some information (defined in `build_texts`)
    which can be later used by InfoDisplayWrapper.
    """
    def add_human_info(self, obs, reward, done, info):
        info = info or {}
        human_info = info.setdefault(HUMAN_INFO_KEY, [])
        human_info.extend(self.build_texts(obs, reward, done, info))
        return obs, reward, done, info

    def build_texts(self, obs, reward, done, info) -> List[str]:
        return ['Override `build_texts` method', 'of HumanPrintWrapper class\nto change this default message']

    def step(self, action):
        res = self.env.step(action)
        return self.add_human_info(*res)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class RewardPrinter(HumanPrintWrapper):
    def build_texts(self, obs, reward, done, info):
        return ['Reward: {}'.format(reward)]


class ValuePrinter(HumanPrintWrapper):
    def __init__(self, env, value_fun):
        """

        Args:
          value_fun: callable: observation -> value
        """
        super().__init__(env)
        self.value_fun = value_fun

    def build_texts(self, obs, reward, done, info):
        val = self.value_fun(obs)
        return ['Value: ' + str(val), 'Reward: ' + str(reward)]


class ChildrenValuePrinter(HumanPrintWrapper):
  def __init__(self, env, value_fun):
    """

    Args:
      value_fun: callable: obs, states -> value, which would be call by key
        `states`
    """
    super().__init__(env)
    self.render_env = SokobanEnv(**env.init_kwargs)
    self.value_fun = value_fun

  def formatted_state_value(self, state):
    return "{:.2f}".format(self.value_fun(states=state)[0][0])

  def build_texts(self, obs, reward, done, info):
    child_values = list()
    state = self.env.clone_full_state()
    value_str = self.formatted_state_value(state)
    for action in range(self.render_env.action_space.n):
      self.render_env.restore_full_state(state)
      self.render_env.step(action)
      child_state = self.render_env.clone_full_state()
      child_value_str = self.formatted_state_value(child_state)
      child_values.append(child_value_str)
    print('Children values: {}'.format(" ".join(child_values)))
    return [
      'Value: {}'.format(value_str),
      'Children values: {}'.format(" ".join(child_values))
    ]


class PlayWrapper(Wrapper, SokobanTransparentWrapperMixin):
    """ Wrapper that enables using `play` gym function to play with keyboard. """
    keys_to_action = {
        (ord('w'),): 0,  # UP
        (ord('s'),): 1,  # DOWN
        (ord('a'),): 2,  # LEFT
        (ord('d'),): 3,  # RIGHT
        (ord('t'),): 4,  # PULL UP
        (ord('g'),): 5,  # PULL DOWN
        (ord('f'),): 6,  # PULL LEFT
        (ord('h'),): 7,  # PULL RIGHT
        # Every other key sequence will be noop
    }

    def __init__(self, env):
        super().__init__(env)
        # Store last observations, rewards, dones and info in case of noop actions
        self.last_obs = None
        self.last_rew = 0
        self.last_done = False
        self.last_info = {}
        self._last_action = None

    def step(self, action):
        if action != NOOP_ACTION and action != self._last_action:
            self.last_obs, self.last_rew, self.last_done, self.last_info = self.env.step(action)
        self._last_action = action
        return self.last_obs, self.last_rew, self.last_done, self.last_info

    def reset(self, **kwargs):
        self.last_obs = self.env.reset(**kwargs)
        self.last_rew = 0
        self.last_done = False
        self.last_info = {}
        return self.last_obs

    def get_keys_to_action(self):
        return KeysToActionMapping(self.keys_to_action, noop_action=NOOP_ACTION)

    def play(self, fps=30, **kwargs):
        from gym.utils.play import play
        return play(env=self, fps=fps, **kwargs)


class RestoreStateWrapper(Wrapper, SokobanTransparentWrapperMixin):
    """ Wrapper that uses predefined states on `reset` function.
    Set desired state in constructor or directly set `next_state` attribute.

    Acceptable values for `next_state` attributes:
    - state
    - list of states
    - iterator of states

    Example usage:
    >>> env = SokobanEnv()
    >>> first_obs = env.reset()
    >>> first_state = env.clone_full_state()
    >>> restore_wrapper = RestoreStateWrapper(env, first_state)
    >>> second_obs = restore_wrapper.reset()
    >>> assert np.equal(first_obs, second_obs).all()
    >>> assert np.equal(first_state, restore_wrapper.next_state).all()

    with list of states
    >>> restore_wrapper = RestoreStateWrapper(SokobanEnv())
    >>> restore_wrapper.next_state = ['state1', 'state2', 'state3']
    >>> assert restore_wrapper.next_state == 'state1'
    >>> assert restore_wrapper.next_state == 'state2'
    >>> assert restore_wrapper.next_state == 'state3'
    """

    def __init__(self, env, state_to_restore=None):
        super().__init__(env)
        self.state_to_restore = state_to_restore

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.env.restore_full_state(self.state_to_restore)
        obs = self.env.render()

        def one_hot_to_value(obs):
            return np.where(obs == 1)[2].reshape(obs.shape[:2])
        print(one_hot_to_value(obs))
        return obs

    def render(self, mode='one_hot', **kwargs):
        return self.env.render(mode=mode, **kwargs)

    @property
    def next_state(self):
        state = self.state_to_restore
        assert state is not None, \
            'Set `next_state` attribute (in constructor or directly on wrapper instance) before calling reset.'

        # Handle list of states
        if isinstance(state, list):
            print(state)
            if state:
                return state.pop(0)
            else:
                raise RuntimeError('States list exhausted')
        # Handle iterator of states
        try:
            return next(state)
        except TypeError:
            pass
        except StopIteration as e:
            raise RuntimeError('States iterator exhausted') from e

        # Assume state is just a state
        return state

    @next_state.setter
    def next_state(self, value):
        self.state_to_restore = value


def one_level_callable(**kwargs):
    env = SokobanEnvFast(**kwargs)
    env.reset()
    state = env.clone_full_state()
    return RestoreStateWrapper(env, state)


def test_rendering():
    env = InfoDisplayWrapper(
        RewardPrinter(
            SokobanEnv()
        ),
        augment_observations=True,
        min_text_area_width=500
    )
    env.reset()
    env.step(0)
    obs = env.render()
    assert obs.shape == (80, 580, 3)

    env.render(mode='human')
    from time import sleep
    sleep(2)


def test_playing():
    env = PlayWrapper(
        InfoDisplayWrapper(
            RewardPrinter(
                SokobanEnv(num_boxes=1, game_mode="Magnetic", penalty_pull_action=-0.3)
            ),
            augment_observations=True,
            min_text_area_width=500
        )
    )
    env.play()


if __name__ == '__main__':
    # test_rendering()
    test_playing()
