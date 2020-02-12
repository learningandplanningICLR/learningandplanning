import time

import gym
import numpy as np
from functools import partial
from gym.envs import register
from gym.spaces import Box, Discrete

from learning_and_planning.mcts.curriculum import Curriculum
from learning_and_planning.mcts.env_model import HashableNumpyArray
from learning_and_planning.mcts.serialization import Serializer
from learning_and_planning.mcts.worker import Worker


class ChainEnvironment(gym.Env):
  """
  Implementation of chain environment from Section 4.2.1
  of Randomized Prior Functions for Deep Reinforcement Learning 
  Osband et al. 2018
  """

  render_cell_size = 30

  def __init__(self, N=10, perfect_path_reward=-0.01, solved_reward=1.):
    """

    Args:
      N: size of grid
      seed: seed for generation of action interpretation mask
      perfect_path_reward: total reward for perfect game, not counting final
        reward (equal to 1)
    """
    super(ChainEnvironment).__init__()
    self.N = N
    self.posX = 0
    self.posY = 0
    self.perfect_path_reward = perfect_path_reward
    self.solved_reward = solved_reward
    # For now we assume that mask is fixed across environments. For this reason
    # we can safely assume that observation == state (this assumption is
    # important for restore_full_state() etc.)
    self.mask = np.random.RandomState(0).rand(N, N) > 0.5
    _cs = ChainEnvironment.render_cell_size + 1
    self.display_mask = 255*np.ones((N*_cs + 1, N*_cs+1, 3), dtype=np.uint8)
    self.display_mask[::_cs, :, :] = 0
    self.display_mask[:, ::_cs, :] = 0
    for x in range(N):
      for y in range(N):
        if self.mask[x, y]:
          self.display_mask[x*_cs:x*_cs + ChainEnvironment.render_cell_size:3,
           y*_cs:y*_cs + ChainEnvironment.render_cell_size:3, :] = 0
    self.observation_space = Box(low=0, high=1, shape=(self.N, self.N, 1), dtype=np.uint8)
    self.state_space = self.observation_space  # state == observation
    self.action_space = Discrete(2)

  @property
  def mode(self):
    return "one_hot"

  @property
  def init_kwargs(self):
    return {
      attr: getattr(self, attr)
      for attr in (
        'N', 'perfect_path_reward', 'solved_reward',
      )
    }

  def _get_state(self):
    state = np.zeros(self.observation_space.shape, dtype=np.uint8)
    state[self.posY, self.posX, 0] = 1
    return state

  def is_done(self):
    return self.posY >= self.N-1

  def is_solved(self):
    return self.posX == self.N - 1

  def step(self, action):
    assert action < self.action_space.n
    if self.is_done():
      # Do not change state, ignore the action.
      return self._get_state(), 0.0, True, {"solved": self.is_solved()}

    direction = 1 if action + self.mask[self.posY, self.posX] == 1 else -1
    self.posX = max(0, self.posX+direction)
    self.posY += 1
    reward = 0.0
    if direction == 1:
      reward = self.perfect_path_reward / (self.N - 1)
    solved = self.is_solved()
    if solved:
      # final reward, for perfect solution
      reward = self.solved_reward
    return self._get_state(), reward, self.is_done(), \
           {"solved": solved}

  def reset(self):
    self.posX = 0
    self.posY = 0

    return self._get_state()

  def render(self, mode='one_hot'):
    if mode == 'human':
      self._get_state()
    elif mode == 'rgb_array':
      ret = np.copy(self.display_mask)
      _cs = ChainEnvironment.render_cell_size + 1
      ret[self.posY*_cs:self.posY*_cs+ChainEnvironment.render_cell_size,
          self.posX*_cs:self.posX*_cs+ChainEnvironment.render_cell_size, 0] = 0
      return ret
    elif mode == "one_hot":
      return self._get_state()

  def restore_full_state(self, state):
    assert isinstance(state, HashableNumpyArray)
    state_np = state.get_np_array_version()
    self.restore_full_state_from_np_array_version(state_np)

  def clone_full_state(self):
    return HashableNumpyArray(self._get_state())

  def seed(self, seed=None):
    # unused for now
    return [seed]

  def restore_full_state_from_np_array_version(self, state_np, quick=False):
    del quick
    assert state_np.shape == self.observation_space.shape
    position_map = state_np[:, :, 0]
    col_sum = position_map.sum(axis=0)
    row_sum = position_map.sum(axis=1)
    assert col_sum.sum() == 1, f'{col_sum.sum()}'
    self.posX = np.where(col_sum == 1)[0][0]
    self.posY = np.where(row_sum == 1)[0][0]

  # def set_maxsteps(self, num_steps):
  #   self.max_steps = num_steps


for N in range(2, 100):
  l = partial(ChainEnvironment, N)
  register(id=rf"ChainEnv{N}-v1", entry_point=l, max_episode_steps=200)


# if __name__ == "__main__":
#
#     import tensorflow as tf
#     import baselines.common.tf_util as U
#     from learning_and_planning.mcts.create_agent import create_agent
#
#     from mrunner.helpers.client_helper import get_configuration
#     import gin.tf.external_configurables
#
#
#     params = get_configuration(print_diagnostics=True,
#                                with_neptune=False,
#                                inject_parameters_to_gin=True)
#
#     sess = U.make_session(num_cpu=1)
#     sess.__enter__()
#
#     value, planner, env_init_kwargs = create_agent(sess)
#     serializer = Serializer()
#
#     sess.run(tf.global_variables_initializer())
#     value.sync()  # sync value weights to root 0
#
#     curriculum = Curriculum(enabled=False)
#
#     worker = Worker(
#             value=value,
#             planner=planner,
#             curriculum=curriculum,
#             game_buffer_size=params['game_buffer_size'],
#             serializer=serializer,
#             training_steps=params['training_steps'],
#     )
#     worker.run()