

class VecEnvCloneRestore(object):
  """Vectorised, single thread environenment.

  API partially same as SubprocVecEnvCloneRestore, note different __init__.
  In particular it passes calls to clone_full_state() and restore_full_state()
  to underlying envs.
  """

  def __init__(self, env_fn, nenvs):
    """
    Args:
      env_fn: callable creating environment.
    """
    self._envs = [env_fn() for _ in range(nenvs)]
    self.action_space = self._envs[0].action_space
    self.observation_space = self._envs[0].observation_space
    self.num_envs = nenvs
    self.dim_room = self._envs[0].unwrapped.dim_room

  @property
  def nenvs(self):
    return self.num_envs

  def step(self, actions):
    return zip(*[env.step(action) for env, action in zip(self._envs, actions)])

  def clone_full_state(self, how_many=None):
    last = self.nenvs if not how_many else min(self.nenvs, how_many)
    return [env.unwrapped.clone_full_state() for env in self._envs[:last]]

  def restore_full_state(self, states, how_many=None):
    last = self.nenvs if not how_many else min(self.nenvs, how_many)
    assert len(states) == last
    return [env.unwrapped.restore_full_state(state) for env, state in
            zip(self._envs[:last], states)]

  def reset(self):
    return [env.reset() for env in self._envs]

  def get_images(self, mode=None):
    return [env.render(mode=mode) for env in self._envs]

  def render(self, how_many=None, mode=None):
    last = self.nenvs if not how_many else min(self.nenvs, how_many)
    return [env.render(mode=mode) for env in self._envs[:last]]

  @property
  def current_seed(self):
    return self._envs[0].unwrapped.current_seed

  # INFO: for curriculum
  @property
  def num_gen_steps(self):
    return self._envs[0].unwrapped.num_gen_steps

  @num_gen_steps.setter
  def num_gen_steps(self, ngs):
    for i in range(self.nenvs):
      self._envs[i].unwrapped.num_gen_steps = ngs

  @property
  def curriculum(self):
    return self._envs[0].unwrapped.curriculum

  @curriculum.setter
  def curriculum(self, curr):
    for i in range(self.nenvs):
      self._envs[i].unwrapped.curriculum = curr

  @property
  def inner_env_init_kwargs(self):
    return self._envs[0].unwrapped.init_kwargs
