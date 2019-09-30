import pickle
import zlib
from enum import Enum
import os

from pathlib import Path

import numpy as np

from gym_sokoban.envs import SokobanEnv
from learning_and_planning.evaluator_utils.env_state_types import EnvState
from learning_and_planning.mcts.value import ValueLoader, PolicyFromValue
from learning_and_planning.evaluator_utils.supervised_target import Target
from learning_and_planning.supervised_loss import UNSOLVABLE_FLOAT


SHARD_PATH_FORMAT = "{}_{:04d}"


def n_channels_from_mode(mode):
  return {
    "one_hot": 7,
    "binary_map": 4,
  }[mode]


def compress_np_array(array):
  dtype_str = str(array.dtype)
  assert dtype_str != "object"
  shape = array.shape
  compressed_array = zlib.compress(array.flatten())
  return dict(dtype=dtype_str, shape=shape, bytes=compressed_array)


def decompress_np_array(compressed):
  return np.fromstring(
      zlib.decompress(compressed['bytes']), dtype=compressed['dtype']
  ).reshape(compressed['shape'])


def stratified_sample(values, q, max_size, random_state, max_solved_ratio=0.1):
  """Sample index acording to values.

  First samples solved state up to max_size * max_solved_ratio,
  then dead and solvable in 50:50 proportion to fill max_size.
  """

  state_types = np.array([env_state_type(v, q).value for v, q in zip(values, q)])
  n_solved = (state_types == EnvState.SOLVED.value).astype(np.int).sum()
  n_solvable = (state_types == EnvState.SOLVABLE.value).astype(np.int).sum()
  n_dead = (state_types == EnvState.DEAD.value).astype(np.int).sum()

  solved_to_sample = min(n_solved, int(max_size * max_solved_ratio))
  max_not_solved_to_sample = max_size - solved_to_sample

  dead_to_sample = int(min(n_solvable, n_dead, max_not_solved_to_sample/2))
  solvable_to_sample = dead_to_sample

  # Might require numpy>=1.16.2
  solved_ix = random_state.choice(
    np.where(state_types == EnvState.SOLVED.value)[0], solved_to_sample,
    replace=False
  )

  solvable_ix = random_state.choice(
    np.where(state_types == EnvState.SOLVABLE.value)[0], solvable_to_sample,
    replace=False
  )

  dead_ix = random_state.choice(
    np.where(state_types == EnvState.DEAD.value)[0], dead_to_sample,
    replace=False
  )

  return np.concatenate([dead_ix, solved_ix, solvable_ix])


def simple_sample(values, q, max_size, random_state):
  """Sample index without replacement."""
  size = min(values.shape[0], max_size)
  return random_state.choice(values.shape[0], size, replace=False)


def one_hot_sign(v):
  """ Returns one_hot array of shape (3,) indicating sign of scalar v.

  Args:
    v - scalar
  """

  ret = np.zeros(3, dtype=np.int32)
  ret[np.int(np.sign(v) + 1)] = 1
  return ret


def _load_shard(shard, data_files_prefix):
  file_name = Path(SHARD_PATH_FORMAT.format(data_files_prefix, shard))
  with open(file_name, "rb") as file:
    data = pickle.load(file)
  return data


def infer_data_version(shard_data):
  data_for_board = shard_data[0]
  version = None
  if isinstance(data_for_board, dict):
    assert "full_env_state" in data_for_board
    assert "perfect_value" in data_for_board
    assert "perfect_q" in data_for_board
    version = "v2"
  elif isinstance(data_for_board, list):
    assert len(data_for_board[0]) == 3, "Expected (state, seed, value) tuple."
    version = "v1"
  if not version:
    raise ValueError("Unable to infer data version.")
  return version


def _load_shard_vf(shard, data_files_prefix, env_kwargs,
                   filter_values_fn=None, transform_values_fn=None):
  data = _load_shard(shard, data_files_prefix)
  render_env = SokobanEnv(**env_kwargs)
  data_x = []
  data_y = []
  vf = ValueLoader()
  for vf_for_root in data:
    root = vf.load_vf_for_root(vf_for_root, compressed=True)
    data = vf.dump_vf_for_root(root)
    for env_state, v in data:
      if filter_values_fn:
        if filter_values_fn(v):
          continue
      if transform_values_fn:
        v = transform_values_fn(v)
      render_env.restore_full_state(env_state)
      ob = render_env.render(mode=render_env.mode)
      data_x.append(ob)
      data_y.append(v)
  data_y = np.asarray(data_y)
  if len(data_y.shape) == 1:
    data_y = data_y.reshape((len(data_y), 1))
  return np.asarray(data_x), data_y, {}


def _load_shard_best_action_ignore_finall(shard, data_files_prefix, env_kwargs):
  """ Choose best action

  If all actions are equally good, give special target value (equal to
  env.action_space.n). For Sokoban this will separate dead ends.
  (for which there is no good action).
  """
  boards = _load_shard(shard, data_files_prefix)
  render_env = SokobanEnv(**env_kwargs)
  data_x = []
  data_y = []
  data_value = []
  vf = ValueLoader()
  policy = PolicyFromValue(vf, env_kwargs)
  assert policy.env_n_actions == render_env.action_space.n
  for vf_for_root in boards:
    root = vf.load_vf_for_root(vf_for_root, compressed=True)
    data = vf.dump_vf_for_root(root)
    for node_state, v in data:
      if v in [0, -float("inf")]:
        continue

      render_env.restore_full_state(node_state)
      ob = render_env.render(mode=render_env.mode)
      data_x.append(ob)
      best_actions = policy.act(node_state, return_single_action=False)
      y = np.min(best_actions)
      one_hot_y = np.zeros(shape=render_env.action_space.n, dtype=np.int)
      one_hot_y[y] = 1
      data_y.append(one_hot_y)
      data_value.append(v)
  return np.asarray(data_x), np.asarray(data_y), \
         dict(value=np.asarray(data_value))


def load_shard_v1(shard, target, data_files_prefix, env_kwargs):
  if target == Target.VF:
    return _load_shard_vf(shard, data_files_prefix, env_kwargs,
                          transform_values_fn=lambda v: -2 if v == -float("inf") else v)
  elif target == Target.VF_SOLVABLE_ONLY:
    return _load_shard_vf(shard, data_files_prefix, env_kwargs,
                          filter_values_fn=lambda v: v <= 0)
  elif target == Target.STATE_TYPE:
    return _load_shard_vf(shard, data_files_prefix, env_kwargs,
                          transform_values_fn=lambda v: one_hot_sign(v))
  elif target == Target.BEST_ACTION:
    return _load_shard_best_action_ignore_finall(shard, data_files_prefix, env_kwargs)
  else:
    raise ValueError("Unknown target {}".format(target))


def env_state_type(perfect_value, perfect_q):
  """

  Returns:
    EnvState instance
  """
  if perfect_value == -np.inf:
    state_type = EnvState.DEAD
  elif np.isnan(perfect_q).any():
    assert np.isnan(perfect_q).all()
    assert perfect_value == 0.
    state_type = EnvState.SOLVED
  else:
    state_type = EnvState.SOLVABLE
  return state_type


def one_hot_env_state_type(perfect_value, perfect_q):
  n_types = 3
  state_type = env_state_type(perfect_value, perfect_q)
  one_hot_type = np.zeros(shape=n_types, dtype=np.int)
  one_hot_type[state_type.value] = 1
  return one_hot_type


def is_solvable_state(perfect_v, perfect_q):
  return env_state_type(perfect_v, perfect_q) == EnvState.SOLVABLE


def best_action_one_hot(perfect_v, perfect_q):
  best_action = np.argmax(perfect_q)
  one_hot_action = np.zeros(shape=perfect_q.shape, dtype=np.int)
  one_hot_action[best_action] = 1
  return one_hot_action


def transform_value(v_arr, q_arr, transform_single_fn):
  y = [transform_single_fn(v, q) for v, q in zip(v_arr, q_arr)]
  y = np.array(y)
  if len(y.shape) == 1:
    y = y.reshape((len(y), 1))
  return y


def large_negative_for_unsolvable(v, q):
  if is_solvable_state(v, q):
    return v
  else:
    return UNSOLVABLE_FLOAT


def extract_target(perfect_v, perfect_q, target):

  transform_single = {
    Target.VF: lambda v, q: -2 if v == -float("inf") else v,
    Target.VF_SOLVABLE_ONLY: large_negative_for_unsolvable,
    Target.STATE_TYPE: one_hot_env_state_type,
    Target.BEST_ACTION: best_action_one_hot,
  }

  if target in transform_single:
    data_y = transform_value(perfect_v, perfect_q, transform_single[target])
  elif target == Target.VF_AND_TYPE:
    data_y = {
      Target.VF_SOLVABLE_ONLY.value:
        transform_value(perfect_v, perfect_q, transform_single[Target.VF_SOLVABLE_ONLY]),
      Target.STATE_TYPE.value:
        transform_value(perfect_v, perfect_q, transform_single[Target.STATE_TYPE]),
    }

  return data_y


def process_board_data(board_data, target, env_kwargs, sample_data,
                       max_sample_size, random_state):
  """

  Args:
    board_data: dictionary with keys containing ["full_env_state",
      "perfect_value",  "perfect_q"], mapping to compressed arrays.
  """
  render_env = SokobanEnv(**env_kwargs)
  keys = board_data.keys()
  for key in ["full_env_state", "perfect_value", "perfect_q"]:
    assert key in keys, "{} not in {}".format(key, list(keys))

  data = {key: decompress_np_array(board_data[key]) for key in keys}

  filter_values_fn = lambda v, q: False

  stratified_sample_fn = lambda values, q: stratified_sample(
      values, q, max_sample_size, random_state
  )
  simple_sample_fn = lambda values, q: simple_sample(
      values, q, max_sample_size, random_state
  )

  if target == Target.VF:
    sample_fn = stratified_sample_fn
  elif target == Target.VF_SOLVABLE_ONLY:
    filter_values_fn = lambda v, q: not is_solvable_state(v, q)
    sample_fn = simple_sample_fn
  elif target == Target.STATE_TYPE:
    sample_fn = stratified_sample_fn
  elif target == Target.BEST_ACTION:
    filter_values_fn = lambda v, q: not is_solvable_state(v, q)
    sample_fn = simple_sample_fn
  elif target == Target.VF_AND_TYPE:
    sample_fn = stratified_sample_fn
  else:
    raise ValueError("Unknown target {}".format(target))

  mask = ~np.array([filter_values_fn(v, q) for v, q in
                    zip(data['perfect_value'], data['perfect_q'])
                    ], dtype=np.bool)
  data = {key: data[key][mask] for key in keys}
  if sample_data:
    sample_ix = sample_fn(data["perfect_value"], data["perfect_q"])
    data = {key: data[key][sample_ix] for key in keys}

  data_x = list()
  for node_state in data['full_env_state']:
    render_env.restore_full_state(node_state)
    ob = render_env.render(mode=render_env.mode)
    data_x.append(ob)
  data_x = np.array(data_x)

  data_y = extract_target(data["perfect_value"], data["perfect_q"], target)
  return data_x, data_y, {}


def keys_eqal(d1, d2):
  return tuple(sorted(d1.keys())) == tuple(sorted(d2.keys()))


def concat_arrays_by_name(arrays):
  """

  Args:
    arrays: iterable of arrays (case 1), or iterable of dictionaries of arrays (case 2)

  Returns:
    array in case 1, dictionary of arrays in case 2
  """
  if isinstance(arrays[0], dict):
    assert all(keys_eqal(arrays[0], arr) for arr in arrays)
    ret = {
      name: np.concatenate([arr[name]
                            for arr in arrays])
      for name in arrays[0].keys()
    }
  else:
    ret = np.concatenate(arrays)
  return ret


def load_shard_v2(shard_data, target, env_kwargs, sample_data,
                  max_samples_per_board, seed):

  seeds = np.random.RandomState(seed).randint(2**31, size=len(shard_data))
  arrays = [
      process_board_data(
          board_data, target, env_kwargs, sample_data,
          max_sample_size=max_samples_per_board,
          random_state=np.random.RandomState(seed))
        for board_data, seed in zip(shard_data, seeds)
  ]

  arrays_x, arrays_y, _ = zip(*arrays)

  data_x = np.concatenate(arrays_x)
  data_y = concat_arrays_by_name(arrays_y)
  return data_x, data_y, {}


def load_shard(shard, target, data_files_prefix, env_kwargs,
               sample_data=False, max_samples_per_board=None,
               seed=None):
  shard_data = _load_shard(shard, data_files_prefix)
  version = infer_data_version(shard_data)
  if version == "v1":
    return load_shard_v1(shard, target, data_files_prefix, env_kwargs)
  elif version == "v2":
    return load_shard_v2(shard_data, target, env_kwargs, sample_data,
                         max_samples_per_board, seed)


def infer_number_of_shards(data_files_prefix):
    number_of_shards = 0
    while os.path.isfile(
        Path(SHARD_PATH_FORMAT.format(data_files_prefix, number_of_shards))
    ):
      number_of_shards += 1
    print("Detected {} shards".format(number_of_shards))
    return number_of_shards
