import pickle
import zlib
from copy import deepcopy
from enum import Enum
import os

from pathlib import Path

import numpy as np

from gym_sokoban.envs import SokobanEnv
from learning_and_planning.supervised.supervised_target import Target
from learning_and_planning.supervised.supervised_loss import UNSOLVABLE_FLOAT


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
        # TODO(kc): ValuePerfect does not produce some states which can be
        # obtained after solving game. How to clean it up?
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
  """

  Args:
    perfect_v, perfect_q arrays of shape (num_actions,)

  Returns:
    one_hot_action: one hot vector of shape (num_actions,)
  """
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


def extract_target_from_value(perfect_v, perfect_q, target):

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


def extract_next_frame_input_and_target(full_env_states, render_env):
  data_x = list()
  data_y = list()
  for node_state in full_env_states:
    render_env.restore_full_state(node_state)
    ob = render_env.render(mode=render_env.mode)
    action = render_env.action_space.sample()
    ob_with_action = image_with_embedded_action(
        ob, action, render_env.action_space.n
    )
    data_x.append(ob_with_action)
    next_frame, _, _, _ = render_env.step(action=action)
    data_y.append(next_frame)
  data_x = np.array(data_x)
  data_y = np.array(data_y)
  return data_x, data_y


def assert_v2_keys(data):
  for key in ["full_env_state", "perfect_value", "perfect_q"]:
    assert key in data.keys(), "{} not in {}".format(key, list(data.keys()))

def next_state(state, action, render_env):
  render_env.restore_full_state(state)
  render_env.step(action)
  return render_env.clone_full_state()


def neighbour_states(state, render_env):
  states = [
    next_state(state, action, render_env)
    for action in range(render_env.action_space.n)
  ]
  render_env.restore_full_state(state)
  return states


def sample_action(current_state, render_env, state_tuples_path, random_state):
  # render_env = SokobanEnv(**env_kwargs)
  candidate_states = neighbour_states(current_state, render_env)
  # for s in candidate_states:
  #   print_state(s)
  avoid_state_tuples = deepcopy(state_tuples_path)
  avoid_state_tuples.append(tuple(current_state))
  candidate_actions = [
    action for action in range(render_env.action_space.n)
    if tuple(candidate_states[action]) not in avoid_state_tuples
  ]
  if not candidate_actions:
    candidate_actions = range(render_env.action_space.n)
  return random_state.choice(candidate_actions)


def epsilon_perfect_path(
    node_state, render_env, max_length, random_state, epsilon=0., data=None
):
  if data is None:
    assert epsilon == 1.
  render_env.restore_full_state(node_state)
  current_state = render_env.clone_full_state()
  states_tuples_path = [tuple(node_state)]
  for _ in range(max_length):
    if random_state.random_sample() > epsilon:
      ix = find_record_in_matrix(current_state, data["full_env_state"])
      action = np.argmax(data["perfect_q"][ix])
    else:
      action = sample_action(
        current_state, render_env, states_tuples_path, random_state
      )
    _, _, done, _ = render_env.step(action)
    current_state = render_env.clone_full_state()
    states_tuples_path.append(tuple(current_state))
    if done:
      break

  return [np.array(state_tuple) for state_tuple in states_tuples_path], done


def random_trajectory(
    node_state, render_env, max_length
):
  render_env.restore_full_state(node_state)
  obs = [render_env.render(mode=render_env.mode)]
  actions = []
  done = False
  for _ in range(max_length):
    action = render_env.action_space.sample()
    ob, _, done, _ = render_env.step(action)
    obs.append(ob)
    actions.append(action)
    if done:
      break
  return obs, actions, done


def sample_close_states_ixs(ix, data, render_env, random_state):
  node_state = data["full_env_state"][ix]
  ixs = list()
  for epsilon in [0., 0.3, 0.7, 1.]:
    max_length = random_state.randint(1, 10)
    path, is_solved = epsilon_perfect_path(
      node_state=node_state, render_env=render_env, max_length=max_length,
      random_state=random_state, epsilon=epsilon, data=data
    )
    if is_solved:
      end_state = path[-2]
    else:
      end_state = path[-1]
    ixs.append(
      np.where(
        (data["full_env_state"] == end_state).all(axis=1)
      )[0][0]
    )
  return ixs


def extract_delta_value(data, ixs, render_env, random_state):
  """"Extract input and target for delta_value prediction task"""
  assert_v2_keys(data)
  transform_value = lambda v: -12 if v == -float("inf") else v
  data_x = list()
  data_y = list()
  for ix in ixs:
    if np.isnan(data["perfect_q"][ix]).any():
      continue  # ignore solved states
    node_state = data["full_env_state"][ix]
    render_env.restore_full_state(node_state)
    ob = render_env.render(mode=render_env.mode)
    close_ixs = sample_close_states_ixs(ix, data, render_env, random_state)
    base_value = transform_value(data["perfect_value"][ix])
    for close_ix in close_ixs:
      close_state = data["full_env_state"][close_ix]
      render_env.restore_full_state(close_state)
      close_ob = render_env.render(mode=render_env.mode)
      data_x.append(concat_observations_for_delta_value(ob, close_ob))
      close_value = data["perfect_value"][close_ix]
      if close_value == -float("inf"):
        delta_value = -12.
      else:
        delta_value = close_value - base_value
      data_y.append(delta_value)

  if data_x:
    data_x = np.array(data_x)
    data_y = np.array(data_y).reshape((-1, 1))
  else:
    # if empty, return empty arrays of rigth shape (to eneable concatenation
    # later)
    ob_shape = render_env.observation_space.shape
    # No observations, channels are duplicated
    empty_x_batch_shape = (0, ob_shape[0], ob_shape[1], ob_shape[2] * 2)
    data_x = np.zeros(shape=empty_x_batch_shape,
                      dtype=render_env.observation_space.dtype)
    data_y = np.zeros(shape=(0, 1), dtype=np.float)
  return data_x, data_y


def assert_env_and_state_match(env_kwargs, state):
  assert state.size == \
         env_kwargs["dim_room"][0] * env_kwargs["dim_room"][1] * 2 + 1, \
    "State size do not match env_kwargs. if you're running this in " \
    "supervised pipeline ensure that you've passed right path to data and " \
    "env_ parameters."


def calc_discounted_perfect_value(
    ix, states, perfect_v, perfect_q, render_env, gamma
):
  perfect_state_value = perfect_v[ix]
  steps = 0
  if perfect_state_value == 0.:
    discounted_value = perfect_state_value
  elif perfect_state_value == -np.inf:
    discounted_value = -2.
  else:
    render_env.restore_full_state(states[ix])
    discounted_value = 0.
    discount = 1.
    undiscounted_reward = 0.
    current_ix = ix
    done = False
    while not done:
      steps += 1
      if steps > 1000:
        raise ValueError("More than 1000 steps for perfect path. Something went "
                         "wrong")
      action = np.argmax(perfect_q[current_ix])
      _, reward, done, _ = render_env.step(action)
      current_state = render_env.clone_full_state()
      current_ix = find_record_in_matrix(current_state, states)
      undiscounted_reward += reward
      discounted_value += reward * discount
      discount *= gamma
    assert round(undiscounted_reward, 2) == round(perfect_state_value, 2)
  return discounted_value


def extract_discounted_value(
    sample_ix, states, perfect_v, perfect_q, render_env, gamma=0.99
):
  data_x = list()
  data_y = list()
  render_env.reset()
  for ix in sample_ix:
    state = states[ix]
    render_env.restore_full_state(state)
    ob = render_env.render(mode=render_env.mode)
    data_x.append(ob)
    y = calc_discounted_perfect_value(
      ix, states=states, perfect_v=perfect_v, perfect_q=perfect_q,
      render_env=render_env, gamma=gamma
    )
    data_y.append(y)
  data_x = np.array(data_x)
  data_y = np.array(data_y).reshape((-1, 1))
  return data_x, data_y


def find_record_in_matrix(record, matrix, raise_missing=True):
  """Give index of first occurance of record in matrix."""
  matching = np.where(
    (matrix == record).all(axis=1)
  )[0]
  if matching.size > 0:
    ix = matching[0]
  elif raise_missing:
    raise ValueError("No such record in matrix.")
  else:
    ix = None
  return ix


def extract_best_action_from_framestack(sample_ix, states, perfect_v, perfect_q,
                                        render_env):
  data_x = list()
  data_y = list()
  render_env.reset()
  for ix in sample_ix:
    state = states[ix]
    middle_action_one_hot = best_action_one_hot(perfect_v=perfect_v[ix],
                                                perfect_q=perfect_q[ix])
    middle_action = np.where(middle_action_one_hot)[0][0]
    render_env.restore_full_state(state)
    first_frame = render_env.render(mode=render_env.mode)
    second_frame, _, _, _ = render_env.step(middle_action)
    ix_next = find_record_in_matrix(render_env.clone_full_state(), states,
                                    raise_missing=False)
    if ix_next is None:
      # second_frame is solved state or dead state.
      continue
    target_action_one_hot = best_action_one_hot(perfect_v=perfect_v[ix_next],
                                                perfect_q=perfect_q[ix_next])
    if np.random.rand() < 0.3:
      # For some cases we do not want to use previous frame - during evaluation
      # we would not get one at the begining of the episode. Moreover, using
      # only one frame might give better training signal for network.
      first_frame = None
    data_x.append(
      concat_observations_for_frame_stack(first_frame, second_frame)
    )
    data_y.append(target_action_one_hot)

  if data_x:
    data_x = np.array(data_x)
    data_y = np.array(data_y)
  else:
    # if empty, return empty arrays of right shape (to enable concatenation
    # later)
    ob_shape = render_env.observation_space.shape
    # No observations, channels are duplicated
    empty_x_batch_shape = (0, ob_shape[0], ob_shape[1], ob_shape[2] * 2)
    data_x = np.zeros(shape=empty_x_batch_shape,
                      dtype=render_env.observation_space.dtype)
    data_y = np.zeros(shape=(0, render_env.action_space), dtype=np.float)
  return data_x, data_y


def process_board_data(compressed_data, target, env_kwargs, sample_data,
                       max_sample_size, random_state):
  """

  Args:
    compressed_data: dictionary with keys containing ["full_env_state",
      "perfect_value",  "perfect_q"], mapping to compressed arrays.
  """
  render_env = SokobanEnv(**env_kwargs)
  keys = compressed_data.keys()
  assert_v2_keys(compressed_data)

  data = {key: decompress_np_array(compressed_data[key]) for key in keys}
  assert_env_and_state_match(env_kwargs, data["full_env_state"][0])

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
  elif target == Target.NEXT_FRAME:
    sample_fn = stratified_sample_fn
  elif target == Target.DELTA_VALUE:
    sample_fn = stratified_sample_fn
  elif target == Target.VF_DISCOUNTED:
    sample_fn = stratified_sample_fn
  elif target == Target.BEST_ACTION_FRAMESTACK:
    filter_values_fn = lambda v, q: not is_solvable_state(v, q)
    sample_fn = simple_sample_fn
  elif target == Target.NEXT_FRAME_AND_DONE:
    sample_fn = stratified_sample_fn
  else:
    raise ValueError("Unknown target {}".format(target))

  mask = ~np.array([filter_values_fn(v, q) for v, q in
                    zip(data['perfect_value'], data['perfect_q'])
                    ], dtype=np.bool)
  data = {key: data[key][mask] for key in keys}
  if sample_data:
    sample_ix = sample_fn(data["perfect_value"], data["perfect_q"])
  else:
    raise NotImplemented()

  if target == Target.DELTA_VALUE:
    data_x, data_y = extract_delta_value(
      data, sample_ix, render_env, random_state
    )
  elif target == Target.VF_DISCOUNTED:
    data_x, data_y = extract_discounted_value(
      sample_ix, states=data["full_env_state"], perfect_v=data["perfect_value"],
      perfect_q=data["perfect_q"], render_env=render_env,
    )
  elif target == Target.BEST_ACTION_FRAMESTACK:
    data_x, data_y = extract_best_action_from_framestack(
      sample_ix, states=data["full_env_state"], perfect_v=data["perfect_value"],
      perfect_q=data["perfect_q"], render_env=render_env,
    )
  else:
    data = {key: data[key][sample_ix] for key in keys}
    if target == Target.NEXT_FRAME:
      data_x, data_y = extract_next_frame_input_and_target(
          data["full_env_state"], render_env
      )
    else:
      obs = list()
      for node_state in data['full_env_state']:
        render_env.restore_full_state(node_state)
        ob = render_env.render(mode=render_env.mode)
        obs.append(ob)
      data_x = np.array(obs)
      data_y = extract_target_from_value(
          perfect_v=data["perfect_value"], perfect_q=data["perfect_q"],
          target=target
      )
  if isinstance(data_y, np.ndarray):
    assert len(data_y.shape) > 1, "data_y should be batched (if target is " \
                                  "scalar it should have shape (num_samples, 1))"
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


def next_frame_and_done_data_params(num_boxes_range=(1,2,3,4)):
  return dict(num_boxes_range=num_boxes_range)


def generate_next_frame_and_done_data(env_kwargs, seed, n_trajectories=100,
                                      trajectory_len=40, clone_done=100):
  num_boxes_range = next_frame_and_done_data_params()["num_boxes_range"]
  if num_boxes_range is None:
    print("num_boxes_range", num_boxes_range)
    num_boxes_range = [env_kwargs["num_boxes"]]
  env_kwargs = deepcopy(env_kwargs)
  np.random.seed(seed)
  env_kwargs["num_boxes"] = num_boxes_range[np.random.randint(len(num_boxes_range))]

  render_env = SokobanEnv(**env_kwargs)
  render_env.seed(seed)
  trajectories = list()  # [(observations, actions, done), ...]
  for i in range(n_trajectories):
    render_env.reset()
    state = render_env.clone_full_state()
    # generate random path
    trajectories.append(random_trajectory(state, render_env, trajectory_len))

  # parse trajectories into arrays
  data_x = list()
  data_y_next_frame = list()
  data_y_if_done = list()

  for obs, actions, done in trajectories:
    data_x.extend([
      image_with_embedded_action(ob, action, render_env.action_space.n)
      for ob, action in zip(obs[:-1], actions)
    ])
    data_y_next_frame.extend([ob for ob in obs[1:]])
    data_y_if_done.extend([False] * (len(actions) - 1) + [done])

    if done and (clone_done > 1):
      data_x.extend(
        [data_x[-1].copy() for _ in range(clone_done)]
      )
      data_y_next_frame.extend(
        [data_y_next_frame[-1].copy() for _ in range(clone_done)]
      )
      data_y_if_done.extend(
        [data_y_if_done[-1] for _ in range(clone_done)]
      )


  data_x = np.array(data_x)
  data_y = {
    Target.NEXT_FRAME.value: np.array(data_y_next_frame),
    "if_done": np.array(data_y_if_done).reshape((-1,1)).astype(int),
  }
  return data_x, data_y, {}


def load_shard(shard, target, data_files_prefix, env_kwargs,
               sample_data=False, max_samples_per_board=None,
               seed=None):
  if target == Target.NEXT_FRAME_AND_DONE:
    # messy, as all code for Target.NEXT_FRAME_AND_DONE
    data = generate_next_frame_and_done_data(env_kwargs, seed=seed)
    return data
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
