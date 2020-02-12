# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial
import gzip
import os

import numpy as np
import tensorflow as tf

import gin.tf

# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.
from learning_and_planning.mcts.mask_game_processors import DummyGameMaskProcessor

ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

# A prefix that can not collide with variable names for checkpoint files.
STORE_FILENAME_PREFIX = '$store$_'

# This constant determines how many iterations a checkpoint is kept for.
MAX_SAMPLE_ATTEMPTS = 1000

@gin.configurable
class PoloOutOfGraphReplayBuffer(object):
    """A simple out-of-graph Replay Buffer.

  Stores transitions, state, action, reward, next_state, terminal (and any
  extra contents specified) in a circular buffer and provides a uniform
  transition sampling function.

  When the states consist of stacks of observations storing the states is
  inefficient. This class writes observations and constructs the stacked states
  at sample time.

  Attributes:
    add_count: int, counter of how many transitions have been added (including
      the blank ones at the beginning of an episode).
  """

    def __init__(self,
                 state_shape,
                 observation_shape,
                 replay_capacity,
                 batch_size,
                 checkpoint_duration=10,
                 update_horizon=1,
                 max_sample_attempts=MAX_SAMPLE_ATTEMPTS,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 state_dtype=np.uint8,
                 renderer=None,
                 solved_unsolved_ratio=None,
                 mask_game_processor_fn=DummyGameMaskProcessor,
                 store_states=True):
        """Initializes OutOfGraphReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      stack_size: int, number of frames to use in state stack.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      store_states: either to store states or render them into observations (in
        both cases assumes that states would be passed to add)

    Raises:
      ValueError: If replay_capacity is too small to hold at least one
        transition.
    """
        assert isinstance(observation_shape, tuple)
        assert isinstance(state_shape, tuple)

        tf.logging.info(
            'Creating a %s replay memory with the following parameters:',
            self.__class__.__name__)
        tf.logging.info('\t observation_shape: %s', str(observation_shape))
        tf.logging.info('\t observation_dtype: %s', str(observation_dtype))
        tf.logging.info('\t replay_capacity: %d', replay_capacity)
        tf.logging.info('\t batch_size: %d', batch_size)
        tf.logging.info('\t update_horizon: %d', update_horizon)

        self.solved_unsolved_ratio = solved_unsolved_ratio
        assert solved_unsolved_ratio is not None, "Solved_unsolved_ratio must be set"
        assert 0 <= self.solved_unsolved_ratio <= 1 or self.solved_unsolved_ratio == -1, \
            "Ratio should be in [0,1] or -1, which is to denote to be derived from data"
        self._state_shape = state_shape
        self._observation_shape = observation_shape
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._checkpoint_duration = checkpoint_duration
        self._update_horizon = update_horizon
        self._state_dtype = state_dtype
        self._observation_dtype = observation_dtype
        self._max_sample_attempts = max_sample_attempts
        self._renderer = renderer
        if extra_storage_types:
            self._extra_storage_types = extra_storage_types
        else:
            self._extra_storage_types = []
        self._store = {'solved': [], 'unsolved': []}

        self.add_count = {'solved': np.array(0), 'unsolved': np.array(0)}
        self.mask_game_processor = mask_game_processor_fn()
        self.store_states = store_states
        if self.store_states:
            assert renderer is not None, "You need to pass renderer to store observations"

    @property
    def total_count(self):
        return sum(self.add_count.values())


    def add(self, game, *args, **kwargs):
        """Adds a transition to the replay memory.

        Game is list of tuples (state, value, action)
        """
        # self._check_add_types(game, *args)
        if not self.store_states:
            # Parse game, replace states with observations
            for i, transition in enumerate(game):
                observation = self._renderer([transition[0]])[0]
                new_transition = (observation,) + transition[1:]
                game[i] = new_transition

        self._add(game, *args, **kwargs)


    def _add(self, *args, **kwargs):
        """Internal add method to add to the storage arrays.

    Args:
      *args: All the elements in a transition.
    """
        assert 'solved' in kwargs, "Need to specify if the game was solved or not"

        key = 'solved' if kwargs['solved'] else 'unsolved'
        game = args[0]
        game = self.mask_game_processor.process_game(game)
        if len(game) == 0:
            return
        if self.is_full(key):
            cursor = self.cursor(key)
            self._store[key][cursor] = game
        else:
            self._store[key].append(game)

        self.add_count[key] += 1


    def is_empty(self, key):
        """Is the Replay Buffer empty?"""
        return self.add_count[key] == 0

    def is_full(self, key):
        """Is the Replay Buffer full?"""
        return self.add_count[key] >= self._replay_capacity

    # INFO(lukasz): in dopamine different functionality
    def cursor(self, key):
        """Index to the location where the next transition will be written."""
        return self.add_count[key] % self._replay_capacity


    def _create_batch_arrays(self, batch_size, num_steps=1):
        """Create a tuple of arrays with the type of get_transition_elements.

    When using the WrappedReplayBuffer with staging enabled it is important to
    create new arrays every sample because StaginArea keeps a pointer to the
    returned arrays.

    Args:
      batch_size: (int) number of transitions returned. If None the default
        batch_size will be used.

    Returns:
      Tuple of np.arrays with the shape and type of get_transition_elements.
    """
        assert num_steps==1, "Still this needs to be merged beca3f3717eec6cc09eda472551d67f3842ec1b4"
        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = {'solved': [], 'unsolved': []}

        for key in ['solved', 'unsolved']:
            for element in transition_elements[key]:
                batch_arrays[key].append(np.zeros(element.shape, dtype=element.type))
        return {key: tuple(value) for key, value in batch_arrays.items()}

    def _get_bs_structure(self, batch_size):
        solved_unsolved_ratio_ = self.solved_unsolved_ratio
        if self.solved_unsolved_ratio == -1:  # -1 => derive from data, +0.01 is to avoid dividing by 0
            solved_unsolved_ratio_ = float(self.add_count['solved'])/\
                                     (float(self.add_count['solved']) + float(self.add_count['unsolved'])+0.01)

        exists_solved = not self.is_empty('solved')
        exists_unsolved = not self.is_empty('unsolved')

        bs_solved_ = int(batch_size * solved_unsolved_ratio_)

        bs_solved = bs_solved_ if exists_solved == exists_unsolved else batch_size * int(exists_solved)
        bs_unsolved = batch_size - bs_solved

        bs = {'solved': bs_solved, 'unsolved': bs_unsolved}
        return bs


    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.

    Args:
      batch_size: int, number of indices returned.

    Returns:
      list of ints, a batch of valid indices sampled uniformly.

    Raises:
      RuntimeError: If the batch was not constructed after maximum number of
        tries.
    """
        indices = {}

        bs = self._get_bs_structure(batch_size)
        for key in ['solved', 'unsolved']:
            max_id = len(self._store[key])
            indices[key] = np.random.choice(max_id, bs[key]) if max_id > 0 else np.array([])

        return indices

    def sample_transition_batch(self, batch_size=None, indices=None,
                                transition_indices=None, num_steps=0):
        """Returns a batch of transitions (including any extra contents).
    If get_transition_elements has been overridden and defines elements not
    stored in self._store, an empty array will be returned and it will be
    left to the child class to fill it. For example, for the child class
    OutOfGraphPrioritizedReplayBuffer, the contents of the
    sampling_probabilities are stored separately in a sum tree.
    When the transition is terminal next_state_batch has undefined contents.
    NOTE: This transition contains the indices of the sampled elements. These
    are only valid during the call to sample_transition_batch, i.e. they may
    be used by subclasses of this replay buffer but may point to different data
    as soon as sampling is done.
    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.
    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().
    Raises:
      ValueError: If an element to be sampled is missing from the replay buffer.
    """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
            # assert len(indices['solved']) + len(indices['unsolved']) == 2 * (batch_size // 2)

        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = self._create_batch_arrays(batch_size)
        for key in ['solved', 'unsolved']:
            for batch_element, index in enumerate(indices[key]):
                assert len(transition_elements[key]) == len(batch_arrays[key])
                game = self._store[key][index]  # game is a list of state-obs-values (np.arrays)
                idx = np.random.choice(range(len(game)))
                state_value = game[idx]

                for element_array, element in zip(batch_arrays[key], transition_elements[key]):

                    # TODO: below hardcoded 0, 1, 2, which might break types
                    # 0 - state or observation (depending on self.store_states)
                    if element.name == 'observation':
                        state_or_observation = state_value[0]
                        if self.store_states:
                            observation = self._renderer([state_or_observation])[0]
                        else:
                            observation = state_or_observation
                        element_array[batch_element] = observation
                    elif element.name == 'value':
                        element_array[batch_element] = state_value[1]
                    elif element.name == 'action':
                        element_array[batch_element] = state_value[2]
                    elif element.name == 'mask':
                        element_array[batch_element] = state_value[3]
                    # We assume the other elements are filled in by the subclass.

        # each batch_arrays is a 'tuple': (batch/2, states), (batch/2, obs), (batch/2, values)
        result = tuple(np.concatenate([entry_solved, entry_unsolved], axis=0)
                       for entry_solved, entry_unsolved in zip(batch_arrays['solved'], batch_arrays['unsolved']))

        return result

    def get_transition_elements(self, batch_size=None, num_steps=0):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size
        bs = self._get_bs_structure(batch_size)

        transition_elements = {}
        for key in ['solved', 'unsolved']:
            transition_elements[key] = [
                ReplayElement('observation', (bs[key],) + self._observation_shape, self._observation_dtype),
                ReplayElement('value', (bs[key],), np.float32),
                ReplayElement('action', (bs[key],), np.uint8),
                ReplayElement('mask', (bs[key], ) + self.mask_game_processor.mask_size, np.float32 )
            ]
            for element in self._extra_storage_types:
                transition_elements[key].append(
                    ReplayElement(element.name, (bs[key],) + tuple(element.shape),
                                  element.type))
        return transition_elements

    def _generate_filename(self, checkpoint_dir, name, suffix):
        return os.path.join(checkpoint_dir, '{}_ckpt.{}.gz'.format(name, suffix))

    def save(self, checkpoint_dir, iteration_number):
        """Save the OutOfGraphReplayBuffer attributes into a file.

    This method will save all the replay buffer's state in a single file.

    Args:
      checkpoint_dir: str, the directory where numpy checkpoint files should be
        saved.
      iteration_number: int, iteration_number to use as a suffix in naming
        numpy checkpoint files.
    """
        if not tf.gfile.Exists(checkpoint_dir):
            return

        filename = self._generate_filename(checkpoint_dir, "games", iteration_number)
        with tf.gfile.Open(filename, 'wb') as f:
            with gzip.GzipFile(fileobj=f) as outfile:
                # Currently this is very inefficient, but that's ok since we
                # only need it for debug purposes.
                np.save(outfile, self._store, allow_pickle=True)

            stale_iteration_number = (
                iteration_number - self._checkpoint_duration
            )
            if stale_iteration_number >= 0:
                stale_filename = self._generate_filename(checkpoint_dir, "games",
                                                         stale_iteration_number)
                try:
                    tf.gfile.Remove(stale_filename)
                except tf.errors.NotFoundError:
                    pass


    def load(self, checkpoint_dir, suffix):
        """Restores the object from bundle_dictionary and numpy checkpoints.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      suffix: str, the suffix to use in numpy checkpoint files.

    Raises:
      NotFoundError: If not all expected files are found in directory.
    """
        filename = self._generate_filename(checkpoint_dir, "games", suffix)
        if not tf.gfile.Exists(filename):
            raise tf.errors.NotFoundError(None, None,
                                          'Missing file: {}'.format(filename))
        filename = self._generate_filename(checkpoint_dir, "games", suffix)
        with tf.gfile.Open(filename, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as infile:
                self._store = np.load(infile, allow_pickle=True)

        # save_elements = self._return_checkpointable_elements()
        ## We will first make sure we have all the necessary files available to avoid
        ## loading a partially-specified (i.e. corrupted) replay buffer.
        # for attr in save_elements:
        #  filename = self._generate_filename(checkpoint_dir, attr, suffix)
        #  if not tf.gfile.Exists(filename):
        #    raise tf.errors.NotFoundError(None, None,
        #                                  'Missing file: {}'.format(filename))
        ## If we've reached this point then we have verified that all expected files
        ## are available.
        # for attr in save_elements:
        #  filename = self._generate_filename(checkpoint_dir, attr, suffix)
        #  with tf.gfile.Open(filename, 'rb') as f:
        #    with gzip.GzipFile(fileobj=f) as infile:
        #      if attr.startswith(STORE_FILENAME_PREFIX):
        #        array_name = attr[len(STORE_FILENAME_PREFIX):]
        #        self._store[array_name] = np.load(infile, allow_pickle=False)
        #      elif isinstance(self.__dict__[attr], np.ndarray):
        #        self.__dict__[attr] = np.load(infile, allow_pickle=False)
        #      else:
        #        self.__dict__[attr] = pickle.load(infile)


@gin.configurable() #blacklist=['observation_shape', 'state_shape', 'update_horizon'])
class PoloWrappedReplayBuffer(object):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
      To add a transition:  call the add function.

      To sample a batch:    Construct operations that depend on any of the
                            tensors is the transition dictionary. Every sess.run
                            that requires any of these tensors will sample a new
                            transition.
    """

    def __init__(self,
                 state_shape,
                 observation_shape,
                 use_staging=True,
                 replay_capacity=1000000,
                 batch_size=32,
                 checkpoint_duration=10,
                 update_horizon=1,
                 wrapped_memory=None,
                 max_sample_attempts=MAX_SAMPLE_ATTEMPTS,
                 extra_storage_types=None,
                 observation_dtype=np.uint8,
                 state_dtype=np.uint8,
                 renderer=None,
                 num_steps=0):

        """Initializes WrappedReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      wrapped_memory: The 'inner' memory data structure. If None,
        it creates the standard DQN replay memory.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      extra_storage_types: list of ReplayElements defining the type of the extra
        contents that will be stored and returned by sample_transition_batch.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.

    Raises:
      ValueError: If update_horizon is not positive.
      ValueError: If discount factor is not in [0, 1].
    """
        if replay_capacity < update_horizon + 1:
            raise ValueError(
                'Update horizon ({}) should be significantly smaller '
                'than replay capacity ({}).'.format(update_horizon, replay_capacity))
        if not update_horizon >= 1:
            raise ValueError('Update horizon must be positive.')

        self.batch_size = batch_size
        self.num_steps = num_steps

        # Mainly used to allow subclasses to pass self.memory.
        if wrapped_memory is not None:
            self.memory = wrapped_memory
        else:
            self.memory = PoloOutOfGraphReplayBuffer(state_shape,
                observation_shape, replay_capacity, batch_size,
                checkpoint_duration,
                update_horizon, max_sample_attempts,
                observation_dtype=observation_dtype,
                extra_storage_types=extra_storage_types,
                renderer=renderer, state_dtype=state_dtype)

        self.create_sampling_ops(use_staging, num_steps)

    def add(self, *args, **kwargs):
        """Adds a transition to the replay memory.

    Since the next_observation in the transition will be the observation added
    next there is no need to pass it.

    If the replay memory is at capacity the oldest transition will be discarded.

    Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: A uint8 acting as a boolean indicating whether the transition
                was terminal (1) or not (0).
      *args: extra contents with shapes and dtypes according to
        extra_storage_types.
    """
        self.memory.add(*args, **kwargs)

    def create_sampling_ops(self, use_staging, num_steps):
        """Creates the ops necessary to sample from the replay buffer.

    Creates the transition dictionary containing the sampling tensors.

    Args:
      use_staging: bool, when True it would use a staging area to prefetch
        the next sampling batch.
    """
        if num_steps:
            shape_pref = (num_steps,)
        else:
            shape_pref = ()

        with tf.name_scope('sample_replay'):
            with tf.device('/cpu:*'):
                transition = self.memory.get_transition_elements()
                transition_type_s = transition['solved']
                transition_type_u = transition['unsolved']
                transition_type = [ReplayElement(tts.name, shape_pref + (ttu.shape[0]+tts.shape[0], *tts.shape[1:]), tts.type)
                                  for tts, ttu in zip(transition_type_s, transition_type_u)]
                transition_tensors = tf.py_func(
                    partial(
                        self.memory.sample_transition_batch, num_steps=num_steps
                    ), [],
                    [return_entry.type for return_entry in transition_type],
                    name='replay_sample_py_func')
                self._set_transition_shape(transition_tensors, transition_type)
                if use_staging:
                    transition_tensors = self._set_up_staging(transition_tensors)
                    self._set_transition_shape(transition_tensors, transition_type)

                # Unpack sample transition into member variables.
                self.unpack_transition(transition_tensors, transition_type)

    def _set_transition_shape(self, transition, transition_type):
        """Set shape for each element in the transition.

    Args:
      transition: tuple of tf.Tensors.
      transition_type: tuple of ReplayElements descriving the shapes of the
        respective tensors.
    """
        for element, element_type in zip(transition, transition_type):
            element.set_shape(element_type.shape)

    def _set_up_staging(self, transition):
        """Sets up staging ops for prefetching the next transition.

    This allows us to hide the py_func latency. To do so we use a staging area
    to pre-fetch the next batch of transitions.

    Args:
      transition: tuple of tf.Tensors with shape
        memory.get_transition_elements().

    Returns:
      prefetched_transition: tuple of tf.Tensors with shape
        memory.get_transition_elements() that have been previously prefetched.
    """
        transition_type = self.memory.get_transition_elements()['solved']  # doesnt matter if 'solved' or 'unsolved'

        # Create the staging area in CPU.
        prefetch_area = tf.contrib.staging.StagingArea(
            [shape_with_type.type for shape_with_type in transition_type])

        # Store prefetch op for tests, but keep it private -- users should not be
        # calling _prefetch_batch.
        self._prefetch_batch = prefetch_area.put(transition)
        initial_prefetch = tf.cond(
            tf.equal(prefetch_area.size(), 0),
            lambda: prefetch_area.put(transition), tf.no_op)

        # Every time a transition is sampled self.prefetch_batch will be
        # called. If the staging area is empty, two put ops will be called.
        with tf.control_dependencies([self._prefetch_batch, initial_prefetch]):
            prefetched_transition = prefetch_area.get()

        return prefetched_transition

    def unpack_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.

    Args:
      transition_tensors: tuple of tf.Tensors.
      transition_type: tuple of ReplayElements matching transition_tensors.
    """
        self.transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.transition[element_type.name] = element

        # TODO(bellemare): These are legacy and should probably be removed in
        # future versions.
        # self.states = self.transition['state']
        self.observations = self.transition['observation']
        self.values = self.transition['value']
        self.actions = self.transition['action']
        self.masks = self.transition['mask']

        # self.states = self.transition['state']
        # self.actions = self.transition['action']
        # self.rewards = self.transition['reward']
        # self.next_states = self.transition['next_state']
        # self.terminals = self.transition['terminal']
        # self.indices = self.transition['indices']

    def save(self, checkpoint_dir, iteration_number):
        """Save the underlying replay buffer's contents in a file.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      iteration_number: int, the iteration_number to use as a suffix in naming
        numpy checkpoint files.
    """
        self.memory.save(checkpoint_dir, iteration_number)

    def load(self, checkpoint_dir, suffix):
        """Loads the replay buffer's state from a saved file.

    Args:
      checkpoint_dir: str, the directory where to read the numpy checkpointed
        files from.
      suffix: str, the suffix to use in numpy checkpoint files.
    """
        self.memory.load(checkpoint_dir, suffix)
