import copy
import enum
import random

import gym
import numba
import numpy as np
from gym.spaces import Discrete, Box
from munch import Munch

from gym_sokoban.envs.sokoban_env import SokobanEnv
import pkg_resources
from PIL import Image
import gin

RENDERING_MODES = ['one_hot', 'rgb_array', 'tiny_rgb_array']


@gin.configurable
class SokobanEnvFast(gym.Env):
    metadata = {
        'render.modes': RENDERING_MODES
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=np.inf,
                 num_boxes=4,
                 num_gen_steps=None,
                 mode='one_hot',
                 fast_state_eq=False,
                 penalty_for_step=-0.1,
                 # penalty_box_off_target=-1,
                 reward_box_on_target=1,
                 reward_finished=10,
                 seed=None,
                 load_boards_from_file=None,
                 load_boards_lazy=True
                 ):
        self._seed = seed
        self.mode = mode
        self.num_gen_steps = num_gen_steps
        self.dim_room = dim_room
        self.max_steps = max_steps
        self.num_boxes = num_boxes

        # Penalties and Rewards
        self.penalty_for_step = penalty_for_step
        # self.penalty_box_off_target = penalty_box_off_target
        self.reward_box_on_target = reward_box_on_target
        self.reward_finished = reward_finished
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(self.dim_room[0], self.dim_room[1], 7), dtype=np.uint8)
        self.state_space = self.observation_space  # state == observation

        self._internal_state = None
        self.fast_state_eq = fast_state_eq

        self._slave_env = SokobanEnv(dim_room=dim_room,
                                     max_steps=max_steps,
                                     num_boxes=num_boxes,
                                     num_gen_steps=num_gen_steps,
                                     mode=mode,
                                     seed=self._seed,
                                     verbose=False)

        self._surfaces = load_surfaces()
        self.initial_internal_state_hash = None
        self.load_boards_from_file = load_boards_from_file
        self.boards_from_file = None
        if not load_boards_lazy:
            self.boards_from_file = np.load(self.load_boards_from_file)


    def seed(self, seed=None):
        self._seed = self._slave_env.seed(seed)[0]
        return self._seed

    @property
    def init_kwargs(self):
        kwargs = {
            attr: getattr(self, attr)
            for attr in ('dim_room', 'max_steps', 'num_boxes', 'num_gen_steps', 'mode')
        }
        kwargs["seed"] = self._seed
        return kwargs

    def step(self, action):
        raw_state, rew, done = step(self._internal_state.get_raw(), action,
                                    self.penalty_for_step,
                                    self.reward_box_on_target,
                                    self.reward_finished)
        self._internal_state = HashableState(*raw_state, fast_eq=self.fast_state_eq)
        return self._internal_state.one_hot, rew, done, {"solved": done}

    def reset(self):
        if self.load_boards_from_file:
            if self.boards_from_file is None: # the case of lazy loading
                self.boards_from_file = np.load(self.load_boards_from_file)
            index = random.randint(0, len(self.boards_from_file)-1)
            one_hot = self.boards_from_file[index]
        else:
            self._slave_env.reset()
            one_hot = self._slave_env.render(mode="one_hot")
        self.restore_full_state_from_np_array_version(one_hot)
        self.initial_internal_state_hash = hash(self._internal_state)
        return self._internal_state.one_hot

    def render(self, mode='one_hot'):
        assert mode in RENDERING_MODES, f"Only {RENDERING_MODES} are supported, not {mode}"
        if mode == 'one_hot':
            return self._internal_state.one_hot
        render_surfaces = None
        if mode == 'rgb_array':
            render_surfaces = self._surfaces['16x16pixels']
        if mode == 'tiny_rgb_array':
            render_surfaces = self._surfaces['8x8pixels']

        size_x = self._internal_state.one_hot.shape[0]*render_surfaces.shape[1]
        size_y = self._internal_state.one_hot.shape[1]*render_surfaces.shape[2]

        res = np.tensordot(self._internal_state.one_hot, render_surfaces, (-1, 0))
        res = np.transpose(res, (0, 2, 1, 3, 4))
        res = np.reshape(res, (size_x, size_y, 3))
        return res

    def clone_full_state(self):
        internal_state = self._internal_state
        internal_state._initial_state_hash = self.initial_internal_state_hash
        return internal_state

    def restore_full_state(self, state):
        self._internal_state = state
        self.initial_internal_state_hash = state._initial_state_hash

    def restore_full_state_from_np_array_version(self, state_np, quick=False):
        if (state_np > 255).any() or (state_np < 0).any():
            raise ValueError(f"restore_full_state_from_np_array_version() got "
                             f"data out of range 0-255 {state_np}")
        if quick:  # PM: This is used when rendering is only needed
            agent_pos = None
            unmatched_boxes = None
        else:
            shape = state_np.shape[:2]
            agent_pos = np.unravel_index(np.argmax(state_np[..., FieldStates.player] +
                                                   state_np[..., FieldStates.player_target]), dims=shape)
            unmatched_boxes = int(np.sum(state_np[..., FieldStates.box]))
        self._internal_state = HashableState(state_np, agent_pos, unmatched_boxes, fast_eq=self.fast_state_eq)


class FieldStates(enum.IntEnum):
    wall = 0
    empty = 1
    target = 2
    box_target = 3
    box = 4
    player = 5
    player_target = 6


@numba.jit(nopython=True)
def step(state, action, penalty_for_step, reward_box_on_target, reward_finished):
    # Copy to be supported by numba. Possibly can be done better
    # wall = 0
    empty = 1
    target = 2
    box_target = 3
    box = 4
    player = 5
    player_target = 6

    delta_x, delta_y = None, None
    if action == 0:
        delta_x, delta_y = -1, 0
    elif action == 1:
        delta_x, delta_y = 1, 0
    elif action == 2:
        delta_x, delta_y = 0, -1
    elif action == 3:
        delta_x, delta_y = 0, 1

    one_hot, agent_pos, unmatched_boxes = state

    arena = np.zeros(shape=(3,), dtype=np.uint8)
    for i in range(3):
        index_x = agent_pos[0] + i * delta_x
        index_y = agent_pos[1] + i * delta_y
        if index_x < one_hot.shape[0] and index_y < one_hot.shape[0]:
            arena[i] = np.where(one_hot[index_x, index_y, :] == 1)[0][0]

    new_unmatched_boxes_ = unmatched_boxes
    new_agent_pos = agent_pos
    new_arena = np.copy(arena)

    box_moves = (arena[1] == box or arena[1] == box_target) and \
                (arena[2] == empty or arena[2] == 2)

    agent_moves = arena[1] == empty or arena[1] == target or box_moves

    if agent_moves:
        targets = (arena == target).astype(np.int8) + \
                  (arena == box_target).astype(np.int8) + \
                  (arena == player_target).astype(np.int8)
        if box_moves:
            last_field = box - 2 * targets[2]  # Weirdness due to inconsistent target non-target
        else:
            last_field = arena[2] - targets[2]

        new_arena = np.array([empty, player, last_field]).astype(np.uint8) + targets.astype(np.uint8)
        new_agent_pos = (agent_pos[0] + delta_x, agent_pos[1] + delta_y)

        if box_moves:
            new_unmatched_boxes_ = int(unmatched_boxes - (targets[2] - targets[1]))

    new_one_hot = np.copy(one_hot)
    for i in range(3):
        index_x = agent_pos[0] + i * delta_x
        index_y = agent_pos[1] + i * delta_y
        if index_x < one_hot.shape[0] and index_y < one_hot.shape[0]:
            one_hot_field = np.zeros(shape=7)
            one_hot_field[new_arena[i]] = 1
            new_one_hot[index_x, index_y, :] = one_hot_field

    done = (new_unmatched_boxes_ == 0)
    reward = penalty_for_step - reward_box_on_target * (float(new_unmatched_boxes_) - float(unmatched_boxes))
    if done:
        reward += reward_finished

    new_state = (new_one_hot, new_agent_pos, new_unmatched_boxes_)

    return new_state, reward, done


class HashableState:
    state = np.random.get_state()
    np.random.seed(0)
    hash_key = np.random.normal(size=10000)
    np.random.set_state(state)

    def __init__(self, one_hot, agent_pos, unmached_boxes, fast_eq=False):
        self.one_hot = one_hot
        self.agent_pos = agent_pos
        self.unmached_boxes = unmached_boxes
        self._hash = None
        self.fast_eq = fast_eq
        self._initial_state_hash = None

    def __iter__(self):
        yield from [self.one_hot, self.agent_pos, self.unmached_boxes]

    def __hash__(self):
        if self._hash is None:
            flat_np = self.one_hot.flatten()
            self._hash = int(np.dot(flat_np, HashableState.hash_key[:len(flat_np)]) * 10e8)
        return self._hash

    def __eq__(self, other):
        if self.fast_eq:
            return hash(self) == hash(other)  # This is a conscious decision to speed up.
        else:
            return np.array_equal(self.one_hot, other.one_hot)

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_raw(self):
        return self.one_hot, self.agent_pos, self.unmached_boxes

    def get_np_array_version(self):
        return self.one_hot


def load_surfaces():

    # Necessarily keep the same order as in FieldStates
    assets_file_name = ['wall.png', 'floor.png', 'box_target.png', 'box_on_target.png',
                        'box.png', 'player.png', 'player_on_target.png']
    sizes = ['8x8pixels', '16x16pixels']

    resource_package = __name__
    surfaces = {}
    for size in sizes:
        surfaces[size] = []
        for asset_file_name in assets_file_name:
            asset_path = pkg_resources.resource_filename(resource_package, '/'.join(('surface', size, asset_file_name)))
            asset_np_array = np.array(Image.open(asset_path))
            surfaces[size].append(asset_np_array)

        surfaces[size] = np.stack(surfaces[size])

    return surfaces


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes, dtype=np.uint8)[a.reshape(-1)]).reshape(a.shape + (num_classes,))


@gin.configurable
def sokoban_hindsight(history, solved, intensity, include_original_board=False):
    # PM This function is unsafe as it alters internal structures
    if solved or random.random() > intensity:
        return history, {'hindsight_solved': solved}

    parameter_env = SokobanEnvFast()  #This is just to harvest parameters!
    penalty_for_step = parameter_env.penalty_for_step
    reward_box_on_target = parameter_env.reward_box_on_target
    reward_finished = parameter_env.reward_finished
    history = history[:random.randint(2, len(history))]
    final_state_np = history[-1][0].state.one_hot

    targets_np = final_state_np[..., FieldStates.target] + \
                 final_state_np[..., FieldStates.player_target] + \
                 final_state_np[..., FieldStates.box_target]

    targets_indices = np.argwhere(targets_np>0)

    boxes_np = final_state_np[..., FieldStates.box] + final_state_np[..., FieldStates.box_target]

    new_targets_indices = np.argwhere(boxes_np>0)

    previous_unmatched_boxes = None

    new_history = []
    new_rewards = []
    original_state = copy.deepcopy(history[0][0].state)

    for node, action, _ in history:
        state_np = node.state.one_hot
        state_np_collapsed = np.argmax(state_np, axis=2)

        #Remove old targets
        for target_index in targets_indices:
            index = tuple(target_index)
            if state_np_collapsed[index] == FieldStates.target:
                state_np_collapsed[index] = FieldStates.empty
            if state_np_collapsed[index] == FieldStates.box_target:
                state_np_collapsed[index] = FieldStates.box
            if state_np_collapsed[index] == FieldStates.player_target:
                state_np_collapsed[index] = FieldStates.player

        #Put new targets
        for target_index in new_targets_indices:
            index = tuple(target_index)
            if state_np_collapsed[index] == FieldStates.empty:
                state_np_collapsed[index] = FieldStates.target
            if state_np_collapsed[index] == FieldStates.box:
                state_np_collapsed[index] = FieldStates.box_target
            if state_np_collapsed[index] == FieldStates.player:
                state_np_collapsed[index] = FieldStates.player_target

        state_np_new = one_hot(state_np_collapsed, 7)


        new_unmatched_boxes_ = int(np.sum(state_np_new[..., FieldStates.box]))
        done = (new_unmatched_boxes_ == 0)
        if previous_unmatched_boxes is None and done:  # Special case of board solved in the first step
            return history, {'hindsight_solved': solved}

        reward = penalty_for_step
        if previous_unmatched_boxes is not None:
            reward += reward_box_on_target * (float(previous_unmatched_boxes) - float(new_unmatched_boxes_))

        previous_unmatched_boxes = new_unmatched_boxes_

        if done:
            reward += reward_finished

        node.state.one_hot = state_np_new
        new_history.append((node, action))
        new_rewards.append(reward)
        if done:
            break

    new_rewards = new_rewards[1:]
    new_history_ = [(node, action, reward) for (node, action), reward in zip(new_history, new_rewards)]
    if include_original_board:
        original_board_node = Munch(state=original_state)
        new_history_ = [(original_board_node, 0, 0)] \
                       + new_history_ # action and rewards are fake and should not be used.

    return new_history_, {"hindsight_solved": True}
