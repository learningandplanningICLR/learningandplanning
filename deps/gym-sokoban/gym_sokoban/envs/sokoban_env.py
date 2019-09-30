import copy
import random

import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box

from gym_sokoban import logger
from .room_utils import SokobanRoomGenerator
from .render_utils import room_to_rgb, room_to_tiny_world_rgb, room_to_one_hot, room_to_binary_map
import numpy as np
from matplotlib.pyplot import imread
import pkg_resources

import traceback

# modes meaning:
#   binary_map - observations of shape (room_size[0], room_size[1], 4) with
#       binary values. Channels represents if given part of room
#           0 - is interior (without wall)
#           1 - consist target
#           2 - consist box
#           3 - consist player
#       e.g. pixel value [1, 1, 1, 0] indicates target tile with a box.
#       Room state can be converted to binary_map and back using
#       render_utils.{room_to_binary_map,binary_map_to_room}.
RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'one_hot', 'one_hot_flatten', "binary_map"]

class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': RENDERING_MODES
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=np.inf,  # 120
                 num_boxes=4,
                 num_gen_steps=None,
                 game_mode="NoAlice",  # Alice, NoAlice, Magnetic
                 only_push_actions=True,
                 max_distinct_rooms=np.inf,  # INFO: if finite, clone and restore do not reflect num_env_steps
                 mode='rgb_array',
                 penalty_pull_action=0.,  # used only in "Magnetic" mode
                 seed=None,
                 curriculum=300,  # depth of DFS in reverse_play
                 reward_shaping='dense',  # dense or sparse
                 verbose=True
                 ):
        assert game_mode in ["NoAlice", "Alice", "Magnetic"], "Incorrect game format!"
        assert reward_shaping in ['dense', 'sparse'], 'Incorrect reward shaping mode!'
        logger.info('Creating SokobanEnv')
        logger.info('Game format: {}'.format(game_mode))

        self.game_mode = game_mode
        self.verbose = verbose

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps is None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0
        self.num_env_steps = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.penalty_pull_action = penalty_pull_action
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_shaping = reward_shaping
        self.max_distinct_rooms = np.inf if max_distinct_rooms == -1 else max_distinct_rooms
        # TODO: handle it better
        assert self.max_distinct_rooms < 50000 or self.max_distinct_rooms == np.inf, \
            "Setting high self.max_distinct_rooms creates potential memory leak. " \
            "Uncomment if you know what your are doing"
        self.generated_rooms = []

        self.scale = 1
        self._create_surface()

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        num_actions = len(ACTION_LOOKUP) if self.game_mode == "Magnetic" else len(ACTION_LOOKUP) // 2
        self.action_space = Discrete(num_actions)

        self.observation_space = self.create_observation_space(mode=mode)

        self.current_room_id = 0

        self.mode = mode
        self.curriculum = curriculum

        self.seed(seed)

    @property
    def init_kwargs(self):
        return {
            attr: getattr(self, attr)
            for attr in (
                'dim_room', 'max_steps', 'num_boxes', 'num_gen_steps',
                'game_mode', 'max_distinct_rooms', 'mode', 'penalty_pull_action',
                'reward_shaping', 'verbose'
            )
        }

    def create_observation_space(self, mode):
        if mode in self.surfaces:
            img = self.surfaces[mode][0]
            img_shape = (1, 1, 3) if isinstance(img, list) else img.shape
            screen_height, screen_width = (self.dim_room[0] * img_shape[0], self.dim_room[1] * img_shape[1])
            return Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        elif mode == 'one_hot':
            return Box(low=0, high=1, shape=(self.dim_room[0], self.dim_room[1], 7), dtype=np.uint8)
        elif mode == 'one_hot_flatten':
            return Box(low=0, high=1, shape=(self.dim_room[0] * self.dim_room[1] * 7,), dtype=np.uint8)
        elif mode == 'binary_map':
            return Box(low=0, high=1, shape=(self.dim_room[0], self.dim_room[1], 4), dtype=np.uint8)
        else:
            raise RuntimeError('unknown mode {}'.format(mode))


    def seed(self, seed=None):
        # INFO(): what guarantees do we want from this?
        # We want the rooms generated to be the same, even if the actions taken in the env are different.

        _, self._seed = seeding.np_random(seed)
        self.other_rng = random.Random(self._seed)

        self.room_generator = SokobanRoomGenerator(seed=self._seed,
                                                   game_mode=self.game_mode,
                                                   verbose=self.verbose)
        self.current_seed = None

        return [self._seed]

    def step(self, action):
        assert action in ACTION_LOOKUP, "Illegal action {}".format(action)
        if not hasattr(self, 'room_generator'):
            raise RuntimeError('Please call reset() first')

        self.num_env_steps += 1
        #if self.game_mode = "Alice":
        #    self.num_steps_bob
        #    self.num_steps_alice

        self.new_box_position = None
        self.old_box_position = None

        moved_player = False
        moved_box = False
        # All push actions are in the range of [0, 3]
        if action < 4:
            moved_player, moved_box = self._push(action)
        else:
            if self.game_mode != "Magnetic":
                raise RuntimeError("Cannot make 'pull' action if not in 'Magnetic' game mode.")
            moved_player, moved_box = self._push(action, with_pull=True)

        reward = self._calc_reward(action)

        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=self.mode)

        info = {
            'action.name': ACTION_LOOKUP[action],
            'action.moved_player': moved_player,
            'action.moved_box': moved_box,
            'text': ['room_id={}'.format(self.current_room_id)],
            'aux_rewards': {
                'solved': self._check_if_all_boxes_on_target()
            }
        }

        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            #info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()
            info["solved"] = self._check_if_all_boxes_on_target()

        return observation, reward, done, info

    def _push(self, action, with_pull=False):
        """
        INFO: if self.game_mode == "Alice", push is in fact pull.
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :param with_pull: if True, make pull along with push ("Magnetic" mode)
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[action % 4]
        if self.game_mode == "Alice":  # Alice does reverse moves
            new_position = self.player_position - change
        else:
            new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        # INFO: no need to check for negative, since it can be the least -1, which is legal in Python
        if self.game_mode == "Alice":
            new_box_position = self.player_position
        else:
            new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False

        if self.game_mode == "Alice":
            # in this case push is a pull
            old_box_position = self.player_position - change
            can_push_box = self.room_state[old_box_position[0], old_box_position[1]] in [3, 4]
        else:
            can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
            can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]

        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            if self.game_mode == "Alice":
                self.old_box_position = tuple(old_box_position)
            else:
                self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            if self.game_mode == "Alice":
                self.room_state[old_box_position[0], old_box_position[1]] = \
                    self.room_fixed[old_box_position[0], old_box_position[1]]
            else:
                self.room_state[current_position[0], current_position[1]] = \
                    self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            moved_player, moved_box = True, True

        # Try to move if no box to push, available
        else:
            moved_player = self._move(action)
            moved_box = False

        # Pull box
        if with_pull and moved_player:
            if self.game_mode == "Alice":
                raise RuntimeError("Using push with 'with_pull' flag in Alice mode is undefined.")
            pull_content_position = current_position - change
            is_box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            if is_box_next_to_player:
                # Move Box
                moved_box = True
                box_type = 4
                if self.room_fixed[current_position[0], current_position[1]] == 2:
                    box_type = 3
                self.room_state[current_position[0], current_position[1]] = box_type
                self.room_state[pull_content_position[0], pull_content_position[1]] = \
                    self.room_fixed[pull_content_position[0], pull_content_position[1]]

        return moved_player, moved_box

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[action % 4]
        if self.game_mode == "Alice":
            new_position = self.player_position - change
        else:
            new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def set_game_mode(self, game_mode=None):
        self.game_mode = game_mode

    def _calc_reward(self, action=None):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        if self.game_mode == "Alice":
            # reward is a tuple (reward_Bob, reward_Alice)
            reward = (self.num_steps_bob, self.num_steps_alice)

        elif self.reward_shaping == 'sparse':
            game_won = self._check_if_all_boxes_on_target()
            reward = 1 if game_won else 0
        else:  # dense reward
            reward = self.penalty_for_step

            # count boxes off or on the target
            empty_targets = self.room_state == 2
            player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
            total_targets = empty_targets | player_on_target

            current_boxes_on_target = self.num_boxes - \
                                      np.where(total_targets)[0].shape[0]

            # Add the reward if a box is pushed on the target and give a
            # penalty if a box is pushed off the target.
            if current_boxes_on_target > self.boxes_on_target:
                reward += self.reward_box_on_target
            elif current_boxes_on_target < self.boxes_on_target:
                reward += self.penalty_box_off_target

            game_won = self._check_if_all_boxes_on_target()
            if game_won:
                reward += self.reward_finished

            self.boxes_on_target = current_boxes_on_target

        # Additional penalty for pull actions
        if self.game_mode == "Magnetic" and action is not None and action > 3:
            # Penalize pull actions
            reward += self.penalty_pull_action

        return reward

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.num_env_steps >= self.max_steps)

    def reset(self, second_player=False):
        try:
            if len(self.generated_rooms) < self.max_distinct_rooms:
                # INFO: currently room_generator.generate_room does not return meaningful box_mapping
                self.current_seed = self._seed
                self.room_fixed, self.room_state, self.box_mapping, self._seed = self.room_generator.generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    second_player=second_player,
                    curriculum=self.curriculum,
                )
                if self.max_distinct_rooms < np.inf:
                    room_to_save = copy.deepcopy((self.room_fixed, self.room_state, self.box_mapping))
                    self.generated_rooms.append(room_to_save)
            else:
                self.current_room_id = self.other_rng.randint(0, len(self.generated_rooms) - 1)
                self.room_fixed, self.room_state, self.box_mapping \
                    = copy.deepcopy(self.generated_rooms[self.current_room_id])

        except (RuntimeError, RuntimeWarning) as e:
            if self.verbose:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
            return self.reset()

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.boxes_on_target = 0

        starting_observation = self.render(mode=self.mode)
        return starting_observation

    # kwargs could include 'mode' and 'scale'
    def render(self, mode=None, close=None, **kwargs):
        mode = mode if mode is not None else self.mode
        assert mode in RENDERING_MODES, 'Unknown rendering mode {}'.format(mode)

        obs = self.get_image(mode=mode, **kwargs)
        if mode in ['one_hot', 'one_hot_flatten', 'binary_map', 'rgb_array']:
            return obs
        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(obs)
            return self.viewer.isopen
        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, **kwargs):

        if mode == 'one_hot':
            return room_to_one_hot(self.room_state, self.room_fixed)
        elif mode == 'one_hot_flatten':
            return room_to_one_hot(self.room_state, self.room_fixed).flatten()
        elif mode == 'binary_map':
            return room_to_binary_map(self.room_state, self.room_fixed)
        elif mode.startswith('tiny_'):
            surfaces = self.surfaces[mode]
            self.scale = kwargs.get('scale', 1)
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, surfaces=surfaces, scale=self.scale)
        else:
            surfaces = self.surfaces[mode]
            img = room_to_rgb(self.room_state, self.room_fixed, surfaces=surfaces)

        return img

    # INFO: it does not clone self.num_env_steps
    def clone_full_state(self):
        state = np.vstack([self.room_fixed, self.room_state])  # copies
        state = state.flatten()
        state = np.append(state, self._seed)
        return state  # np.array can be np.save'd

    # INFO: it does not restore self.num_env_steps
    def restore_full_state(self, state):
        seed = state[-1]
        state = state[:-1].astype(np.uint8)

        split = len(state) // 2
        dim_room = self.dim_room
        room_fixed = state[:split].reshape((dim_room[0], dim_room[1]))
        room_state = state[split:].reshape((dim_room[0], dim_room[1]))

        self.room_fixed = np.array(room_fixed)  # copy
        self.room_state = np.array(room_state)  # copy
        self._seed = seed

        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target
        boxes_on_target = self.num_boxes - np.where(total_targets)[0].shape[0]
        self.boxes_on_target = boxes_on_target

        player_position = np.where(self.room_state == 5)
        player_position = np.array([player_position[0][0], player_position[1][0]])
        self.player_position = player_position

        # INFO: self.num_env_steps is not restored
        self.num_env_steps = 0

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP

    @staticmethod
    def load_surfaces():
        surfaces = {}

        # Load images, representing the corresponding situation
        resource_package = __name__
        box_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', '8x8pixels', 'box.png')))
        box = imread(box_filename)
        box_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                 '/'.join(('surface', '8x8pixels', 'box_on_target.png')))
        box_on_target = imread(box_on_target_filename)
        box_target_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', '8x8pixels', 'box_target.png')))
        box_target = imread(box_target_filename)
        floor_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', '8x8pixels', 'floor.png')))
        floor = imread(floor_filename)
        player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', '8x8pixels', 'player.png')))
        player = imread(player_filename)
        player_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                    '/'.join(('surface', '8x8pixels', 'player_on_target.png')))
        player_on_target = imread(player_on_target_filename)
        wall_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', '8x8pixels', 'wall.png')))
        wall = imread(wall_filename)
        surfaces["rgb_array"] = [wall, floor, box_target, box_on_target, box, player, player_on_target]

        wall = [0, 0, 0]
        floor = [243, 248, 238]
        box_target = [254, 126, 125]
        box_on_target = [254, 95, 56]
        box = [142, 121, 56]
        player = [160, 212, 56]
        player_on_target = [219, 212, 56]
        surfaces["tiny_rgb_array"] = [wall, floor, box_target, box_on_target, box, player, player_on_target]

        return surfaces

    def _create_surface(self):
        self.surfaces = self.load_surfaces()

        # TODO: add to surfaces other modes
        self._recover = {}
        for mode in ['rgb_array', 'tiny_rgb_array']:  # add 'one_hot', 'one_hot_flatten'
            if mode not in self._recover:
                self._recover[mode] = {}
            #self._mode_shapes[mode] = np.array(self.surfaces[mode][0]).shape
            for idx, img in enumerate(self.surfaces[mode]):
                img = tuple(np.array(img).flatten())
                self._recover[mode][img] = idx

        # TODO: other modes
        self._mode_shapes = {}
        self._mode_shapes['rgb_array'] = (self.dim_room[0] * self.surfaces['rgb_array'][0].shape[0],
                                          self.dim_room[1] * self.surfaces['rgb_array'][1].shape[1],
                                          3)
        self._mode_shapes['tiny_rgb_array'] = (self.dim_room[0] * self.scale,
                                          self.dim_room[1] * self.scale,
                                          3)

    def recover_state(self, obs):
        mode = self.mode
        assert mode in self._recover, "Mode {} not supported".format(mode)
        assert obs.shape == self._mode_shapes[mode], \
            "Dimension of obs {} and mode {} does not match".format(obs.shape, self._mode_shapes[mode])
        sprite_shape = getattr(self.surfaces[mode][0], 'shape', (1, 1))  # when list need this to be (1,1)
        room_state = np.zeros_like(self.room_state)
        for i in range(room_state.shape[0]):
            x_i = i * sprite_shape[0]
            for j in range(room_state.shape[1]):
                y_j = j * sprite_shape[1]
                sprite = obs[x_i:(x_i + sprite_shape[0]), y_j:(y_j + sprite_shape[1]), :]
                sprite = tuple(np.array(sprite).flatten())
                room_state[i, j] = self._recover[mode][sprite]
        room_structure = room_state.copy()
        room_structure[room_structure == 3] = 2  # box on target
        room_structure[room_structure == 6] = 2  # player on target
        room_structure[room_structure == 5] = 1
        room_structure[room_structure == 4] = 1
        room_state[room_state == 6] = 5  # in restore_full_state player on target is not 6
        state = np.vstack([room_structure, room_state])  # copies
        state = state.flatten()
        state = np.append(state, self._seed)
        return state  # np.array can be np.save'd


class OneHotTypes(object):
    wall = 0
    empty = 1
    target = 2
    box_target = 3
    box = 4
    player = 5
    player_target = 6


class OneHotTypeSets(object):
    wall = {OneHotTypes.wall}
    free = {OneHotTypes.empty, OneHotTypes.target}
    target = {
        OneHotTypes.target, OneHotTypes.box_target, OneHotTypes.player_target
    }
    box = {OneHotTypes.box, OneHotTypes.box_target}
    player = {OneHotTypes.player, OneHotTypes.player_target}


ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'pull up',
    5: 'pull down',
    6: 'pull left',
    7: 'pull right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

