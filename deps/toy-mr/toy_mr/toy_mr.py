import os
import re

import cv2
import gym
import pygame
import numpy as np
from gym import Wrapper
from gym.spaces.discrete import Discrete
from gym.spaces import Box

from learning_and_planning.mcts.env_model import HashableNumpyArray

GRID_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)

AGENT_COLOR = (100, 100, 100)
WALL_COLOR = (0, 0, 0)
KEY_COLOR = (218, 165, 32)
DOOR_COLOR = (50, 50, 255)
TRAP_COLOR = (255, 0, 0)
LIVES_COLOR = (0, 255, 0)


# ACTIONS
NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

# Cell Code
WALL_CODE = 1
KEY_CODE = 2
DOOR_CODE = 3
TRAP_CODE = 4
AGENT_CODE = 5
LIVES_CODE = 5


# Orders in which keys are taken and doors are open.
# one_room_shifted.txt and hall_way_shifted.txt are modifications of original
# levels with all rooms coordinates increased by one (to avoid zero coordinates)
KEYS_ORDER = {
    "one_room_shifted.txt": [((1, 1), (1, 8))],
    "four_rooms.txt": [((2, 2), (8, 8))],
    "hall_way_shifted.txt": [((1, 1), (8, 1)), ((2, 1), (1, 8)), ((3, 1), (8, 8))],
    "full_mr_map.txt": None,  # there are multiple possible orders
}

DOORS_ORDER = {
    "one_room_shifted.txt": [((1, 1), (8, 9))],
    "four_rooms.txt": [((1, 1), (3, 9))],
    "hall_way_shifted.txt": [((1, 1), (9, 4)), ((2, 1), (9, 4)), ((3, 1), (9, 4))],
    "full_mr_map.txt": None,  # there are multiple possible orders
}

# used for display only
ROOM_ORDER = {
    "hall_way_shifted.txt": [(1, 1), (2, 1), (3, 1)],
    "four_rooms.txt": [(1, 1), (2, 1), (2, 2)],
}

class StaticRoom:
    """ Room data which does NOT change during episode (no keys or doors)"""
    def __init__(self, loc, room_size):
        self.loc = loc
        self.size = room_size
        self.map = np.zeros(room_size, dtype=np.uint8)
        self.walls = set()
        # self.keys = set()
        # self.doors = set()
        self.traps = set()
        self.list_generated = False

    def generate_lists(self):
        if self.list_generated:
            raise ValueError("This is supposed to be called only once.")
        self.walls = set()
        # self.keys = set()
        # self.doors = set()
        self.traps = set()

        for x in range(self.size[0]):
            for y in range(self.size[1]):
                if self.map[x, y] != 0:
                    if self.map[x, y] == WALL_CODE:
                        self.walls.add((x, y))
                    # elif self.map[x, y] == KEY_CODE:
                    #     self.keys.add((x, y))
                    # elif self.map[x, y] == DOOR_CODE:
                    #     self.doors.add((x, y))
                    elif self.map[x, y] == TRAP_CODE:
                        self.traps.add((x, y))


tile_size = 10
level_tile_size = 5
hud_height = 10
RENDERING_MODES = ['rgb_array', 'one_hot', "codes"]


class ToyMR(gym.Env):

    def __init__(self, map_file, max_lives=1, absolute_coordinates=False,
                 doors_keys_scale=1, save_enter_cell=True, trap_reward=0.):
        """
        Based on implementation provided here
        https://github.com/chrisgrimm/deep_abstract_q_network/blob/master/toy_mr.py

        Args:
            absolute_coordinates: If use absolute coordinates in observed state.
                E.g. for room = (2, 3), agent = (1, 7), room size = 10, absolute
                coordinates are (2.1, 3.7)
            save_enter_cell: if state should consist enter_cell. Even if set to
                False if max_lives > 1 enter_cell would be encoded into state.
        """
        self.map_file = map_file
        self.max_lives = max_lives

        self.rooms, self.starting_room, self.starting_cell, self.goal_room, self.keys, self.doors = \
            self.parse_map_file(map_file)
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.num_keys = 0

        self.lives = max_lives
        self.enter_cell = self.agent
        self.previous_action = 0
        self.terminal = False
        # self.max_num_actions = max_num_actions
        # self.discovered_rooms = set()
        self.key_neighbor_locs = []
        self.door_neighbor_locs = []
        # self.action_ticker = 0
        self.action_space = Discrete(4)
        # fix order of doors and keys used when cloning / restoring state
        filename = os.path.basename(map_file)
        assert filename in KEYS_ORDER, f"filename {filename} unknown, should " \
            f"be one of {list(KEYS_ORDER.keys())}"
        if KEYS_ORDER[filename] is None:
            self.doors_order = tuple(sorted(self.doors.keys()))
            self.keys_order = tuple(sorted(self.keys.keys()))
        else:
            self.doors_order = DOORS_ORDER[filename]
            self.keys_order = KEYS_ORDER[filename]
        assert all(
            [(key_loc in self.keys) for key_loc in self.keys_order])
        assert all(
            [(door_loc in self.doors) for door_loc in self.doors_order])

        self.absolute_coordinates = absolute_coordinates
        if self.absolute_coordinates:
            self.room_size = self.room.size[0]
            for room in self.rooms.values():
                assert room.size == (self.room_size, self.room_size)
        self.doors_keys_scale = doors_keys_scale
        self.save_enter_cell = save_enter_cell
        self.trap_reward = trap_reward
        np_state = self.get_np_state()
        self.observation_space = Box(low=0, high=255, shape=np_state.shape, dtype=np.uint8)
        self.state_space = self.observation_space  # state == observation


    @property
    def init_kwargs(self):
        return {
            attr: getattr(self, attr)
            for attr in (
                'map_file', 'max_lives', 'absolute_coordinates',
                'doors_keys_scale', 'save_enter_cell', 'trap_reward'
            )
        }

    @staticmethod
    def flood(y, x, symbol, unchecked_sections, whole_room):
        height = len(whole_room)
        width = len(whole_room[0])
        flood_area = {(y, x)}
        to_flood = {(y, x)}
        while to_flood:
            (y, x) = next(iter(to_flood))
            unchecked_sections.remove((y, x))
            to_flood.remove((y, x))
            neighbors = [(y, x) for (y, x) in [(y + 1, x), (y - 1, x), (y, x - 1), (y, x + 1)]
                         if 0 <= x < width and 0 <= y < height and (y, x) in unchecked_sections
                         and whole_room[y][x] == symbol]
            for n in neighbors:
                to_flood.add(n)
                flood_area.add(n)
        return flood_area

    def check_room_abstraction_consistency(self, whole_room, room_number):
        height = len(whole_room)
        width = len(whole_room[0])
        unchecked_sections = set([(y, x) for x in range(width) for y in range(height)
                                  if whole_room[y][x] != '|'])
        symbol_area_mapping = dict()
        while unchecked_sections:
            y, x = next(iter(unchecked_sections))
            symbol = whole_room[y][x]
            flood_area = self.flood(y, x, symbol, unchecked_sections, whole_room)
            if symbol in symbol_area_mapping:
                raise Exception('Improper Abstraction in Room %s with symbol %s' % (room_number, symbol))
            else:
                symbol_area_mapping[symbol] = flood_area

    @property
    def mode(self):
        return "one_hot"

    def get_state_named_tuple(self):
        attrs = dict()

        if not self.absolute_coordinates:
            attrs["agent"] = self.agent
            attrs['.loc'] = self.room.loc
        else:
            attrs["abs_position"] = tuple(
                np.array(self.room.loc) * self.room_size +
                np.array(self.agent)
            )
        attrs['.num_keys'] = (self.num_keys,)
        if self.max_lives > 1 or self.save_enter_cell:
            attrs["enter_cell"] = self.enter_cell

        for i, key_position in enumerate(self.keys_order):
            attrs['key_%s' % i] = (
                self.keys[key_position] * self.doors_keys_scale,
            )

        for i, doors_possition in enumerate(self.doors_order):
            attrs['door_%s' % i] = (
                self.doors[doors_possition] * self.doors_keys_scale,
            )

        attrs["lives"] = (self.lives, )

        attrs["terminal"] = (self.terminal,)

        return tuple(sorted(attrs.items()))

    def room_and_agent_coordinate(self, abs_coordinate: float):
        room_coord = abs_coordinate // self.room_size
        agent_coord = abs_coordinate % self.room_size
        return room_coord, agent_coord

    def room_and_agent_coordinates(self, abs_coordinates: tuple):
        room_0, agent_0 = self.room_and_agent_coordinate(abs_coordinates[0])
        room_1, agent_1 = self.room_and_agent_coordinate(abs_coordinates[1])
        return (room_0, room_1), (agent_0, agent_1)

    def restore_full_state_from_np_array_version(self, state_np, quick=False):
        del quick
        assert state_np.shape == self.observation_space.shape, f"{state_np.shape} {self.observation_space.shape}"
        # We will use this for structure and names, and ignore values
        assert state_np.shape == self.observation_space.shape
        # state_np = state_np[:, 0, 0]  # remove unused dimensions
        state_tuple_template = self.get_state_named_tuple()
        self.agent = None

        ix = 0
        for name, value in state_tuple_template:
            atrr_size = len(value)
            value = state_np[ix: ix + atrr_size]
            ix += atrr_size
            if name == "agent":
                self.agent = tuple(value)
            elif name == ".loc":
                self.room = self.rooms[tuple(value)]
            elif name == "abs_position":
                assert self.absolute_coordinates
                room_coor, agent_coor = \
                    self.room_and_agent_coordinates(tuple(value))
                self.agent = agent_coor
                self.room = self.rooms[room_coor]
            elif name == "enter_cell":
                assert self.max_lives > 1 or self.save_enter_cell, \
                    "enter_cell is meant to be used in state only when " \
                    "max_lives > 1 or when self.save_enter_cell == True"
                self.enter_cell = tuple(value)
            elif name == ".num_keys":
                self.num_keys = value[0]
            elif name.startswith("key"):
                key_number = int(re.match(
                    'key_(?P<key_number>\d*)', name
                ).group("key_number"))
                key_position = self.keys_order[key_number]
                self.keys[key_position] = bool(value[0] // self.doors_keys_scale)
            elif name.startswith("door"):
                door_number = int(re.match(
                    'door_(?P<door_number>\d*)', name
                ).group("door_number"))
                door_position = self.doors_order[door_number]
                self.doors[door_position] = bool(value[0] // self.doors_keys_scale)
            elif name == "lives":
                self.lives = value[0]
            elif name == "terminal":
                self.terminal = value[0]
            else:
                raise NotImplementedError(
                    f"get_state_named_tuple() is not compatible with this "
                    f"method, please update this method, to handle '{name}'"
                )
        assert ix == state_np.size, f"Data missmatch, loaded {ix} numbers, " \
            f"got np_state of size {state_np.size}"

    def get_np_state(self):
        state_tuple = self.get_state_named_tuple()
        state = list()
        for name, val in state_tuple:
            assert isinstance(val, tuple)
            for elem in val:
                assert np.isscalar(elem), f"{elem} {type(elem)}"
                state.append(elem)
        state = np.array(state)
        assert (state >= 0).all()
        assert (state < 256).all()
        state = state.astype(np.uint8)
        return state

    def clone_full_state(self):
        state = self.get_np_state()
        return HashableNumpyArray(state)

    def restore_full_state(self, state):
        assert isinstance(state, HashableNumpyArray)
        state_np = state.get_np_array_version()
        self.restore_full_state_from_np_array_version(state_np)

    @staticmethod
    def parse_map_file(map_file):
        rooms = {}
        keys = {}
        doors = {}

        r = -1
        starting_room, starting_cell, goal_room = None, None, None
        with open(map_file) as f:
            for line in f.read().splitlines():
                if r == -1:
                    room_x, room_y, room_w, room_h = map(int, line.split(' '))
                    room = StaticRoom((room_x, room_y), (room_w, room_h))
                    r = 0
                else:
                    if len(line) == 0:
                        room.generate_lists()
                        rooms[room.loc] = room
                        r = -1
                    elif line == 'G':
                        goal_room = room
                    else:
                        for c, char in enumerate(line):
                            if char == '1':
                                room.map[c, r] = '1'
                            elif char == 'K':
                                # room.map[c, r] = KEY_CODE
                                keys[(room.loc, (c, r))] = True
                            elif char == 'D':
                                # room.map[c, r] = DOOR_CODE
                                doors[(room.loc, (c, r))] = True
                            elif char == 'T':
                                room.map[c, r] = TRAP_CODE
                            elif char == 'S':
                                starting_room = room
                                starting_cell = (c, r)
                        r += 1
        if r >= 0:
            room.generate_lists()
            rooms[room.loc] = room

        if starting_room is None or starting_cell is None:
            raise Exception('You must specify a starting location and goal room')
        return rooms, starting_room, starting_cell, goal_room, keys, doors

    @staticmethod
    def _get_delta(action):
        dx = 0
        dy = 0
        if action == NORTH:
            dy = -1
        elif action == SOUTH:
            dy = 1
        elif action == EAST:
            dx = 1
        elif action == WEST:
            dx = -1
        return dx, dy

    def _move_agent(self, action):
        dx, dy = self._get_delta(action)
        return self.agent[0] + dx, self.agent[1] + dy

    def step(self, action):
        new_agent = self._move_agent(action)
        reward = 0

        # room transition checks
        if (new_agent[0] < 0 or new_agent[0] >= self.room.size[0] or
                new_agent[1] < 0 or new_agent[1] >= self.room.size[1]):
            room_dx = 0
            room_dy = 0

            if new_agent[0] < 0:
                room_dx = -1
                new_agent = (self.room.size[0] - 1, new_agent[1])
            elif new_agent[0] >= self.room.size[0]:
                room_dx = 1
                new_agent = (0, new_agent[1])
            elif new_agent[1] < 0:
                room_dy = -1
                new_agent = (new_agent[0], self.room.size[1] - 1)
            elif new_agent[1] >= self.room.size[1]:
                room_dy = 1
                new_agent = (new_agent[0], 0)

            new_room = self.rooms[(self.room.loc[0] + room_dx, self.room.loc[1] + room_dy)]

            # check intersecting with adjacent door
            if self.doors.get((new_room.loc, new_agent), False):
                if self.num_keys > 0:
                    # new_room.doors.remove(new_agent)
                    self.num_keys -= 1
                    self.doors[(new_room.loc, new_agent)] = False

                    self.room = new_room
                    self.agent = new_agent
                    self.enter_cell = new_agent
            else:
                self.room = new_room
                self.agent = new_agent
                self.enter_cell = new_agent

            if self.room == self.goal_room:
                reward = 1
                self.terminal = True
        else:
            # collision checks
            if self.keys.get((self.room.loc, new_agent), False):
                cell_type = KEY_CODE
            elif self.doors.get((self.room.loc, new_agent), False):
                cell_type = DOOR_CODE
            elif new_agent in self.room.walls:
                cell_type = WALL_CODE
            elif new_agent in self.room.traps:
                cell_type = TRAP_CODE
            else:
                cell_type = 0

            if cell_type == 0:
                self.agent = new_agent
            elif cell_type == KEY_CODE:
                # if self.keys[(self.room.loc, new_agent)]:
                # assert new_agent in self.room.keys
                # self.room.keys.remove(new_agent)
                self.num_keys += 1
                assert (self.room.loc, new_agent) in self.keys
                self.keys[(self.room.loc, new_agent)] = False
                self.agent = new_agent
            elif cell_type == DOOR_CODE:
                if self.num_keys > 0:
                    # assert new_agent in self.room.doors
                    # self.room.doors.remove(new_agent)
                    self.num_keys -= 1
                    self.agent = new_agent
                    assert (self.room.loc, new_agent) in self.doors
                    self.doors[(self.room.loc, new_agent)] = False
            elif cell_type == TRAP_CODE:
                self.lives -= 1
                reward = self.trap_reward
                if self.lives == 0:
                    self.terminal = True
                else:
                    self.agent = self.enter_cell

        # self.action_ticker += 1

        # self.discovered_rooms.add(self.room.loc)
        # return self._get_encoded_room(), reward, self.is_current_state_terminal(), {}
        obs = self.get_np_state()
        done = self.is_current_state_terminal()
        info = {"solved": self.room == self.goal_room}
        return obs, reward, done, info


    def reset(self):
        self.room = self.starting_room
        self.agent = self.starting_cell
        self.num_keys = 0
        self.terminal = False
        # self.action_ticker = 0
        self.lives = self.max_lives
        self.enter_cell = self.agent

        # for room in self.rooms.values():
        #     room.reset()

        for key, val in self.keys.items():
            self.keys[key] = True

        for key, val in self.doors.items():
            self.doors[key] = True

        return self.get_np_state()

    def is_current_state_terminal(self):
        return self.terminal  # or self.action_ticker > self.max_num_actions

    def is_action_safe(self, action):
        new_agent = self._move_agent(action)
        if new_agent in self.room.traps:
            return False
        return True

    def _get_encoded_room(self):
        encoded_room = np.zeros((self.room.size[0], self.room.size[1]), dtype=np.uint8)
        encoded_room[self.agent] = AGENT_CODE
        for coord in self.room.walls:
            encoded_room[coord] = WALL_CODE

        for (room_coord, key_coord), present in self.keys.items():
            if room_coord == self.room.loc and present:
                encoded_room[key_coord] = KEY_CODE

        for (room_coord, door_coord), present in self.doors.items():
            if room_coord == self.room.loc and present:
                encoded_room[door_coord] = DOOR_CODE

        for coord in self.room.traps:
            encoded_room[coord] = TRAP_CODE

        return encoded_room

    def render(self, mode="one_hot"):
        if mode == 'codes':
            return self._get_encoded_room()
        if mode == 'rgb_array':
            return render_screen(self._get_encoded_room(), self.num_keys, self.lives, self.max_lives)
        if mode == "one_hot":
            return self.get_np_state()

    def save_map(self, file_name, tile_size=level_tile_size):
        pygame.init()
        map_h = max(coord[1] for coord in self.rooms)
        map_w = max(coord[0] for coord in self.rooms)

        map_ = pygame.Surface((tile_size * self.room.size[0] * map_w, tile_size * self.room.size[1] * map_h))
        map_.fill(BACKGROUND_COLOR)

        for room_loc in self.rooms:
            print(f"loc {room_loc}, size {self.rooms[room_loc].size}")
            room = self.rooms[room_loc]

            room_x, room_y = room_loc
            room_x = (room_x - 1) * tile_size * room.size[0]
            room_y = (room_y - 1) * tile_size * room.size[1]

            if room == self.goal_room:
                continue
                rect = (room_x, room_y, tile_size * room.size[0], tile_size * room.size[1])
                pygame.draw.rect(map_, (0, 255, 255), rect)

                myfont = pygame.font.SysFont('Helvetica', 8 * tile_size)

                # render text
                label = myfont.render("G", True, (0, 0, 0))
                label_rect = label.get_rect(center=(room_x + (tile_size * room.size[0])/2,
                                                    room_y + (tile_size * room.size[1])/2))
                map_.blit(label, label_rect)
                continue

            # loop through each row
            for row in range(self.room.size[1] + 1):
                pygame.draw.line(map_, GRID_COLOR, (room_x, row * tile_size + room_y),
                                 (room.size[1] * tile_size + room_x, row * tile_size + room_y))
            for column in range(self.room.size[0] + 1):
                pygame.draw.line(map_, GRID_COLOR, (column * tile_size + room_x, room_y),
                                 (column * tile_size + room_x, room.size[0] * tile_size + room_y))

            # draw walls
            for coord in room.walls:
                rect = (coord[0] * tile_size + room_x, coord[1] * tile_size + room_y, tile_size, tile_size)
                pygame.draw.rect(map_, WALL_COLOR, rect)

            # draw key
            for room_coord, key_coord in self.keys.keys():
                if room_coord == room.loc:
                    rect = (key_coord[0] * tile_size + room_x, key_coord[1] * tile_size + room_y, tile_size, tile_size)
                    pygame.draw.rect(map_, KEY_COLOR, rect)

            # # draw doors
            for room_coord, door_coord in self.doors.keys():
                if room_coord == room.loc:
                    rect = (door_coord[0] * tile_size + room_x, door_coord[1] * tile_size + room_y, tile_size, tile_size)
                    pygame.draw.rect(map_, DOOR_COLOR, rect)

            # draw traps
            for coord in room.traps:
                rect = (coord[0] * tile_size + room_x, coord[1] * tile_size + room_y, tile_size, tile_size)
                pygame.draw.rect(map_, TRAP_COLOR, rect)

        pygame.image.save(map_, file_name + '.png')


def draw_circle(screen, coord, color):
    rect = (coord[0] * tile_size, coord[1] * tile_size + hud_height, tile_size, tile_size)
    pygame.draw.ellipse(screen, color, rect)


def draw_rect(screen, coord, color):
    rect = (coord[0] * tile_size, coord[1] * tile_size + hud_height, tile_size, tile_size)
    pygame.draw.rect(screen, color, rect)


def render_screen(code_state, num_keys, lives, max_lives):
    room_size = code_state.shape
    screen = pygame.Surface((room_size[0] * tile_size, room_size[1] * tile_size + hud_height))
    screen.fill(BACKGROUND_COLOR)

    # loop through each row
    for row in range(room_size[1] + 1):
        pygame.draw.line(screen, GRID_COLOR, (0, row * tile_size + hud_height),
                         (room_size[1] * tile_size, row * tile_size + hud_height))
    for column in range(room_size[0] + 1):
        pygame.draw.line(screen, GRID_COLOR, (column * tile_size, hud_height),
                         (column * tile_size, room_size[0] * tile_size + hud_height))

    for index_, x in np.ndenumerate(code_state):
        if x == AGENT_CODE:
            draw_circle(screen, index_, AGENT_COLOR)

        if x == WALL_CODE:
            draw_rect(screen, index_, WALL_COLOR)

        if x == DOOR_CODE:
            draw_rect(screen, index_, DOOR_COLOR)

        if x == TRAP_CODE:
            draw_rect(screen, index_, TRAP_COLOR)

        if x == KEY_CODE:
            draw_rect(screen, index_, KEY_COLOR)

    for i in range(num_keys):
        draw_rect(screen, (i, -1), KEY_COLOR)
    if max_lives > 1:
        for i in range(lives):
            draw_rect(screen, (room_size[0] - 1 - i, -1), LIVES_COLOR)

    image = pygame.surfarray.array3d(screen)
    return image.transpose([1, 0, 2])


class DebugCloneRestoreWrapper(Wrapper):
    """"Performs clone restore operations during step."""
    def __init__(self, env: ToyMR):
        super(DebugCloneRestoreWrapper, self).__init__(env)
        self.second_env = ToyMR(**env.init_kwargs)

    def step(self, action):
        if action < 5: #  and random.random() > 0.5:
            state = self.env.clone_full_state()
            self.second_env.restore_full_state(state)
            assert (state.get_np_array_version() == self.second_env.clone_full_state().get_np_array_version()).all()
            assert state == self.second_env.clone_full_state()
            # swap envs
            self.env, self.second_env = self.second_env, self.env
            print(f"enter cell {self.env.enter_cell}")
        return self.env.step(action)


if __name__ == "__main__":
    import random
    from PIL import Image

    map_file_ = 'mr_maps/four_rooms.txt'  # 'mr_maps/hall_way_shifted.txt' 'mr_maps/four_rooms.txt' 'mr_maps/full_mr_map.txt' 'mr_maps/one_room_shifted.txt'

    env_kwargs = dict(map_file=map_file_, max_lives=5, absolute_coordinates=True)
    # env_kwargs = dict(map_file=map_file_, max_lives=1, save_enter_cell=False,
    #                   absolute_coordinates=True)
    env = ToyMR(**env_kwargs)
    restore_env = ToyMR(**env_kwargs)
    env.save_map("/tmp/map")

    env.reset()
    for i in range(20):
        if i % 3 == 0:
            state = env.clone_full_state()
            restore_env.restore_full_state(state)
            assert (state == restore_env.clone_full_state())
        a = random.randint(0, 3)
        env.step(a)

        ob = env.render(mode="rgb_array")
        im = Image.fromarray(ob)
        im.save(f"/tmp/mr{i}.png")

    env = DebugCloneRestoreWrapper(ToyMR(**env_kwargs))
    # env = ToyMR(map_file_, max_lives=5)
    from gym.utils.play import play
    keys_to_action = {
        (ord('d'),): EAST,
        (ord('a'),): WEST,
        (ord('w'),): NORTH,
        (ord('s'),): SOUTH}

    def callback(obs_t, obs_tp1, action, rew, done, info):
        if done or rew > 0:
            print(f"done {done}, reward {rew}, info {info}")
    # You need to change this line in gym.play to use it nicely:
    # action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)
    # change 0 -> 999  (or some other integer out of ToyMR action space)
    # Without this change gym.play will choose action 0 when no key is pressed.
    play(env, keys_to_action=keys_to_action, fps=10, callback=callback)
