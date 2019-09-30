import random

import numba
import numpy as np
import pkg_resources
import marshal
from scipy import misc


def get_room_state_and_structure(flat_state, dim_room):
    """Splits flat room state into dynamic state and room structure.

    Args:
        flat_state: Flat state got from SokobanEnv.clone_full_state.
        dim_room: Pair (room_height, room_width).

    Returns:
        Pair (room_state, room_structure) of arrays of shape dim_room.
    """
    state = flat_state.astype(np.int8)
    state = state[:-1]
    split = len(state) // 2
    return tuple(
        x.reshape(dim_room) for x in (state[split:], state[:split])
    )


def make_standalone_state(room, room_structure):
    """Incorporates room structure into the room state.

    Args:
        room: Room state (changing between timesteps).
        room_structure: Room structure (fixed for the entire episode). If None,
            returns room.

    Returns:
        Room state with structure information.
    """
    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 5) & (room_structure == 2)] = 6
    return room


def room_to_rgb(room, room_structure=None, surfaces=None):
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    room = make_standalone_state(room, room_structure)
    img_shape = surfaces[0].shape

    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * img_shape[0], room.shape[1] * img_shape[1], 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * img_shape[0]

        for j in range(room.shape[1]):
            y_j = j * img_shape[1]
            surfaces_id = room[i, j]

            room_rgb[x_i:(x_i + img_shape[0]), y_j:(y_j + img_shape[1]), :] = surfaces[surfaces_id]

    return room_rgb


def room_to_tiny_world_rgb(room, room_structure=None, surfaces=None, scale=1):
    room = make_standalone_state(room, room_structure)

    # Assemble the new rgb_room, with all loaded images
    room_small_rgb = np.zeros(shape=(room.shape[0]*scale, room.shape[1]*scale, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * scale
        for j in range(room.shape[1]):
            y_j = j * scale
            surfaces_id = int(room[i, j])
            room_small_rgb[x_i:(x_i+scale), y_j:(y_j+scale), :] = np.array(surfaces[surfaces_id])

    return room_small_rgb


@numba.jit(nopython=True)
def array_2d_to_one_hot(array, nvalues=7):
  array_one_hot = np.zeros(shape=(array.shape[0], array.shape[1], nvalues),
                          dtype=np.uint8)
  for i in range(array.shape[0]):
    for j in range(array.shape[1]):
      surfaces_id = int(array[i, j])
      array_one_hot[i, j, surfaces_id] = 1
  return array_one_hot


def room_to_one_hot(room, room_structure=None):
    room = make_standalone_state(room, room_structure)
    return array_2d_to_one_hot(room, 7)


def room_to_binary_map(room, room_stucture=None):
    room = make_standalone_state(room, room_stucture)
    ret = np.zeros([room.shape[0], room.shape[1], 4], dtype=np.uint8)
    # interior state, no walls
    ret[:, :, 0] = room != 0
    # target (unoccupied, with box or with player)
    ret[:, :, 1] = (room == 2) | (room == 3) | (room == 6)
    # box (on empty tile or on target)
    ret[:, :, 2] = (room == 3) | (room == 4)
    # player (on empty tile or on target)
    ret[:, :, 3] = (room == 5) | (room == 6)
    return ret


def binary_map_to_room(binary_map):
    """Inverse function of room_to_binary_map."""
    (height, width, _) = binary_map.shape
    room = np.zeros((height, width), dtype=np.uint8)
    def set_binary(bits, category):
        # Set room state to a given category wherever given bits are set in
        # binary_map.
        mask = np.ones((height, width), dtype=np.bool)
        for (position, bit) in enumerate(bits):
            mask &= binary_map[:, :, position] == bit
        room[mask] = category

    # wall
    set_binary((0, 0, 0, 0), 0)
    # empty tile
    set_binary((1, 0, 0, 0), 1)
    # box on empty tile
    set_binary((1, 0, 1, 0), 4)
    # player on empty tile
    set_binary((1, 0, 0, 1), 5)
    # target
    set_binary((1, 1, 0, 0), 2)
    # box on target
    set_binary((1, 1, 1, 0), 3)
    # player on target
    set_binary((1, 1, 0, 1), 6)

    return room


def room_to_rgb_FT(room, box_mapping, room_structure=None, surfaces=None):
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    room = make_standalone_state(room, room_structure)

    img_shape = surfaces[0].shape

    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * img_shape[0], room.shape[1] * img_shape[1], 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * img_shape[0]

        for j in range(room.shape[1]):
            y_j = j * img_shape[1]

            surfaces_id = room[i, j]
            surface = surfaces[surfaces_id]
            if 1 < surfaces_id < 5:
                try:
                    surface = get_proper_box_surface(surfaces_id, box_mapping, i, j)
                except:
                    pass
            room_rgb[x_i:(x_i + img_shape[0]), y_j:(y_j + img_shape[1]), :] = surface

    return room_rgb


def get_proper_box_surface(surfaces_id, box_mapping, i, j):
    names = ["wall", "floor", "box_target", "box_on_target", "box", "player", "player_on_target"]

    box_id = 0
    situation = ''

    if surfaces_id == 2:
        situation = '_target'
        box_id = list(box_mapping.keys()).index((i, j))
    elif surfaces_id == 3:
        box_id = list(box_mapping.values()).index((i, j))
        box_key = list(box_mapping.keys())[box_id]
        if box_key == (i, j):
            situation = '_on_target'
        else:
            situation = '_on_wrong_target'
        pass
    elif surfaces_id == 4:
        box_id = list(box_mapping.values()).index((i, j))

    surface_name = 'box{}{}.png'.format(box_id, situation)
    resource_package = __name__
    filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'multibox', surface_name)))
    surface = misc.imread(filename)

    return surface


def room_to_tiny_world_rgb_FT(room, box_mapping, room_structure=None, surfaces=None, scale=1):
        room = make_standalone_state(room, room_structure)

        # Assemble the new rgb_room, with all loaded images
        room_small_rgb = np.zeros(shape=(room.shape[0] * scale, room.shape[1] * scale, 3), dtype=np.uint8)
        for i in range(room.shape[0]):
            x_i = i * scale
            for j in range(room.shape[1]):
                y_j = j * scale

                surfaces_id = int(room[i, j])
                surface = np.array(surfaces[surfaces_id])
                if 1 < surfaces_id < 5:
                    try:
                        surface = get_proper_tiny_box_surface(surfaces_id, box_mapping, i, j)
                    except:
                        pass
                room_small_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = surface

        return room_small_rgb


def get_proper_tiny_box_surface(surfaces_id, box_mapping, i, j):

    box_id = 0
    situation = 'box'

    if surfaces_id == 2:
        situation = 'target'
        box_id = list(box_mapping.keys()).index((i, j))
    elif surfaces_id == 3:
        box_id = list(box_mapping.values()).index((i, j))
        box_key = list(box_mapping.keys())[box_id]
        if box_key == (i, j):
            situation = 'on_target'
        else:
            situation = 'on_wrong_target'
        pass
    elif surfaces_id == 4:
        box_id = list(box_mapping.values()).index((i, j))

    surface = [255, 255, 255]
    if box_id == 0:
        if situation == 'target':
            surface = [111, 127, 232]
        elif situation == 'on_target':
            surface = [6, 33, 130]
        elif situation == 'on_wrong_target':
            surface = [69, 81, 122]
        else:
            # Just the box
            surface = [11, 60, 237]

    elif box_id == 1:
        if situation == 'target':
            surface = [195, 127, 232]
        elif situation == 'on_target':
            surface = [96, 5, 145]
        elif situation == 'on_wrong_target':
            surface = [96, 63, 114]
        else:
            surface = [145, 17, 214]

    elif box_id == 2:
        if situation == 'target':
            surface = [221, 113, 167]
        elif situation == 'on_target':
            surface = [140, 5, 72]
        elif situation == 'on_wrong_target':
            surface = [109, 60, 71]
        else:
            surface = [239, 0, 55]

    elif box_id == 3:
        if situation == 'target':
            surface = [247, 193, 145]
        elif situation == 'on_target':
            surface = [132, 64, 3]
        elif situation == 'on_wrong_target':
            surface = [94, 68, 46]
        else:
            surface = [239, 111, 0]

    return surface


def color_player_two(room_rgb, position, room_structure):
    resource_package = __name__

    player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'multiplayer', 'player1.png')))
    player = misc.imread(player_filename)

    player_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                '/'.join(('surface', 'multiplayer', 'player1_on_target.png')))
    player_on_target = misc.imread(player_on_target_filename)

    x_i = position[0] * 16
    y_j = position[1] * 16

    if room_structure[position[0], position[1]] == 2:
        room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = player_on_target

    else:
        room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = player

    return room_rgb


def color_tiny_player_two(room_rgb, position, room_structure, scale = 4):

    x_i = position[0] * scale
    y_j = position[1] * scale

    if room_structure[position[0], position[1]] == 2:
        room_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = [195, 127, 232]

    else:
        room_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = [96, 5, 145]

    return room_rgb


SURFACES = None


def render_state(state, tiny=False):
    # To avoid circular import.
    from gym_sokoban.envs import SokobanEnv

    # Cache the surfaces to avoid reloading.
    if SURFACES is None:
        globals()['SURFACES'] = SokobanEnv.load_surfaces()

    if tiny:
        render_fn = room_to_tiny_world_rgb
        surface_name = 'tiny_rgb_array'
    else:
        render_fn = room_to_rgb
        surface_name = 'rgb_array'
    return render_fn(state, surfaces=SURFACES[surface_name])


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
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
