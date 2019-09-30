import copy
from ctypes import c_double, c_int, c_uint, c_bool, cdll, byref, POINTER
import pkg_resources
import numpy as np
import numpy.ctypeslib as npct
from gym.utils.seeding import _int_list_from_bigint, hash_seed
import os

from sys import platform

from ourlib.distributed_utils import get_mpi_rank_or_0

if platform == "linux" or platform == "linux2":
    ext = 'so'
    if get_mpi_rank_or_0() != 0:
      #This is a hack to let the rank 0 node to compile
      import time
      time.sleep(60)

    try:
      lib_filename = pkg_resources.resource_filename(__name__, './room_utils_fast.' + ext)
      lib = cdll.LoadLibrary(lib_filename)
    except:
      #May help with verions problems

      print("loading failed. Try to recompile")
      lib_path = os.path.dirname(os.path.abspath(__file__))
      command = "cd {};g++ -std=c++11 -c -fPIC -O3 room_utils_fast.cpp -o room_utils_fast.o".format(lib_path)
      os.system(command)
      command = "cd {};g++ -shared -Wl,-soname,room_utils_fast.so -o room_utils_fast.so room_utils_fast.o".format(lib_path)
      os.system(command)
      # Try once more
      lib_filename = pkg_resources.resource_filename(__name__, './room_utils_fast.' + ext)
      lib = cdll.LoadLibrary(lib_filename)

elif platform == "darwin":
    ext = 'dylib'
    lib_filename = pkg_resources.resource_filename(__name__, './room_utils_fast.' + ext)
    lib = cdll.LoadLibrary(lib_filename)

lib.generate_room.argtypes = [npct.ndpointer(dtype=np.uint8, ndim=1),  # room
                              npct.ndpointer(dtype=np.int32, ndim=1),  # dims
                              c_double,  # p_change_directions
                              c_int,  # num_steps
                              c_int,  # num_boxes
                              c_int,  # tries
                              POINTER(c_uint),  # seed
                              c_bool,  # do_reverse_playing
                              c_bool,
                              c_int]  # second_player

class SokobanRoomGenerator(object):
    def __init__(self, seed, game_mode=None, verbose=True):
        assert game_mode in ["NoAlice", "Alice", "Magnetic"], "Incorrect game format!"
        self.do_reverse_playing = c_bool(game_mode == "NoAlice" or game_mode == "Magnetic")
        self.seed = c_uint(seed)
        self.verbose = verbose

    def generate_room(self, dim=(13, 13),
                      p_change_directions=0.35,
                      num_steps=25,
                      num_boxes=3,
                      tries=4,
                      second_player=False,
                      curriculum=300):
        """
        Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
        wall = 0
        empty space = 1
        box target = 2
        box not on target = 3
        box on target = 4
        player = 5

        :param dim:
        :param p_change_directions:
        :param num_steps:
        :return: Numpy 2d Array
        """

        room_state = np.zeros(dim[0]*dim[1], dtype=np.uint8)
        dim = np.array(dim, dtype=np.int32)

        seed_ = copy.copy(self.seed)
        score = lib.generate_room(room_state,
                                  dim,
                                  p_change_directions,
                                  num_steps,
                                  num_boxes,
                                  tries,
                                  byref(seed_),
                                  self.do_reverse_playing,
                                  second_player,
                                  c_int(curriculum))

        # rehash the seed returned from generate_room
        self.seed = c_uint(_int_list_from_bigint(hash_seed(self.seed))[1])

        # error codes from generate_room
        # -3: BAD, "Not enough free spots (#%d) to place %d player and %d boxes." (place_box_and_players)
        # -2: BAD, "More boxes (%d) than allowed (%d)!"
        # -1: BAD "Sokoban size (%dx%d) bigger than maximum size (%dx%d)!"
        # score: OK
        if score == -3:
            raise RuntimeError('Not enough free spots to place player and boxes.')
        elif score == -2:
            raise RuntimeError('More boxes ({}) than allowed!'.format(num_boxes))
        elif score == -1:
            raise RuntimeError('Sokoban size ({}x{}) bigger than maximum size!'.format(dim[0], dim[1]))
        elif score == 0:
            if self.do_reverse_playing:
                msg = 'Generated Model with score == 0' if self.verbose else ""
                raise RuntimeWarning(msg)

        room_state = room_state.reshape((dim[0], dim[1]))
        player_on_target = room_state == 6

        room_structure = room_state.copy()
        if player_on_target.any():
            room_state[player_on_target] = 5
            room_structure[player_on_target] = 2
        room_structure[room_structure > 3] = 1
        room_structure[room_structure == 3] = 2

        return room_structure, room_state, {}, np.uint32(self.seed)


