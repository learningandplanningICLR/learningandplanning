from ctypes import c_double, c_int, c_bool, cdll
import numpy as np
import numpy.ctypeslib as npct

lib = cdll.LoadLibrary('./room_utils_fast.dylib')

lib.generate_room.argtypes = [npct.ndpointer(dtype=np.uint8, ndim=1),
    npct.ndpointer(dtype=np.int32, ndim=1), c_double, c_int, c_int, c_int, c_int, c_bool]

#void generate_room(int result[max_size*max_size],
#                  int d[2], 
#                  double p_change_directions, 
#                  int num_steps, 
#                  int nb,  // num boxes
#                  int tries, 
#                  int seed,
#                  bool second_player) {

res = np.zeros(100, dtype=np.uint8)
dims = np.array([10,10], dtype=np.int32)
lib.generate_room(res, dims, 0.35, 25, 4, 4, 1234567, False)
res = res.reshape((10,10))
print(res)
