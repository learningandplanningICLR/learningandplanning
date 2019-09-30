import os
try:
  from mpi4py import MPI
except ImportError:
  MPI = None

from baselines import logger
from baselines.bench import Monitor

from learning_and_planning.gym_sokoban_tweaks.sokoban_envs import CustomSokobanEnv
from learning_and_planning.gym_sokoban_tweaks.save_restore_wrapper import SaveRestoreWrapper
from learning_and_planning.common_utils.subproc_vec_env_save_restore import SubprocVecEnvCloneRestore

def make_vec_env(args, num_env, seed, wrapper_kwargs=None, start_index=0, reward_scale=1.0, **kwargs):
  mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
  def make_env(rank, args): # pylint: disable=C0111
    def _thunk():
      env = CustomSokobanEnv(args)
      env = SaveRestoreWrapper(env, **kwargs)
      env.seed(seed + 10000*mpi_rank + rank if seed is not None else None)
      env = Monitor(env,
                    logger.get_dir() and os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(rank)),
                    allow_early_resets=True)
      return env
    return _thunk
  #set_global_seeds(seed)
  return SubprocVecEnvCloneRestore([make_env(i + start_index, args) for i in range(num_env)])
