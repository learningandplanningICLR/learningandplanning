import os
from collections import Mapping, OrderedDict
from itertools import product


def get_git_head_info():
  import git  # GitPython
  try:
    path = os.getcwd()
    repo = git.Repo(path)
    sha = repo.head.commit.hexsha
    return " ".join(list(repo.remote().urls) + [sha])
  except Exception as e:
    print('error while inferring git path : {}'.format(e))
    return ""


def get_container_types():
  ret = [list, tuple]
  try:
    import numpy as np
    ret.append(np.ndarray)
  except ImportError:
    pass
  try:
    import pandas as pd
    ret.append(pd.core.series.Series)
  except ImportError:
    pass
  return tuple(ret)


def get_combinations(param_grids, limit=None):
  """
  Based on sklearn code for grid search. Get all hparams combinations based on
  grid(s).
  :param param_grids: dict representing hparams grid, or list of such
  mappings
  :returns: list of OrderedDict (if params_grids consisted OrderedDicts,
   the Order of parameters will be sustained.)
  """
  allowed_container_types = get_container_types()
  if isinstance(param_grids, Mapping):
    # Wrap dictionary in a singleton list to support either dict or list of
    # dicts.
    param_grids = [param_grids]

  combinations = list()
  for param_grid in param_grids:
    items = param_grid.items()
    if not items:
      combinations.append(OrderedDict())
    else:
      keys, grids = zip(*param_grid.items())
      for grid in grids:
        assert isinstance(grid, allowed_container_types), \
          'grid values should be passed in one of given types: {}, got {} ({})'\
            .format(allowed_container_types, type(grid), grid)
      for param_values in product(*grids):
        combination = OrderedDict(zip(keys, param_values))
        combinations.append(combination)

  if limit:
    combinations = combinations[:limit]
  return combinations

