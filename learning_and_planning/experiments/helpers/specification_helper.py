from collections import Mapping, OrderedDict
from itertools import product

from munch import Munch

import copy

from learning_and_planning.experiments.helpers.experiment import Experiment


def create_experiments_helper(experiment_name: str, base_config: dict, params_grid,
                              python_path: str,
                              paths_to_dump: str,
                              exclude=[],
                              update_lambda=lambda d1, d2: d1.update(d2),
                              callbacks=(), env={}):

    params_configurations = get_combinations(params_grid)
    experiments = []

    # Last chance to change something
    for callback in callbacks:
        callback(**locals())
    for params_configuration in params_configurations:
        config = copy.deepcopy(base_config)
        update_lambda(config, params_configuration)
        config = Munch(config)

        experiments.append(Experiment(name=experiment_name,
                                      parameters=config, python_path=python_path,
                                      paths_to_dump=paths_to_dump, env=env,
                                      exclude=exclude))

    return experiments


def get_container_types():
    ret = [list, tuple]
    try:
        import numpy as np
        ret.append(np.ndarray)
    except ImportError:
        pass
    try:
        import pandas as pd
        ret.append(pd.Series)
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

    combinations = []
    for param_grid in param_grids:
        items = param_grid.items()
        if not items:
            combinations.append(OrderedDict())
        else:
            keys___ = []
            grids___ = []
            keys = []
            grids = []

            for key, grid in items:
                if '___' in key:
                    keys___.append(key[:-3])
                    grids___.append(grid)
                else:
                    keys.append(key)
                    grids.append(grid)

            for grid in grids + grids___:
                assert isinstance(grid, allowed_container_types), \
                    'grid values should be passed in one of given types: {}, got {} ({})' \
                        .format(allowed_container_types, type(grid), grid)

            if grids___:
                for param_values___ in zip(*grids___):
                    for param_values in product(*grids):
                        combination = OrderedDict(zip(keys___ + keys, param_values___ + param_values))
                        combinations.append(combination)
            else:
                for param_values in product(*grids):
                    combination = OrderedDict(zip(keys, param_values))
                    combinations.append(combination)

    if limit:
        combinations = combinations[:limit]
    return combinations
