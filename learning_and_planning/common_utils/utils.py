import random
from typing import List

import numpy as np


def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def to_callable(x):
    if isinstance(x, float):
        def f(_):
            return x
        return f
    else:
        assert callable(x)
        return x


def sum_len(lists: List):
    return sum([len(list) for list in lists])


def shuffle_consistently_(arrays: List[List]) -> None:
    indices = list(range(len(arrays[0])))
    random.shuffle(indices)
    for array in arrays:
        array[:] = [array[i] for i in indices]
