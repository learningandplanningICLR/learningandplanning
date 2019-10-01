from abc import ABC, abstractmethod
import gin
import numpy as np
import random
from gym_sokoban_fast import SokobanEnvFast
from learning_and_planning.mcts.ensemble_configurator import EnsembleConfigurator


class GameMaskProcessor(ABC):

    @property
    def mask_size(self):
        raise NotImplementedError()

    @abstractmethod
    def process_game(self, game):
        pass

    def one_hot_mask(self, num):
        mask = np.zeros(self.mask_size)
        mask[num] = 1.0
        return mask

class DummyGameMaskProcessor(GameMaskProcessor):

    @property
    def mask_size(self):
        return ()

    def process_game(self, game):
        # Adding dummy 0.0
        return [(state, value, action, 0.33) for state, value, action in game]


@gin.configurable
class RandomEnsembleTupleMaskProcessor(GameMaskProcessor):

    def __init__(self):
        self.number_of_ensembles = EnsembleConfigurator().num_ensembles

    @property
    def mask_size(self):
        return (self.number_of_ensembles, )

    def process_game(self, game):

        return [(state, value, action, self.one_hot_mask(random.randint(0, self.number_of_ensembles-1)))
                for state, value, action in game]


@gin.configurable
class RandomEnsembleGameMaskProcessor(GameMaskProcessor):

    def __init__(self):
        self.number_of_ensembles = EnsembleConfigurator().num_ensembles

    @property
    def mask_size(self):
        return (self.number_of_ensembles, )

    def process_game(self, game):
        ensemble_num = random.randint(0, self.number_of_ensembles-1)
        mask = self.one_hot_mask(ensemble_num)

        return [(state, value, action, mask) for state, value, action in game]


@gin.configurable
class BernoulliMask(GameMaskProcessor):

    def __init__(self, number_of_ensembles, p, one_mask_per_game=True):
        self.number_of_ensembles = number_of_ensembles
        self.p = p
        self.one_mask_per_game = one_mask_per_game

    @property
    def mask_size(self):
        return (self.number_of_ensembles, )

    def process_game(self, game):
        if self.one_mask_per_game:
            mask = np.random.binomial(1, self.p, self.mask_size)
            return [(state, value, action, mask) for state, value, action in game]
        else:
            return [
                (
                    state, value, action,
                    np.random.binomial(1, self.p, self.mask_size)
                )
                for state, value, action in game
            ]


@gin.configurable
class SokobanFastEnvDispatcherGameMaskProcessor(GameMaskProcessor):

    def __init__(self, hash_to_num_fn):
        self.number_of_ensembles = EnsembleConfigurator().num_ensembles
        self.hash_to_num_fn = hash_to_num_fn
        self._processing_env = SokobanEnvFast()

    @property
    def mask_size(self):
        return (self.number_of_ensembles, )

    def process_game(self, game):
        first_state_np = game[0][0]
        game = game[1:]  # The first entry might be fake
        self._processing_env.restore_full_state_from_np_array_version(first_state_np)
        initial_game_hash = hash(self._processing_env.clone_full_state())
        ensemble_num = self.hash_to_num_fn(initial_game_hash, self.number_of_ensembles)
        mask = self.one_hot_mask(ensemble_num)

        return [(state, value, action, mask) for state, value, action in game]


@gin.configurable
class RandomGameMask(GameMaskProcessor):

    def __init__(self, number_of_ensembles):
        self.number_of_ensembles = number_of_ensembles

    @property
    def mask_size(self):
        return (self.number_of_ensembles, )

    def process_game(self, game):
        mask = np.zeros(self.mask_size)
        game_label = np.random.randint(self.number_of_ensembles)
        mask[game_label] = 1.0
        return [(state, value, action, mask) for state, value, action in game]


@gin.configurable
class ConstantEnsembleMask(GameMaskProcessor):

    def __init__(self, number_of_ensembles):
        self.number_of_ensembles = number_of_ensembles

    @property
    def mask_size(self):
        return (self.number_of_ensembles, )

    def process_game(self, game):
        mask = np.zeros(self.mask_size)
        # game_label = 1
        mask[:] = 1.0
        return [(state, value, action, mask) for state, value, action in game]
