from gym import Wrapper
import random
import gin
import copy

from gym_sokoban_fast import SokobanEnvFast


class FiniteNumberOfGamesWrapper(Wrapper):

    def __init__(self, env, number_of_games):
        super().__init__(env)
        self.number_of_games = number_of_games
        self.states = []

    def reset(self, **kwargs):
        if len(self.states) < self.number_of_games:
            ob = self.env.reset()
            state = self.env.clone_full_state()
            self.states.append((ob, state))
        else:
            ob, state = random.choice(self.states)
            state_ = copy.deepcopy(state)
            self.env.restore_full_state(state_)

        return ob

    # Just to make 'one_hot' the default
    def render(self, mode='one_hot', **kwargs):
        return self.env.render(mode=mode, **kwargs)


@gin.configurable
def sokoban_with_finite_number_of_games(number_of_games):
    env = SokobanEnvFast()
    env = FiniteNumberOfGamesWrapper(env, number_of_games)

    return env