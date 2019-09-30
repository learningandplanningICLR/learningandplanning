import random

from learning_and_planning.evaluation.solvers import ActionByActionSolver


class TrivialSolver(ActionByActionSolver):
    def choose_action(self, obs, env):
        return 0


class RandomSolver(ActionByActionSolver):
    def __init__(self, num_actions=4):
        self.num_actions = num_actions

    def choose_action(self, obs, env):
        return random.randint(0, self.num_actions - 1)
